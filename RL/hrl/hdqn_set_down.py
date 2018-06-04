"""Main DQN agent."""

import tensorflow as tf
from keras.layers import (Activation, Dense, Flatten, merge, Input, Multiply)
from keras.models import Model, Sequential
import numpy as np
import copy
import inspect

from RL.hrl.objectives import mean_huber_loss
from RL.hrl.memory import ReplayMemory
import RL.hrl.utils as utils

# default hyper-parameters for DQN
DEBUG = 0
TENSORBOARD_PLOTS = 1
MATLAB_PLOTS = 1
GAMMA = 0.99
ALPHA = 25e-5  # finished
NUM_EPISODES = 12000  # finished
REPLAY_BUFFER_SIZE = 1000000  # not specified, found in Ethan's code
BATCH_SIZE = 16  # not specified
TARGET_UPDATE_FREQ = 1000  # not specified
NUM_BURNIN = 1000  # not specified
SAVE_FREQ = 1000  # model save freq
ANNEAL_NUM_STEPS = 50000  # finished for the metacontroller, check what adaptive anneal means for the controller
EVAL_FREQ = 1,
EVAL_NUM_EPISODES = 10  # finished
OPTIMIZER = 'sgd'
TESTING = False


def create_deep_model(input_shape, num_outputs, act_func='relu'):
	input_state = [Input(input_shape[i]) for i in range(len(input_shape))]
	input_mask = Input(shape=(num_outputs,))

	# does it flatten the full list?
	flat_input_state = [Flatten()(input_state[i]) for i in range(len(input_shape))]
	if len(flat_input_state) > 1:
		merged_input_state = merge(flat_input_state, mode='concat')
	else:
		merged_input_state = flat_input_state[0]

	b1 = Dense(500)(merged_input_state)
	b2 = Activation('sigmoid')(b1)

	# c1 = Dense(100)(b2)
	# c2 = Activation('sigmoid')(c1)

	e1 = Dense(num_outputs)(b2)
	e2 = Activation('linear')(e1)

	f = Multiply()([e2, input_mask])
	model = Model(inputs=input_state + [input_mask], outputs=[f])

	return model


def save_scalar(step, name, value, writer):
	"""Save a scalar value to tensorboard.

	Parameters
	----------
	step: int
	Training step (sets the position on x-axis of tensorboard graph.
	name: str
	Name of variable. Will be the name of the graph in tensorboard.
	value: float
	The value of the variable at this step.
	writer: tf.FileWriter
	The tensorboard FileWriter instance.
	"""
	summary = tf.Summary()
	summary_value = summary.value.add()
	summary_value.simple_value = float(value)
	summary_value.tag = name
	writer = writer.add_summary(summary, step)


class HQNAgent:
	# learning module, used for the controller and metacontroller
	class Module:

		def __init__(self,
		             module_type,
		             state_shape,
		             num_choices,
		             burnin_policy,
		             training_policy,
		             testing_policy,
		             num_burnin=NUM_BURNIN,
		             gamma=GAMMA,
		             optimizer=OPTIMIZER,
		             loss_function=mean_huber_loss,
		             target_update_freq=TARGET_UPDATE_FREQ,
		             batch_size=BATCH_SIZE,
		             mem_size=REPLAY_BUFFER_SIZE,
		             model_dir=None,
		             exp_name='None'):

			self.exp_name = exp_name

			# network parameters
			self.module_type = module_type
			self.state_shape = state_shape
			self.num_choices = num_choices

			# learning parameters
			# the discounting reward factor
			self.gamma = gamma

			# the learning rate
			# self.alpha = alpha

			# the tensorflow optimizer to be used
			self.optimizer = optimizer
			# the loss function to be minimized
			self.loss_function = loss_function

			# after how many updates the target network will be synced
			self.target_update_freq = target_update_freq
			# umber of samples used to initialize the memory
			self.num_burnin = num_burnin
			# the batch_size for the parameters update
			self.batch_size = batch_size

			# agent's policies
			self.burnin_policy = burnin_policy
			self.testing_policy = testing_policy
			self.training_policy = training_policy

			# auxiliary variables for the training

			# number of parameter updates
			self.num_updates = 0
			# number of interactions with the learning samples
			self.num_samples = 0
			# number of episodes used for training
			self.num_train_episodes = 0

			# modules's components

			# the network
			self.network = create_deep_model(input_shape=self.state_shape, num_outputs=self.num_choices,
			                                 act_func='relu')
			self.target_network = create_deep_model(input_shape=self.state_shape, num_outputs=self.num_choices,
			                                        act_func='relu')

			if model_dir:
				self.network.load_weights(model_dir)
				self.target_network.load_weights(model_dir)

			self.network.compile(loss=self.loss_function, optimizer=self.optimizer)

			# the replay memory
			self.memory = ReplayMemory(mem_size)

			# tensorboard logistics
			# self.writer = tf.summary.FileWriter('./logs_' + module_type + '/' + self.exp_name)

		def calc_q_values(self, states):
			"""Given a state (or batch of states) calculate the Q-values.
			states: list of states
			choices:list with the mask of choices for each state
			Return
			------
			Q-values for the state(s)
			"""

			batch_size = len(states)
			states_batch = [np.zeros((batch_size,) + self.state_shape[i]) for i in range(len(self.state_shape))]

			for idx in range(batch_size):
				for in_idx in range(len(self.state_shape)):
					assert states[idx][in_idx].shape == self.state_shape[in_idx]
					states_batch[in_idx][idx] = states[idx][in_idx]

			q_values = self.network.predict(states_batch + [np.ones((batch_size, self.num_choices))], batch_size=1)
			assert q_values.shape == (batch_size, self.num_choices)

			return q_values

		def select(self, policy, **kwargs):
			"""Select the output(goal/action) based on the current state.
			Returns
			--------
			selected choice (goal or action)
			"""
			policy_args = inspect.getargspec(policy.select)[0]
			# print 'Policy args'
			# print policy_args

			if len(policy_args) > 1:
				if 'num_update' in policy_args:
					# num_samples is current iteration, for choosing the epsilon
					choice, q = policy.select(self.calc_q_values([kwargs['state']]), self.num_samples)

				else:
					choice, q = policy.select(self.calc_q_values([kwargs['state']]))

			else:
				choice, q = policy.select()

			assert 0 <= choice < self.num_choices

			return choice, q

		def update_policy(self, agent_writer=None):
			"""Updates the modules's policy.
			"""

			# sample the memory replay to get samples of experience <state, action, reward, next_state, is_terminal>
			exp_samples = self.memory.sample(self.batch_size)
			assert len(exp_samples) == self.batch_size

			# process the experience samples

			# try this: state_batch=[np.zeros((self.batch_size, )+self.state_shape[i][1:]) for i in range(len(self.state_shape))]
			state_batch = [np.zeros((self.batch_size,) + self.state_shape[i]) for i in range(len(self.state_shape))]

			# the next state-input batches
			next_state_batch = [np.zeros((self.batch_size,) + self.state_shape[i]) for i in
			                    range(len(self.state_shape))]

			# input mask needed to chose only one q-value
			mask_batch = np.zeros((self.batch_size, self.num_choices))

			# the q-value which corresponds to the applied action of the sample
			target_q_batch = np.zeros((self.batch_size, self.num_choices))

			for sample_idx in range(self.batch_size):
				for in_idx in range(len(self.state_shape)):
					assert exp_samples[sample_idx].state[in_idx].shape == self.state_shape[in_idx]
					assert exp_samples[sample_idx].next_state[in_idx].shape == self.state_shape[in_idx]

					state_batch[in_idx][sample_idx] = exp_samples[sample_idx].state[in_idx]
					next_state_batch[in_idx][sample_idx] = exp_samples[sample_idx].next_state[in_idx]

				# activate the output of the applied action
				mask_batch[sample_idx, exp_samples[sample_idx].action] = 1

			# on the next state, chose the best predicted q-value on the fixed-target network
			predicted_q_batch = self.target_network.predict(
				next_state_batch + [np.ones((self.batch_size, self.num_choices))], batch_size=self.batch_size)

			assert predicted_q_batch.shape == (self.batch_size, self.num_choices)

			best_q_batch = np.amax(predicted_q_batch, axis=1)
			assert best_q_batch.shape == (self.batch_size,)

			# compute the target q-value r+gamma*max{a'}(Q(nextstat,a',qt)
			for sample_idx in range(self.batch_size):
				target_q_batch[sample_idx, exp_samples[sample_idx].action] = exp_samples[sample_idx].reward
				if not exp_samples[sample_idx].terminal:
					target_q_batch[sample_idx, exp_samples[sample_idx].action] = exp_samples[
						                                                             sample_idx].reward + self.gamma *\
					                                                                                      best_q_batch[
						                                                                                      sample_idx]

			loss = self.network.train_on_batch(x=state_batch + [mask_batch], y=target_q_batch)

			# save_scalar(self.num_updates, 'Loss for {0}'.format(self.module_type), loss, self.writer)
			# if agent_writer is not None:
			# 	save_scalar(self.num_updates, 'Loss for {0}'.format(self.module_type), loss, agent_writer)

			# update the target network
			if self.num_updates > 0 and self.num_updates % self.target_update_freq == 0:
				utils.get_hard_target_model_updates(self.target_network, self.network)

		def save_model(self):
			self.network.save_weights('model/{0}_source_{1}_{2}.weight'.format(self.module_type, self.exp_name,
			                                                                   self.num_updates))
			# self.target_network.save_weights('model/{0}_target_{1}.weight'.format(self.module_type, self.exp_name))

	"""Class implementing Hierarchical Q-learning Agent.

	"""

	def __init__(self,
	             state_shape, # state[0] is the history  length
	             goal_shape,
	             num_actions,
	             num_goals,
	             controller_training_policy,
	             metacontroller_training_policy,
	             controller_testing_policy,
	             metacontroller_testing_policy,
	             controller_burnin_policy,
	             metacontroller_burnin_policy,
	             controller_gamma=GAMMA,
	             metacontroller_gamma=GAMMA,
	             # controller_alpha=ALPHA,
	             # metacontroller_alpha=ALPHA,
	             controller_target_update_freq=TARGET_UPDATE_FREQ,
	             metacontroller_target_update_freq=TARGET_UPDATE_FREQ,
	             controller_batch_size=BATCH_SIZE,
	             metacontroller_batch_size=BATCH_SIZE,
	             controller_optimizer='sgd',
	             metacontroller_optimizer='sgd',
	             controller_loss_function=mean_huber_loss,
	             metacontroller_loss_function=mean_huber_loss,
	             eval_freq=EVAL_FREQ,
	             controller_num_burnin=NUM_BURNIN,
	             metacontroller_num_burnin=NUM_BURNIN,
	             replay_buffer_size=REPLAY_BUFFER_SIZE,
	             controller_dir=None,
	             meta_controller_dir=None,
	             exp_name='None',
	             save_freq=SAVE_FREQ,
	             set_down_model_dir = None,
	             fit_env = None
	             ):

		# env setup:
		self.fit_env = fit_env

		# agent's description
		self.state_shape = state_shape
		self.goal_shape = goal_shape
		self.num_actions = num_actions
		self.num_goals = num_goals
		self.exp_name = exp_name

		# agent's parameters
		self.eval_freq = eval_freq
		self.save_freq = save_freq

		# agent's learning modules
		self.controller = self.Module(module_type='controller',
		                              state_shape=[self.state_shape, self.goal_shape],
		                              num_choices=num_actions,
		                              burnin_policy=controller_burnin_policy,
		                              training_policy=controller_training_policy,
		                              testing_policy=controller_testing_policy,
		                              gamma=controller_gamma,
		                              # alpha=controller_alpha,
		                              optimizer=controller_optimizer,
		                              loss_function=controller_loss_function,
		                              batch_size=controller_batch_size,
		                              num_burnin=controller_num_burnin,
		                              target_update_freq=controller_target_update_freq,
		                              mem_size=replay_buffer_size,
		                              model_dir=controller_dir,
		                              exp_name=self.exp_name)

		self.metacontroller = self.Module(module_type='metacontroller',
		                                  state_shape=[self.state_shape],
		                                  num_choices=num_goals,
		                                  burnin_policy=metacontroller_burnin_policy,
		                                  training_policy=metacontroller_training_policy,
		                                  testing_policy=metacontroller_testing_policy,
		                                  gamma=metacontroller_gamma,
		                                  # alpha=metacontroller_alpha,
		                                  optimizer=metacontroller_optimizer,
		                                  loss_function=metacontroller_loss_function,
		                                  batch_size=metacontroller_batch_size,
		                                  num_burnin=metacontroller_num_burnin,
		                                  target_update_freq=metacontroller_target_update_freq,
		                                  mem_size=replay_buffer_size,
		                                  model_dir=meta_controller_dir,
		                                  exp_name=self.exp_name)

		if set_down_model_dir:
			# prepare external set-down module
			self.set_down_model = Sequential()
			self.set_down_model.add(Flatten(input_shape=(1,) + (2*(self.fit_env.predicting_steps+1),)))
			# self.set_down_model.add(Flatten(input_shape=(1,) + (22,)))

			self.set_down_model.add(Dense(100))
			self.set_down_model.add(Activation('sigmoid'))
			self.set_down_model.add(Dense(self.num_actions))
			self.set_down_model.add(Activation('linear'))
			self.set_down_model.load_weights(set_down_model_dir)

		# tensorbboard logistics
		self.sess = tf.Session()
		# self.controller.writer.add_graph(tf.get_default_graph())
		# self.metacontroller.writer.add_graph(tf.get_default_graph())
		self.writer = tf.summary.FileWriter('./logs_hdqn/'+self.exp_name)

	def goal_preprocess(self, goal):
		# print self.goal_shape
		# goal to one_hot
		vector = np.zeros(self.goal_shape)
		vector[0, goal] = 1.0
		return vector

	def fit(self, env, eval_env, num_episodes, eval_num_episodes):
		"""Fit your model to the provided environment.

		Parameters
		----------
		env: agent's environment
		eval_env: copy of agent's environment used for the evaluation
		num_episodes: number of episodes for the training
		eval_num_episodes: number of evaluation episodes
		"""
		# replay memories (of controller and metacontroller) burnin

		print('Start Burn-in')
		for episode_idx in range(self.metacontroller.num_burnin):
			# episode level
			# start new episode

			# get initial state
			state = env.reset()

			# select next goal
			goal, _ = self.metacontroller.select(self.metacontroller.burnin_policy)
			proc_goal = self.goal_preprocess(goal)
			setattr(env, 'goal', proc_goal)

			selected_goal = [goal]
			total_extrin_eps = 0

			# new episode, completing the goal
			while True:
				# goal level

				# the first state of the new goal
				state0 = state

				# the total environmental reward achieved by setting this goal
				extrinsic_reward = 0

				# new goal
				if DEBUG:
					print('Next goal {0}'.format(proc_goal))

				# goal has not been reached and the episode has not finished
				while True:
					# action level

					# select next action given the current goal
					action, _ = self.controller.select(self.controller.burnin_policy)

					# apply the action to the environment, get reward and nextstate
					next_state, in_reward, is_goal_completed, is_goal_over, is_eps_completed, is_eps_over = env.step(
						action)

					# compute the internal and external reward
					extrinsic_reward += env.get_extrinsic_reward()

					# store the experience in the controller's memory
					self.controller.memory.append([state, proc_goal],
					                              action,
					                              in_reward,
					                              [copy.copy(next_state), proc_goal],
					                              is_goal_completed or is_goal_over)

					# a deep copy of the value: important!
					state = copy.copy(next_state)
					if is_goal_completed or is_goal_over:
						break

				# store the experience in the metacontroller's memory
				self.metacontroller.memory.append([state0],
				                                  goal,
				                                  extrinsic_reward,
				                                  [copy.copy(next_state)],
				                                  is_eps_completed or is_eps_over)

				total_extrin_eps += extrinsic_reward

				if is_eps_over or is_eps_completed:
					# start new episode
					print("Burn-in episode: {0}".format(episode_idx))
					break
				else:
					# select next goal
					goal, _ = self.metacontroller.select(self.metacontroller.burnin_policy)
					proc_goal = self.goal_preprocess(goal)
					setattr(env, 'goal', proc_goal)
					selected_goal.append(goal)

		# start training the networks

		# vector counting the number of each goal that is selected by meta controller
		goal_num_samples = np.zeros(self.num_goals)

		for self.num_train_episodes in range(num_episodes):
			# start new episode

			# check if it's time to evaluate the agent
			if self.num_train_episodes > 0 and self.num_train_episodes % self.eval_freq == 0:
				self.evaluate(eval_env, eval_num_episodes)

			# get initial state
			state = env.reset()

			# select next goal, num_samples is for choosing epsilon
			goal, q = self.metacontroller.select(policy=self.metacontroller.training_policy, state=[state],
			                                     num_update=self.metacontroller.num_samples)

			assert goal is not None
			proc_goal = self.goal_preprocess(goal)

			setattr(env, 'goal', proc_goal)

			total_extrin_eps = 0

			selected_goal = [goal]

			mean_q = [q]

			while True:

				# goal level

				state0 = state
				extrinsic_reward = 0

				# new goal
				# goal has not been reached and the episode has not finished
				while True:

					if goal != self.num_goals - 1:
						# select next action given the current goal and state, individual epsilon for each goal
						action, _ = self.controller.select(policy=self.controller.training_policy,
						                                   state=[state, proc_goal], num_update=goal_num_samples[goal])
					else:
						# state_set_down = [state[0][0]]
						# state_set_down.extend(state[0][2:env.predicting_steps + 2])
						# state_set_down = np.reshape(state_set_down, (1, 1, env.predicting_steps + 1))
						state_set_down = np.reshape(state,(1,1,2*(env.predicting_steps+1)))
						q_values = self.set_down_model.predict(state_set_down)
						action = np.argmax(q_values)

					# apply the action to the environment, get reward and nextstate
					next_state, in_reward, is_goal_completed, is_goal_over, is_eps_completed, is_eps_over = env.step(
						action)
					assert next_state is not None

					# compute external reward
					extrinsic_reward += env.get_extrinsic_reward()

					if goal != self.num_goals - 1:
						# store the experience in the controller's memory
						self.controller.num_samples += 1
						self.controller.memory.append([state, proc_goal],
						                              action,
						                              in_reward,
						                              [copy.copy(next_state), proc_goal],
						                              is_goal_over or is_eps_completed)

					# update the weights of the controller's network
					self.controller.update_policy(self.writer)
					self.controller.num_updates += 1

					# update the weights of the metacontroller's network
					self.metacontroller.update_policy(self.writer)
					self.metacontroller.num_updates += 1

					# check if it's time to store the controller's model
					if self.controller.num_updates > 0 and self.controller.num_updates % self.save_freq == 0:
						self.controller.save_model()

					if self.metacontroller.num_updates > 0 and self.metacontroller.num_updates % self.save_freq == 0:
						self.metacontroller.save_model()

					# a deep copy of the value: important!
					state = copy.copy(next_state)

					if is_goal_over or is_goal_completed:
						goal_num_samples[goal] += 1
						break

				# store the experience in the metacontroller's memory
				self.metacontroller.num_samples += 1
				self.metacontroller.memory.append([state0],
				                                  goal,
				                                  extrinsic_reward,
				                                  [copy.copy(next_state)],
				                                  is_eps_completed or is_eps_over)

				total_extrin_eps += extrinsic_reward

				if is_eps_over or is_eps_completed:
					# start new episode
					if is_eps_completed:
						print(selected_goal)
					break
				else:
					# select next goal given new state, epsilon is determined by previous goal
					goal, q = self.metacontroller.select(policy=self.metacontroller.training_policy, state=[state],
					                                     num_update=goal_num_samples[goal])
					mean_q.append(q)

					proc_goal = self.goal_preprocess(goal)
					setattr(env, 'goal', proc_goal)
					selected_goal.append(goal)

					if goal == self.num_goals - 1: # goal is set-down
						setattr(env, 't_set_down', env.cur_step-env.initial_waiting_steps)
						setattr(env, 'height_set_down', env.cur_d_sb)

			save_scalar(self.num_train_episodes,'Total extrinsic reward', total_extrin_eps, self.writer)
			save_scalar(self.num_train_episodes, 'Episode mean Q', float(np.mean(mean_q)), self.writer)
			save_scalar(self.num_train_episodes, 'Num of selected goal', len(selected_goal), self.writer)
			save_scalar(self.num_train_episodes, 'Length episode', env.cur_step-env.initial_waiting_steps, self.writer)
			if env.height_set_down:
				save_scalar(self.num_train_episodes, 'Moment of set-down', float(env.t_set_down/(
					env.cur_step-env.initial_waiting_steps)),
				                                                                 self.writer)
				save_scalar(self.num_train_episodes, 'Height of set-down',  env.height_set_down, self.writer)

			print('Num update: {0}, Training episode: {1}, Impact_vel: {2},Total Reward: {3}, Mean_q: {4}'.format(
				self.metacontroller.num_updates, self.num_train_episodes, env.final_imp_vel, total_extrin_eps,
				np.mean(mean_q)))

	def evaluate(self, env, num_episodes, save_tensorboard = False):
		"""
		Evaluate the performance of the agent in the environment.
		Parameters
		----------
		env: agent's environment
		num_episodes: number of episodes for the testing
		"""

		episode_length = 0

		imp_vels = []

		print('Start Evaluation')
		for episode_idx in range(num_episodes):

			# start new episode

			total_reward = 0

			state = env.reset()

			assert state is not None

			# select next goal
			goal, _ = self.metacontroller.select(policy=self.metacontroller.testing_policy, state=[state])
			proc_goal = self.goal_preprocess(goal)
			setattr(env, 'goal', proc_goal)

			selected_goal = [goal]

			# new episode
			while True:

				# new goal
				while True:

					# if goal != self.num_goals + 1:

					# uncomment if you want to use extra skill
					if goal != self.num_goals - 1:

						# select next action given the current goal and state, individual epsilon for each goal
						action, _ = self.controller.select(policy=self.controller.training_policy,
						                                   state=[state, proc_goal])

					else:
						# state_set_down = [state[0][0]]

						#temp
						# state_set_down.extend([state[0][1]])

						# # state_set_down.extend(state[0][2:env.predicting_steps+2])
						# state_set_down.extend(state[0][2:12])

						# state_set_down.extend(state[0][27:37])

						# state_set_down = np.reshape(state_set_down,(1,1,22))
						# # state_set_down = np.reshape(state_set_down, (1, 1, env.predicting_steps+1))
						state_set_down = np.reshape(state, (1, 1, 2*(env.predicting_steps+1)))
						q_values = self.set_down_model.predict(state_set_down)
						action = np.argmax(q_values)

					# apply the action to the environment, get reward and next state
					next_state, in_reward, is_goal_completed, is_goal_over, is_eps_completed, is_eps_over = env.step(
						action)
					assert next_state is not None

					# compute the internal and external reward
					total_reward += env.get_extrinsic_reward()

					episode_length += 1

					if is_goal_over or is_goal_completed:
						break

					state = next_state

				if is_eps_over or is_eps_completed:
					# start new episode
					# print(selected_goal)
					# print(env.height_set_down)
					if is_eps_completed:
						print("Evl episode:{0}, imp_vel:{1}, total reward: {2}".format(episode_idx,
						                                                               env.final_imp_vel, total_reward))
						if env.final_imp_vel:
							imp_vels.append(env.final_imp_vel)
							# env.plot(show_motion=True)
							if save_tensorboard:
								save_scalar(episode_idx, 'Testing Reward', total_reward, self.writer)
								save_scalar(episode_idx, 'Testing Velocity ', env.final_imp_vel, self.writer)
								save_scalar(episode_idx, 'Testing height of set-down', env.height_set_down,
								            self.writer)
								save_scalar(episode_idx, 'Testing moment of set-down', float(env.t_set_down/env.cur_step),
								            self.writer)
					else:
						print("Evl episode:{0} fail".format(episode_idx))
						# env.plot(show_motion=True)
					break
				else:
					# select next goal
					goal, _ = self.metacontroller.select(policy=self.metacontroller.testing_policy, state=[state])
					proc_goal = self.goal_preprocess(goal)
					setattr(env, 'goal', proc_goal)
					selected_goal.append(goal)
					if goal == 6:
						setattr(env, 't_set_down', env.cur_step-env.initial_waiting_steps)
						setattr(env, 'height_set_down', env.cur_d_sb)

		print("mean vel: {0}, std: {1}".format(np.mean(imp_vels),np.std(imp_vels)))
		print("Completed: {0}, fail: {1}".format(len(imp_vels), num_episodes-len(imp_vels)))

