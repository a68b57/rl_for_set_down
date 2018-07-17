import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

import RL


from baselines import bench, logger

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy


ENV_NAME = 'SetDown-v2'

def train(env_id, num_timesteps, seed):
	ncpu = 1
	config = tf.ConfigProto(allow_soft_placement=True,
							intra_op_parallelism_threads=ncpu,
							inter_op_parallelism_threads=ncpu)
	tf.Session(config=config).__enter__()

	def make_env():
		env = gym.make(env_id)
		env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
		return env

	env = DummyVecEnv([make_env])

	set_global_seeds(seed)
	policy = MlpPolicy
	model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
					   lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
					   ent_coef=0.0,
					   lr=3e-4,
					   cliprange=0.2,
					   total_timesteps=num_timesteps)


	return model, env


def main():
	logger.configure()
	model, env = train(ENV_NAME, num_timesteps=1500000, seed=0)

	# if args.play:
	# 	logger.log("Running trained model")
	# 	obs = np.zeros((env.num_envs,) + env.observation_space.shape)
	# 	obs[:] = env.reset()
	# 	while True:
	# 		actions = model.step(obs)[0]
	# 		obs[:]  = env.step(actions)[0]
	# 		env.render()


if __name__ == '__main__':
	main()


























# def model(inpt, num_actions, scope, reuse=False):
# 	"""This model takes as input an observation and returns values of all actions."""
# 	with tf.variable_scope(scope, reuse=reuse):
# 		out = inpt
# 		out = layers.fully_connected(out, num_outputs=200, activation_fn=tf.nn.sigmoid)
# 		out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
# 		return out
#
#
# if __name__ == '__main__':
# 	with U.make_session(8):
# 		# Create the environment
# 		ENV_NAME = 'SetDown-v2' # 1D
# 		env = gym.make(ENV_NAME)
# 		# Create all the functions necessary to train the model
# 		act, train, update_target, debug = deepq.build_train(
# 			make_obs_ph=lambda name:ObservationInput(env.observation_space, name=name),
# 			q_func=model,
# 			num_actions=env.action_space.n,
# 			optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
# 			double_q=True,
#
# 		)
# 		# Create the replay buffer
# 		replay_buffer = ReplayBuffer(50000)
# 		# replay_buffer = PrioritizedReplayBuffer(50000, 0.6)
#
#
# 		# Create the schedule for exploration starting from 1 (every action is random) down to
# 		# 0.02 (98% of actions are selected according to values predicted by the model).
# 		# beta_anneal = LinearSchedule(schedule_timesteps=1500000, initial_p=0.4, final_p=1)
# 		exploration = LinearSchedule(schedule_timesteps=50000, initial_p=0.5, final_p=0.1)
#
#
# 		# Initialize the parameters and copy them to the target network.
# 		U.initialize()
# 		update_target()
#
# 		episode_rewards = [0.0]
# 		obs = env.reset()
# 		for t in itertools.count():
# 			# Take action and update exploration to the newest value
# 			action = act(obs[None], update_eps=exploration.value(t))[0]
# 			new_obs, rew, done, _ = env.step(action)
# 			# Store transition in the replay buffer.
# 			replay_buffer.add(obs, action, rew, new_obs, float(done))
# 			obs = new_obs
#
# 			episode_rewards[-1] += rew
# 			if done:
# 				obs = env.reset()
# 				episode_rewards.append(0)
#
# 			is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 150
# 			if is_solved:
# 				# Show off the result
# 				# env.render()
# 				break
# 			else:
# 				# Minimize the error in Bellman's equation on a batch sampled from replay buffer.
# 				if t > 10000:
# 					# experience = replay_buffer.sample(32, beta=beta_anneal.value(t))
# 					# (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
# 					# td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
# 					obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
# 					train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
# 					# new_priorities = np.abs(td_errors) + 1e-6
# 					# replay_buffer.update_priorities(batch_idxes, new_priorities)
# 				# Update target network periodically.
# 				if t % 10000 == 0:
# 					update_target()
#
# 			if done and len(episode_rewards) % 10 == 0:
# 				logger.record_tabular("steps", t)
# 				logger.record_tabular("episodes", len(episode_rewards))
# 				logger.record_tabular("mean reward last 300", round(np.mean(episode_rewards[-301:-1]), 1))
# 				logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
# 				logger.dump_tabular()
