#!/usr/bin/env python

import RL.hrl.policy as policy
from RL.hrl.objectives import mean_huber_loss
from RL.hrl.hdqn_set_down import HQNAgent
from RL.envs.hrl_env import HRL_gym
from keras.optimizers import SGD, Adam
import numpy as np

#agent parameters
GAMMA = 0.99
ALPHA = 1e-3 #finished
NUM_EPISODES = 20000
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 10000
NUM_BURNIN = 1000
ANNEAL_NUM_STEPS = 500 # in steps
EVAL_FREQ = 1000
EXP_NAME = 'exp13'
model_save_freq = 100000 # in steps

# env parameters
INIT_BARGE_CT = 8
INIT_HOIST_LEN = 3
LIMIT_DECAY = 0.99
LIMIT_MIN = 3
DT = 0.2
EPS_TIMEOUT = 200 # in second
GOAL_TIMEOUT = 30 # in second

OBS_LEN = 0.2
PRED_LEN = 2
HS = 1.5
TP = 15
NUM_ACTIONS = 13
NUM_GOALS = 7
EVAL_NUM_EPISODES = 500
INITIAL_WAITING_STEP = 600

def main():
	env = HRL_gym(init_barge_ct=INIT_BARGE_CT, init_hoist_len=INIT_HOIST_LEN, limit_decay=LIMIT_DECAY,
	              limit_min=LIMIT_MIN, dt=DT, eps_timeout=EPS_TIMEOUT, goal_timeout=GOAL_TIMEOUT, obs_len=OBS_LEN,
	              pred_len=PRED_LEN, hs=HS, tp=TP, num_actions=NUM_ACTIONS, num_goals=NUM_GOALS, use_AR=False,
	              initial_waiting_step=INITIAL_WAITING_STEP)
	test_env = HRL_gym(init_barge_ct=INIT_BARGE_CT, init_hoist_len=INIT_HOIST_LEN, limit_decay=0.998,
	                   limit_min=LIMIT_MIN, dt=DT, eps_timeout=EPS_TIMEOUT, goal_timeout=GOAL_TIMEOUT,
	                   obs_len=OBS_LEN,
	                   pred_len=PRED_LEN, hs=HS, tp=TP, num_actions=NUM_ACTIONS, num_goals=NUM_GOALS, use_AR=True,
	                   initial_waiting_step=INITIAL_WAITING_STEP)

	np.random.seed(123)
	env.seed(123)
	test_env.seed(123)

	hdqn_agent = HQNAgent(
		state_shape=(1, env.state.shape[1]),
		goal_shape=(1, NUM_GOALS),
		num_actions=NUM_ACTIONS,
		num_goals=NUM_GOALS,
		controller_burnin_policy=policy.UniformRandomPolicy(NUM_ACTIONS),
		metacontroller_burnin_policy=policy.UniformRandomPolicy(NUM_GOALS),
		controller_training_policy=policy.LinearDecayGreedyEpsilonPolicy(NUM_ACTIONS, 0.8, 0.1, ANNEAL_NUM_STEPS),
		metacontroller_training_policy=policy.LinearDecayGreedyEpsilonPolicy(NUM_GOALS, 0.8, 0.1, ANNEAL_NUM_STEPS),
		controller_testing_policy=policy.GreedyEpsilonPolicy(0, NUM_ACTIONS),
		metacontroller_testing_policy=policy.GreedyEpsilonPolicy(0, NUM_GOALS),
		controller_gamma=GAMMA,
		metacontroller_gamma=GAMMA,
		controller_target_update_freq=TARGET_UPDATE_FREQ,
		metacontroller_target_update_freq=TARGET_UPDATE_FREQ,
		controller_batch_size=BATCH_SIZE,
		metacontroller_batch_size=BATCH_SIZE,
		controller_optimizer=Adam(lr=ALPHA),
		metacontroller_optimizer=Adam(lr=ALPHA),
		controller_loss_function=mean_huber_loss,
		metacontroller_loss_function=mean_huber_loss,
		eval_freq=EVAL_FREQ,
		controller_num_burnin=NUM_BURNIN,
		metacontroller_num_burnin=NUM_BURNIN,
		replay_buffer_size=REPLAY_BUFFER_SIZE,
		controller_dir='model/controller_source_exp13_1100000.weight',
		meta_controller_dir='model/metacontroller_source_exp13_1100000.weight',
		exp_name=EXP_NAME,
		save_freq=model_save_freq,
		set_down_model_dir='../model/exp22/following_23.1.2.4_weights_1300000.h5f',
		fit_env=env,
	)

	# hdqn_agent.fit(env, test_env, NUM_EPISODES, EVAL_NUM_EPISODES)
	hdqn_agent.evaluate(test_env, EVAL_NUM_EPISODES, save_tensorboard=False)


if __name__ == '__main__':
	main()
