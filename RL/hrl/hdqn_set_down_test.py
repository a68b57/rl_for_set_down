#!/usr/bin/env python

import RL.hrl.policy as policy
from RL.hrl.objectives import mean_huber_loss
from RL.hrl.hdqn_set_down import HQNAgent
from RL.envs.hrl_env import HRL_gym
from keras.optimizers import SGD

#agent parameters
GAMMA = 0.99
ALPHA = 5e-4 #finished
NUM_EPISODES = 10000
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 10000
NUM_BURNIN = 1000
ANNEAL_NUM_STEPS = 500
EVAL_FREQ = 500
EXP_NAME = 'exp4'
model_save_freq = 100000

# env parameters
INIT_BARGE_CT = 10
INIT_HOIST_LEN = 3
LIMIT_DECAY = 0.993
LIMIT_MIN = 3
DT = 0.2
EPS_TIMEOUT = 200 # in second
GOAL_TIMEOUT = 200 # in second
OBS_LEN = 0.2
PRED_LEN = 5
HS = 1.5
TP = 15
NUM_ACTIONS = 13
NUM_GOALS = 7
EVAL_NUM_EPISODES = 500


def main():
	env = HRL_gym(init_barge_ct=INIT_BARGE_CT, init_hoist_len=INIT_HOIST_LEN, limit_decay=LIMIT_DECAY,
	              limit_min=LIMIT_MIN, dt=DT, eps_timeout=EPS_TIMEOUT, goal_timeout=GOAL_TIMEOUT, obs_len=OBS_LEN,
	              pred_len=PRED_LEN, hs=HS, tp=TP, num_actions=NUM_ACTIONS, num_goals=NUM_GOALS)
	test_env = HRL_gym(init_barge_ct=INIT_BARGE_CT, init_hoist_len=INIT_HOIST_LEN, limit_decay=LIMIT_DECAY,
	                   limit_min=LIMIT_MIN, dt=DT, eps_timeout=EPS_TIMEOUT, goal_timeout=GOAL_TIMEOUT, obs_len=OBS_LEN,
	                   pred_len=PRED_LEN, hs=HS, tp=TP, num_actions=NUM_ACTIONS, num_goals=NUM_GOALS)

	hdqn_agent = HQNAgent(
		state_shape=(1, env.state.shape[1]),
		goal_shape=(1, NUM_GOALS),
		num_actions=NUM_ACTIONS,
		num_goals=NUM_GOALS,
		controller_burnin_policy=policy.UniformRandomPolicy(NUM_ACTIONS),
		metacontroller_burnin_policy=policy.UniformRandomPolicy(NUM_GOALS),
		controller_training_policy=policy.LinearDecayGreedyEpsilonPolicy(NUM_ACTIONS, 0.3, 0.1, ANNEAL_NUM_STEPS),
		metacontroller_training_policy=policy.LinearDecayGreedyEpsilonPolicy(NUM_GOALS, 0.3, 0.1, ANNEAL_NUM_STEPS),
		controller_testing_policy=policy.GreedyEpsilonPolicy(0, NUM_ACTIONS),
		metacontroller_testing_policy=policy.GreedyEpsilonPolicy(0, NUM_GOALS),
		controller_gamma=GAMMA,
		metacontroller_gamma=GAMMA,
		controller_target_update_freq=TARGET_UPDATE_FREQ,
		metacontroller_target_update_freq=TARGET_UPDATE_FREQ,
		controller_batch_size=BATCH_SIZE,
		metacontroller_batch_size=BATCH_SIZE,
		controller_optimizer=SGD(lr=ALPHA),
		metacontroller_optimizer=SGD(lr=ALPHA),
		controller_loss_function=mean_huber_loss,
		metacontroller_loss_function=mean_huber_loss,
		eval_freq=EVAL_FREQ,
		controller_num_burnin=NUM_BURNIN,
		metacontroller_num_burnin=NUM_BURNIN,
		replay_buffer_size=REPLAY_BUFFER_SIZE,
		controller_dir='model/controller_source_exp4_2900000.weight',
		meta_controller_dir='model/metacontroller_source_exp4_2900000.weight',
		exp_name=EXP_NAME,
		save_freq=model_save_freq
	)

	# hdqn_agent.fit(env, test_env, NUM_EPISODES, EVAL_NUM_EPISODES)
	hdqn_agent.evaluate(test_env, EVAL_NUM_EPISODES)


if __name__ == '__main__':
	main()
