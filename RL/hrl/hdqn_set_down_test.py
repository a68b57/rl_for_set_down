#!/usr/bin/env python

import argparse
import os
import random

import numpy as np

import RL.hrl.policy as policy
from RL.hrl.objectives import mean_huber_loss
from RL.hrl.hdqn_set_down import HQNAgent
from RL.envs.hrl_env import HRL_gym

# debuging parameters
#agent parameters
#GAMMA = 0.99
#ALPHA = 25e-5
#NUM_EPISODES=10
#REPLAY_BUFFER_SIZE = 100 #not specified
#BATCH_SIZE = 2 #not specified
#TARGET_UPDATE_FREQ = 1#not specified
#NUM_BURNIN = 2 #not specified
#TRAIN_FREQ=2 #not specified
#ANNEAL_NUM_STEPS = 5 #finished for the metacontroller, check what adaptive anneal means for the controller
#EVAL_FREQ=1



#environment parameters
#INITIAL_STATE=1
#TERMINAL_STATE=0
#NUM_STATES=6
#BONUS_STATE=5
#ACTION_SUCCESS_PROB=0.5

#Evaluation Parameters
#EVAL_NUM_EPISODES=1 #finished


#agent parameters
GAMMA = 0.99
ALPHA = 5e-4 #finished
NUM_EPISODES = 100000
REPLAY_BUFFER_SIZE = 15000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 1000
NUM_BURNIN = 100
ANNEAL_NUM_STEPS = 5000
EVAL_FREQ = 1000

# env parameters
INIT_BARGE_CT = 10
INIT_HOIST_LEN = 3
LIMIT_DECAY = 0.998
LIMIT_MIN = 3
DT = 0.2
EPS_TIMEOUT = 200 # in second
GOAL_TIMEOUT = 200 # in second
OBS_LEN = 0.2
PRED_LEN = 2
HS = 1.5
TP = 15
NUM_ACTIONS = 13
NUM_GOALS = 7

RESET_FIT_LOGS = 1000 #for counting the number of visits per state

EVAL_NUM_EPISODES = 10


def main():
	env = HRL_gym(init_barge_ct=INIT_BARGE_CT,init_hoist_len=INIT_HOIST_LEN,limit_decay=LIMIT_DECAY,
	              limit_min=LIMIT_MIN,dt=DT,eps_timeout=EPS_TIMEOUT,goal_timeout=GOAL_TIMEOUT,obs_len=OBS_LEN,
	              pred_len=PRED_LEN,hs=HS,tp=TP,num_actions=NUM_ACTIONS,num_goals=NUM_GOALS)
	hdqn_agent = HQNAgent(
		# controller_network_type='Linear',
		# metacontroller_network_type='Linear',
		controller_network_type='Deep',
		metacontroller_network_type='Deep',
		state_shape=(1, env.state.shape[1]),
		goal_shape=(1, NUM_GOALS),
		num_actions=NUM_ACTIONS,
		num_goals=NUM_GOALS,
		controller_burnin_policy=policy.UniformRandomPolicy(NUM_ACTIONS),
		metacontroller_burnin_policy=policy.UniformRandomPolicy(NUM_GOALS),
		controller_training_policy=policy.LinearDecayGreedyEpsilonPolicy(NUM_ACTIONS, 1.0, 0.1, ANNEAL_NUM_STEPS),
		metacontroller_training_policy=policy.LinearDecayGreedyEpsilonPolicy(NUM_GOALS, 1.0, 0.1, ANNEAL_NUM_STEPS),
		controller_testing_policy=policy.GreedyEpsilonPolicy(0.05, NUM_ACTIONS),
		metacontroller_testing_policy=policy.GreedyEpsilonPolicy(0.05, NUM_GOALS),
		controller_gamma=GAMMA,
		metacontroller_gamma=GAMMA,
		controller_alpha=ALPHA,
		metacontroller_alpha=ALPHA,
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
		metacontroller_num_burnin=NUM_BURNIN
	)

	hdqn_agent.fit(env, env, NUM_EPISODES, EVAL_NUM_EPISODES, reset_env_fit_logs=RESET_FIT_LOGS)


if __name__ == '__main__':
	main()
