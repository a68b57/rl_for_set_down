import numpy as np
import gym
import pickle

# this import RL is necessary
import RL

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam,SGD
from keras.regularizers import l2

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory, PrioritizedMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, TestLogger, MemoryIntervalCheckpoint

from RL.tools import MCTS

# ENV_NAME = 'SetDown-v1'
# ENV_NAME = 'Following-v1'
# ENV_NAME = 'SetDown-v2' # 1D
ENV_NAME = 'SetDown-v3' # 2D

model_dir = './model/exp22/'
log_dir = './log/exp22/'

memory_dir = './memory/'

exp_name = '26.4.2'

WINDOW_LENGTH = 1

if __name__ == "__main__":
	env = gym.make(ENV_NAME)
	env_MCTS = gym.make(ENV_NAME)
	np.random.seed(123)
	env.seed(123)
	nb_actions = env.action_space.n
	input_shape = (WINDOW_LENGTH,) + env.observation_space.shape

	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(200))
	model.add(Activation('sigmoid'))
	# model.add(Dense(64))
	# model.add(Activation('relu'))
	model.add(Dense(nb_actions, activation='linear'))
	print(model.summary())


	memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)

	# TODO: just for collecting the transistions of expert
	# memory = SequentialMemory(limit=5000000, window_length=1)

	# use PER
	# memory = PrioritizedMemory(limit=50000, alpha=.6, start_beta=.4, end_beta=1, steps_annealed=2000000,
	#                             window_length=WINDOW_LENGTH)

	# memory_file_name = memory_dir + '24.2_with_MCTS'+'_memory.pickle'


	# Keep appending previous memory buffer
	# with open(memory_file_name, 'rb') as pickle_file:
	# 	memory = pickle.load(pickle_file)

	# memory.deq_to_per(memory_expert_pickle)

	policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=0.01, value_test=0,
	                              nb_steps=700000)

	dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, memory_expert=None, nb_steps_warmup=10,
	               target_model_update=10000, policy=policy, enable_dueling_network=False, enable_double_dqn=True,
	               max_MCTS_eps=30, use_mc=False)

	lr = 1e-3

	if type(memory) == PrioritizedMemory:
		lr /= 4
	dqn.compile(Adam(lr=lr))

	# dqn.load_weights(model_dir + 'following_'+'25.3.2'+'_weights_400000.h5f')

	# dqn.load_weights('./model/dagger/behavior_cloning_on_25.2.2_full_states.h5')

	weights_filename = model_dir + 'following_{}_weights.h5f'.format(exp_name)
	checkpoint_weights_filename = model_dir + 'following_' + exp_name + '_weights_{step}.h5f'
	log_filename = log_dir + 'following_{}_log.csv'.format(exp_name)
	callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=100000)]
	callbacks += [FileLogger(log_filename, interval=1)]

	dqn.fit(env, nb_steps=1500000, visualize=False, verbose=2, callbacks=callbacks)

	# callbacks = [MemoryIntervalCheckpoint(memory_dir + 'dagger_memory_2(appending_on_24.2_with_MCTS_memory).pickle',
	#                                       interval=1000)]
	# dqn.test(env=env, mcts_env=env_MCTS, nb_episodes=2000, visualize=False, callbacks=callbacks)

	dqn.test(env=env, mcts_env=env_MCTS, nb_episodes=500, visualize=False)
