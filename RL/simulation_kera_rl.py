import numpy as np
import gym

import RL

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam,SGD

from rl.agents import SARSAAgent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, TestLogger


ENV_NAME = 'SetDown-v1'
# ENV_NAME = 'Following-v1'


model_dir = './model/exp22/'
log_dir = './log/exp22/'

exp_name = '23.8'

if __name__ == "__main__":
	env = gym.make(ENV_NAME)
	np.random.seed(123)
	env.seed(123)
	nb_actions = env.action_space.n

	model = Sequential()
	model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
	model.add(Dense(100))
	model.add(Activation('sigmoid'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))
	print(model.summary())

	memory = SequentialMemory(limit=5000, window_length=1)
	policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.5, value_min=0.1, value_test=0,
	                              nb_steps=10000)

	dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
	               target_model_update=1000, policy=policy, enable_dueling_network=False, enable_double_dqn=True)

	dqn.compile(SGD(lr=1e-3), metrics=['mse'])

	dqn.load_weights(model_dir + 'following_'+'23.1'+'_weights_1300000.h5f')

	# weights_filename = model_dir + 'following_{}_weights.h5f'.format(exp_name)
	# checkpoint_weights_filename = model_dir + 'following_' + exp_name + '_weights_{step}.h5f'
	# log_filename = log_dir + 'following_{}_log.csv'.format(exp_name)
	# callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=100000)]
	# callbacks += [FileLogger(log_filename, interval=1)]

	# dqn.fit(env, nb_steps=1500000, visualize=False, verbose=2, callbacks=callbacks)
	dqn.test(env, nb_episodes=500, visualize=False)
