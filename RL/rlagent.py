
from collections import deque
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, SGD
import random
import numpy as np
import os
from RNN import toolkit


class RLAgent:
	def __init__(self, state_size, action_size, testing):

		self.action_temp = None
		self.state_temp = None
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=100000)
		self.epsilon_min = 0.1
		self.learning_rate = 0.005
		self.testing = testing
		self.gamma_max = 0.99
		if self.testing:
			self.model = None
			self.epsilon = 0.0
			self.gamma = 0.98
		else:
			self.model = self._build_model()
			self.target_model = self._build_model()
			self.epsilon = 1.0
			self.gamma = 0.98

	def _build_model(self):
		model = Sequential()
		model.add(Dense(100, input_shape=(self.state_size,)))
		model.add(Activation('sigmoid'))
		# model.add(Dropout(0.5))
		# model.add(Dense(100))
		# model.add(Activation('relu'))
		# model.add(Dropout(0.5))
		model.add(Dense(self.action_size, activation='linear'))
		# model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size), 0
		act_values = self.model.predict(np.reshape(state, [1, self.state_size]))
		return np.argmax(act_values[0]), np.max(act_values)

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)

		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
			target_f = self.target_model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		# double q-learning:
		# for state, action, reward, next_state, done in minibatch:
		# 	target = self.model.predict(state)
		# 	if done:
		# 		target[0][action] = reward
		# 	if not done:
		# 		a = np.argmax(self.model.predict(next_state)[0])
		# 		target[0][action] = reward + self.gamma * self.target_model.predict(next_state)[0][a]
		#
		# 	self.model.fit(state, target, epochs=1, verbose=0)

	def target_train(self):
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i]
		self.target_model.set_weights(target_weights)

	def saveModel(self, dir, name):
		if not os.path.exists(dir):
			os.makedirs(dir)
		model_json = self.model.to_json()
		with open(dir + name + ".json", "w") as json:
			json.write(model_json)
		self.model.save_weights(dir + name + ".h5")

	def loadModel(self, dir, name):
		model = toolkit.loadModel(dir, name)
		self.model = model



