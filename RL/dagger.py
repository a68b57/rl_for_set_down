import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, CSVLogger
from keras import utils
import gym
import numpy as np
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import RL

import collections


exp_name = 'behavior_cloning_on_25.2.2_full_states'
model_dir = './model/dagger/'
memory_dir = './memory/'
log_dir = './log/dagger/'
ENV_NAME = 'SetDown-v2' # 1D

nb_iter = 5
nb_epoch = 1000
batch_size = 32



with open(memory_dir + 'demo_25.2.2_1900000.pickle', 'rb') as pickle_file:
	memory_expert = pickle.load(pickle_file)

# y = np.array(memory_expert.actions.data[123994:memory_expert.nb_entries:])
# x = np.array(memory_expert.observations.data[123994:memory_expert.nb_entries:])
y = np.array(memory_expert.actions.data[0:memory_expert.nb_entries])
x = np.array(memory_expert.observations.data[0:memory_expert.nb_entries])


y = utils.to_categorical(y, num_classes=5)

# lb = preprocessing.LabelBinarizer()
# lb.fit(y)
# y = lb.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



# WINDOW_LENGTH = 1
#
# env = gym.make(ENV_NAME)
# env_MCTS = gym.make(ENV_NAME)
# np.random.seed(123)
# env.seed(123)
#
# input_shape = (WINDOW_LENGTH,) + env.observation_space.shape

model = Sequential()
model.add(Dense(64, input_dim=13, kernel_regularizer=l2(1e-4)))
model.add(Activation('relu'))
model.add(Dense(64, kernel_regularizer=l2(1e-4)))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))
print(model.summary())


model.compile(optimizer=Adam(lr=1e-3), loss=categorical_crossentropy, metrics=['accuracy'])

# model.load_weights(model_dir+'exp2.1.h5')

csvlogger = CSVLogger(log_dir+exp_name+'.log')
Eearlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

model.fit(x=x_train, y=y_train, batch_size=128, epochs=1000, verbose=1, validation_split=0.2, shuffle=True,
          callbacks=[csvlogger,Eearlystop])

model.save(model_dir+exp_name+'.h5')







