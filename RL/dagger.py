import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, CSVLogger
import gym
import numpy as np
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import RL


exp_name = 'exp2.1'
model_dir = './model/dagger/'
memory_dir = './memory/'
log_dir = './log/dagger/'
ENV_NAME = 'SetDown-v2' # 1D

nb_iter = 5
nb_epoch = 1000
batch_size = 32




with open(memory_dir + 'dagger_memory_3(dagger_memory_2)_temp.pickle', 'rb') as pickle_file:
	memory_expert = pickle.load(pickle_file)

y = np.array(memory_expert.actions.data[123994:memory_expert.nb_entries:])
x = np.array(memory_expert.observations.data[123994:memory_expert.nb_entries:])
lb = preprocessing.LabelBinarizer()
lb.fit(y)
y = lb.transform(y)

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
model.add(Dense(200, input_dim=85, kernel_regularizer=l2(1e-4)))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
print(model.summary())


def custom_loss(y_true, y_pred):
	indices = K.argmax(y_true, axis=-1)
	return -K.log(K.gather(reference=y_pred, indices=indices))


model.compile(optimizer=Adam(lr=1e-4), loss=categorical_crossentropy, metrics=['categorical_accuracy'])

model.load_weights(model_dir+'exp2.h5')

# csvlogger = CSVLogger(log_dir+exp_name+'.log')
# Eearlystop = arlyStopping(monitor='val_loss', min_delta=0, patience=5)

model.fit(x=x_train, y=y_train, batch_size=32, epochs=1000, verbose=1, validation_split=0.2, shuffle=True,
          callbacks=[])

# model.save(model_dir+exp_name+'.h5')







