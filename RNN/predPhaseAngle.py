from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping
import numpy as np
from RNN import toolkit
from keras import regularizers
from keras import initializers
import os


K.set_learning_phase(1) #set learning phase


pred_len = 21
obs_len = 5000
latent_dim = 100


dataset_dir = 'dataSet/d0.2_f0.7-0.9_pa/'
model_dir = "model/ppa_o5000_d0.2_f0.7-0.9/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

x_train, x_test, y_train, y_test = toolkit.splitDataSet(dataset_dir,
                                                        test_size=0.1,
                                                        pred_len=pred_len,
                                                        obs_len=obs_len)

model = Sequential()
model.add(Dense(units=latent_dim, use_bias=True, input_shape=(obs_len,), activation='relu',
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=None),
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(units=latent_dim, use_bias=True,activation='relu',
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=None),
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(units=pred_len,
                kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.05, seed=None),
                kernel_regularizer=regularizers.l2(0.01)))

adam = Adam(lr=1e-1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])

model.summary()

model.fit(x_train, y_train,
          batch_size=40,
          epochs=200,
          validation_split=0.2,
          shuffle=True,
          verbose=2,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10)])


pred = model.predict(x_test)
mean,std,mean_row,std_row = toolkit.getStat(pred,y_test)

print("error mean: %g, std: %g" % (mean, std))
np.savetxt(model_dir+'pred.csv', pred, delimiter=",")
np.savetxt(model_dir+'y.csv', y_test, delimiter=",")