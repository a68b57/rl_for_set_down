from keras.layers import Input, Dense, CuDNNLSTM, Dropout, CuDNNGRU, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping
from RNN import toolkit
from keras import regularizers
from keras import initializers
import os
import spec_tools.spec_tools as st
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import preprocessing
import seaborn as sns
from statsmodels.tsa.ar_model import AR


K.set_learning_phase(1)

pred_len = 20
obs_len = 600 # 120 seconds
tol_length = 2000

dt = 0.2
hs = 1.5 #1, 2, 3
tp = 15   #5, 10, 15, 20

save_model = True

model_dir = 'model/mlp/'
model_name = 'exp8_1_output'
test_data_set_dir = 'dataSet/'


# layers = [obs_len, 1024, pred_len]
#
#
# if not os.path.exists(model_dir) and save_model:
# 	os.makedirs(model_dir)
#
#
resp = st.Spectrum.from_synthetic(spreading=None, Hs=hs, Tp=tp)
#
# data_set = []
# for i in range(90000):
# 	if i%1000 == 0:
# 		print(i)
# 	rel_motion_sc_t, data = resp.make_time_trace(2000, dt)
# 	data_set.append(data)
#
# data_set = np.array(data_set)
# x = data_set[:, 0:obs_len]
# y = data_set[:, obs_len:obs_len + pred_len]
#
#
#
# def build_model(layers):
# 	model = Sequential()
# 	model.add(Dense(units=layers[1], input_dim=layers[0]))
# 	model.add(Activation("sigmoid"))
# 	model.add(Dense(units=layers[-1]))
# 	model.add(Activation("linear"))
# 	model.compile(loss="mse", optimizer="adam")
# 	return model
#
#
# model = build_model(layers)
# model.summary()
#
#
# model.fit(x, y,
# batch_size=32,
# epochs=20,
# validation_split=0.2,
# shuffle=True,
# callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3)])
#
#
# toolkit.saveModel(model_dir, model_name, model)


model = toolkit.loadModel(model_dir, model_name)

# x_test = np.genfromtxt(test_data_set_dir+'x_test_1_15.csv', delimiter=',')
# y_test = np.genfromtxt(test_data_set_dir+'y_test_1_15.csv', delimiter=',')

# pred = model.predict(x_test)


for j in range(20):
	pred = []
	t, raw_data = resp.make_time_trace(1000, 0.2)
	data = raw_data[0:obs_len].reshape(1,obs_len)

	# pred = toolkit.pointWiseMLP(model,data,pred_len)

	pred = toolkit.computeAR(data, pred_len)

	label = raw_data[obs_len:obs_len+pred_len]
	plt.plot(np.array(pred).reshape(pred_len, ))
	plt.plot(label)
	plt.title(r2_score(label, np.array(pred).reshape(pred_len,)))
	plt.show()


# r = []
# mae = []
# mse = []
# for i, p in enumerate(pred):
# 	r.append(max(0,r2_score(p,y_test[i])))
# 	mae.append(mean_absolute_error(p,y_test[i]))
# 	mse.append(mean_squared_error(p,y_test[i]))
#
#
# mse = np.mean(np.array(mse))
# mae = np.mean(np.array(mae))
# r2 = np.mean(np.array(r))
# print(round(mae,5),round(mse,5),round(r2,5))