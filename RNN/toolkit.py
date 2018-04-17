from keras.models import model_from_json
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
from sklearn import model_selection
import glob
import time, os
import matplotlib.gridspec as gridspec
import tensorflow as tf
from numpy import fft
from statsmodels.tsa.ar_model import AR


graph = tf.get_default_graph()


def loadModel(model_dir, model_name):
	json_file = open(model_dir + model_name+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(model_dir + model_name + '.h5')
	return loaded_model


def saveModel(model_dir, model_name, model):
	model_json = model.to_json()
	with open(model_dir + model_name + ".json", "w") as json:
		json.write(model_json)
	model.save_weights(model_dir + model_name +".h5")


def addDim(data):
	return np.expand_dims(data, axis=2)


def prepareSplitDataSet(data_dir, test_size = 0.2, filename_pred_len=50, filename_obs_len=1000, pred_len=50, obs_len = 1000):
	x = []
	y = []
	for file in sorted(glob.glob(data_dir + "X_o{}_p{}*.mat".format(filename_obs_len, filename_pred_len))):
		key = file.split('/')[-1][1:]
		y_file = 'Y' + key
		x_data = loadmat(file)
		x.extend([x_data['X']])
		print(file)
		for y_file in glob.glob(data_dir + y_file):
			print(y_file)
			y_data = loadmat(y_file)
			y.extend([y_data['Y']])
	x = np.array(x)
	x = np.transpose(x, axes=(0,2,1))
	x = np.concatenate(x, axis=0)
	y = np.concatenate(np.transpose(np.array(y),axes=(0,2,1)), axis=0)

	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size)
	x_train = x_train[:, -obs_len:]
	x_test = x_test[:, -obs_len:]
	y_train = y_train[:, 0:pred_len]
	y_test = y_test[:, 0:pred_len]
	return addDim(x_train), addDim(x_test), addDim(y_train), y_test


def prepareDecoderData(x_train, y_train):
	encoder_input_data = x_train
	decoder_target_data = y_train
	last_col_x = encoder_input_data[:, -1, :]
	decoder_input_data = y_train[:, :-1, :]
	decoder_input_data = np.insert(decoder_input_data, 0, last_col_x, 1)
	return encoder_input_data, decoder_input_data, decoder_target_data


def processRawData(mat_file, len=1000):
	x = []
	x_data = loadmat(mat_file)
	x.extend([x_data['X']])
	x = np.array(x)
	x = np.transpose(x, axes=(0, 2, 1))
	x = np.concatenate(x, axis=0)
	return addDim(x[:, -len:])


def getStat(decoded, target):
	stat = {}
	err = np.abs(np.subtract(decoded, target))
	stat['mae'] = np.mean(err)
	stat['std'] = np.std(err)
	stat['mae_row'] = np.mean(err, axis=1)
	stat['std_row'] = np.std(err, axis=1)
	print(stat['mae'])
	print(stat['std'])
	return stat


def predict_no_loading(encoder_model, decoder_model, input_seq, pred_len):
	states_value = encoder_model.predict(input_seq)
	target_seq = np.zeros((len(input_seq), 1, 1))
	stop_condition = False
	decoded_seq = []
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
		decoded_seq.append(output_tokens)
		if len(decoded_seq) == pred_len:
			stop_condition = True
		target_seq = output_tokens
		states_value = [h, c]
	decoded_seq = np.array(decoded_seq).squeeze(axis=3)
	pred = np.transpose(decoded_seq, (1, 0, 2)).squeeze(axis=2)
	return pred


def predSeq(model_dir, input_seq, pred_len):
	encoder_model = loadModel(model_dir, "encoder")
	decoder_model = loadModel(model_dir, "decoder")
	pred = predict_no_loading(encoder_model, decoder_model, input_seq, pred_len)
	return pred


def makeSubplots(num_plot=6, decoded=None, target=None):
	fig = plt.figure(figsize=(10, 8))
	outer = gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.3)
	sel = np.random.choice(len(decoded), num_plot, replace=False)
	sel_dec = decoded[sel]
	if target is not None:
		sel_tar = target[sel]

	for i in range(num_plot):
		if target is not None:
			inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.8)
			for j in range(3):
				ax = plt.Subplot(fig, inner[j])

				if j == 0:
					ax.plot(sel_dec[i])
					ax.set_title('predicted')
				if j == 1:
					ax.plot(sel_tar[i],'r')
					ax.set_title('target')
				if j == 2:
					ax.plot(sel_dec[i])
					ax.plot(sel_tar[i], 'r')
					error = np.mean(np.abs(sel_dec[i] - sel_tar[i]))
					ax.set_title("MAE: %.3f" % error)
				fig.add_subplot(ax)

		else:
			inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.6)
			ax = plt.Subplot(fig, inner[0])
			ax.plot(sel_dec[i])
			ax.set_title('predicted')
			fig.add_subplot(ax)

	return plt


def evaPred(model_dir=None, decoded=None, target=None, save_pred=False, save_plot = False, plot=True):

	pred_dir = model_dir + 'prediction/'
	fig_dir = model_dir + 'figures/'
	time_str = time.strftime("_%H%M%S")
	if not os.path.exists(pred_dir):
		os.makedirs(pred_dir)

	fig = makeSubplots(decoded=decoded,target=target)

	if save_plot:
		if not os.path.exists(fig_dir):
			os.makedirs(fig_dir)
		fig.savefig(fig_dir+'random_testing_samples'+ time_str +'.png')

	if save_pred and target is not None:
		np.savetxt(pred_dir + 'predicted_'+ time_str+'.csv', decoded, delimiter=",")
		np.savetxt(pred_dir + 'target_' + time_str + '.csv', target, delimiter=",")

	elif save_pred and target is None:
		np.savetxt(pred_dir + 'deploy_predicted_'+ time_str+'.csv', decoded, delimiter=",")

	if plot:
		fig.show()


def fourierExtrapolation(x, n_predict):
	x = np.array(x).reshape(x.size,)
	n = x.size
	n_harm = 40
	t = np.arange(0, n)
	p = np.polyfit(t, x, 1)
	x_notrend = x - p[0] * t
	x_freqdom = fft.fft(x_notrend)
	f = fft.fftfreq(n)
	indexes = list(range(n))
	# sort indexes by frequency, lower -> higher
	indexes.sort(key=lambda i:np.absolute(f[i]))

	# t = np.arange(0, n + n_predict)
	t = np.arange(0, n_predict)
	restored_sig = np.zeros(t.size)
	for i in indexes[:1 + n_harm * 2]:
		ampli = np.absolute(x_freqdom[i]) / n   # amplitude
		phase = np.angle(x_freqdom[i])          # phase
		restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
	output = restored_sig + p[0]*t
	# output = restored_sig

	return output
	# return output.reshape(output.size, 1)[-n_predict:]


def goodnessOfFit(pred,gt):
	de = np.sqrt(np.sum((pred-gt)**2))
	no = np.sqrt(np.sum(gt**2))
	fit = 100*(1 - de/no)
	return fit


def makeMatSubset(mat_file, num_sample):
	dir = os.path.dirname(os.path.abspath(mat_file))
	data = loadmat(mat_file)
	data = data['X']
	dict = {}
	idx = np.random.choice(len(data), num_sample, replace=False)
	subset = data[:, idx]
	dict['X'] = subset
	file_name = os.path.basename(mat_file)
	file_name = os.path.splitext(file_name)[0]+'_small'
	savemat(dir+'/'+file_name, dict, oned_as='row')


def computeAR(data, pred_len, mode="same"):
	data = data[0]
	if mode == "same":
		model = AR(endog=data)
		model_fit = model.fit(maxlag=14, disp=False, ic='aic')
		y = model_fit.predict(start=len(data), end=len(data)+pred_len-1, dynamic=False)

	else:
		y = []
		for i in range(pred_len):
			model = AR(endog=data)
			model_fit = model.fit(disp=False)
			pred = model_fit.predict(start=len(data), end=len(data), dynamic=False)
			data = data[1:]
			data = np.vstack((data, pred))
			y.append(pred)
	return np.array(y)


def pointWiseMLP(model, data, pred_len):
	pred = []
	for i in range(pred_len):
		feed = data
		y = model.predict(feed)
		data = data[:, 1:]
		data = np.hstack((data, y))
		pred.append(y[0])
	return np.array(pred)



