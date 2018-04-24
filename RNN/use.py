from RNN import toolkit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import spec_tools.spec_tools as st
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

obs_len = 1000
pred_len = 25

high_pass_f = 0.7
low_pass_f = 0.9
dt = 0.2

# dataset_dir = 'dataSet/d%.1f_f%.1f-%.1f_onetrace/' % (dt, high_pass_f, low_pass_f)
model_dir = "model/o%d_p%d_d%.1f_f%.1f-%.1f/" % (obs_len, pred_len, dt, high_pass_f, low_pass_f)


# x_train, x_test, y_train, y_test = toolkit.prepareSplitDataSet(dataset_dir, test_size=0.1, pred_len=pred_len, obs_len=obs_len)

# uncomment to specify the mat file to use
# mat_dir = dataset_dir + 'X_o5000000_p50_d0.2_f%.1f-%.1f_1.mat' % (high_pass_f, low_pass_f)

# x_test = toolkit.processRawData(mat_dir, obs_len=obs_len)


# offline testing
# pred = toolkit.predSeq(model_dir, x_test, pred_len)
# toolkit.evaPred(model_dir=model_dir, decoded=pred, target=None, save_pred=False, save_plot=False, plot=True)
# stat = toolkit.getStat(pred,y_test)


# prepare small time trace for real time testing
# toolkit.makeMatSubset(mat_dir, num_sample=1)


###########################################
# real-time feed

np.random.seed(10)
obs_len = 600
pred_len = 20
tol_len = 500000


# 3, 8 cut
# 1.5,15
# 3,5


hs = 3
tp = 5

resp = st.Spectrum.from_synthetic(spreading=None, Hs=hs, Tp=tp)
# resp.plot()
# resp.low_pass_FreqFilter(1.0)
# resp.high_pass_FreqFilter(0.6)


rel_motion_sc_t, data = resp.make_time_trace(tol_len, dt)

t = np.loadtxt('modeltestenv3.txt')
# data = np.array(t[:,1]).reshape(1,len(t[:,1]),1)


data = data.reshape(1,tol_len,1)

num_frame = data.shape[1] // obs_len

# hs3, tp8, 1.1 and 0.6 for LSTM
encoder_model = toolkit.loadModel(model_dir, "encoder_model")
decoder_model = toolkit.loadModel(model_dir, "decoder_model")

MLP_pointwise = toolkit.loadModel('model/mlp/', 'exp8_1_output')
MLP = toolkit.loadModel('model/mlp/', 'exp1')

f = plt.figure()
lim = (-hs*3/4,hs*3/4)
# lim = (-0.15,0.15)
ax1 = f.add_subplot(3, 1, 1)
ax1.set_ylim(lim)

ax2 = f.add_subplot(3, 1, 2)
ax2.set_ylim(lim)
ax3 = f.add_subplot(3, 1, 3)
ax3.set_ylim(lim)

ax1.set_xticks([0, obs_len-1, obs_len+pred_len], minor=False)
ax2.set_xticks([0, obs_len-1, obs_len+pred_len], minor=False)
ax3.set_xticks([0, obs_len-1, obs_len+pred_len], minor=False)

ax1.xaxis.grid(True, which='major')
ax2.xaxis.grid(True, which='major')
ax3.xaxis.grid(True, which='major')

ax1.set_ylabel('LSTM',  fontsize=15)
ax2.set_ylabel('MLP', fontsize=15)
ax3.set_ylabel('AR', fontsize=15)
ax3.set_xlabel('time_step(dt=0.2s)', fontsize=15)
start_show = int(obs_len*0.95)

def getPred(input_seq, mode="LSTM"):
	if mode == "LSTM":
		pred = toolkit.predict_no_loading(encoder_model, decoder_model, input_seq[:,0:obs_len,:].reshape(1,obs_len,1), pred_len)
		pred = np.transpose(pred)

	if mode == "MLP":
		pred = MLP.predict(input_seq[:,0:obs_len,:].reshape(1,obs_len)).reshape(pred_len,1)
		# pred = toolkit.pointWiseMLP(MLP_pointwise,input_seq[:,0:obs_len,:].reshape(1,obs_len), pred_len).reshape(pred_len,1)
	if mode == "FA":
		pred = toolkit.fourierExtrapolation(input_seq[:,0:obs_len,:], pred_len).reshape(pred_len,1)

	if mode == "AR":
		pred = toolkit.computeAR(input_seq[:,0:obs_len,:], pred_len, lag=14).reshape(pred_len,1)

	return np.concatenate((input_seq[0][0:obs_len], pred), axis=0)[start_show:]


x = np.arange(start_show, obs_len + pred_len, 1)

line1 = []
line1.extend(ax1.plot(x, data[0][x], 'b'))
line1.extend(ax1.plot(x, getPred(data[:, 0:obs_len+pred_len, :], mode="LSTM"), 'r'))

line2 = []
line2.extend(ax2.plot(x, data[0][x], 'b'))
line2.extend(ax2.plot(x, getPred(data[:, 0:obs_len+pred_len, :], mode="MLP"), 'r'))

line3 = []
line3.extend(ax3.plot(x, data[0][x], 'b'))
line3.extend(ax3.plot(x, getPred(data[:, 0:obs_len+pred_len, :], mode="AR"), 'r'))

err_tol1 = []
fit_tol1 = []
err_tol2 = []
fit_tol2 = []
err_tol3 = []
fit_tol3 = []

def calmae(target,pred):
	return mean_absolute_error(target,pred)

def calr2(target,pred):
	return r2_score(target,pred)

def animate(i):
	gt = data[0][x+i]
	input_seq = data[:, i:i+obs_len, :]

	pred1 = getPred(input_seq, mode="LSTM")
	pred2 = getPred(input_seq, mode="MLP")
	pred3 = getPred(input_seq, mode="AR")
	line1[0].set_ydata(gt)
	line1[1].set_ydata(pred1)
	line2[0].set_ydata(gt)
	line2[1].set_ydata(pred2)
	line3[0].set_ydata(gt)
	line3[1].set_ydata(pred3)
	target = gt[-pred_len:]


	mae1 = calmae(target,pred1[-pred_len:])
	mae2 = calmae(target,pred2[-pred_len:])
	mae3 = calmae(target,pred3[-pred_len:])
	r21 = calr2(target,pred1[-pred_len:])
	r22 = calr2(target,pred2[-pred_len:])
	r23 = calr2(target,pred3[-pred_len:])

	# err = np.sqrt(mean_squared_error(target,pred))
	# err = mean_absolute_error(target,pred)

	err_tol1.append(mae1)
	err_tol2.append(mae2)
	err_tol3.append(mae3)


	# fit = r2_score(target, pred)  # result of custom fit wrt mean fir compared to the best possible fit wrt mean fit

	fit_tol1.append(max(0, r21))
	fit_tol2.append(max(0, r22))
	fit_tol3.append(max(0, r23))

	ax1.set_title("step: %d, mean_mae: %.3f, mean_r2: %.3f" % (len(err_tol1), np.mean(np.array(err_tol1)),np.mean(np.array(fit_tol1))), color='g', fontsize=15)
	ax2.set_title("step: %d, mean_mae: %.3f, mean_r2: %.3f" % (len(err_tol2), np.mean(np.array(err_tol2)),np.mean(np.array(fit_tol2))), color='g', fontsize=15)
	ax3.set_title("step: %d, mean_mae: %.3f, mean_r2: %.3f" % (len(err_tol3), np.mean(np.array(err_tol3)),np.mean(np.array(fit_tol3))), color='g', fontsize=15)

	return line1, line2, line3,


def init():
	gt = data[0][x]
	line1[0].set_ydata(gt)
	line1[1].set_ydata(gt)
	line2[0].set_ydata(gt)
	line2[1].set_ydata(gt)
	line3[0].set_ydata(gt)
	line3[1].set_ydata(gt)
	return line1, line2, line3,


ani = animation.FuncAnimation(fig=f, func=animate, frames=num_frame, init_func=init, interval=50, blit=False)
plt.show()

