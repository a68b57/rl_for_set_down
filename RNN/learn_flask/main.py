from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt, mpld3
from RNN import toolkit
import numpy as np
import tensorflow as tf

app = Flask(__name__)

obs_len = 1000
pred_len = 50
high_pass_f = 0.7
low_pass_f = 0.9
dt = 0.2

model_dir = "/home/michael/Desktop/workspace/rl_for_set_down/RNN/model/o%d_p%d_d%.1f_f%.1f-%.1f/" % (obs_len, pred_len, dt, high_pass_f, low_pass_f)
encoder_model = toolkit.loadModel(model_dir, "encoder_model")
decoder_model = toolkit.loadModel(model_dir, "decoder_model")
graph = tf.get_default_graph()

@app.route('/')
def upload():
	return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# global graph

		# with graph.as_default():
		f = request.files['file']
		data = toolkit.processRawData(f, obs_len=500000)

		num_frame = data.shape[1] // obs_len

		for i in range(num_frame):
			# plt.clf()
			# plt.plot(data[0][i:i+obs_len, :])
			input_seq = data[:, i:i+obs_len, :]
			pred = toolkit.predict_no_loading(encoder_model, decoder_model, input_seq, pred_len)
			# plt.plot(np.concatenate(data[0][i:i+obs_len], pred))
			# plt.pause(0.001)
			print(pred)


		# while True:
		# 	plt.pause(0.005)
		# return redirect(url_for('plot', data=data))


@app.route('/plot/<data>')
def plot(data):
	plt.plot(np.array(data))
	mpld3.show()


if __name__ == '__main__':
	app.run(port=5000,debug=True)