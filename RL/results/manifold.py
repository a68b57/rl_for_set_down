import numpy as np
import gym
import RL
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



exp_name = '22.20.2'
ENV_NAME = 'Following-v1'
model_dir = '/home/michael/Desktop/workspace/rl_for_set_down/RL/model/exp22/'
model_name = 'following_'+exp_name+'_weights.h5f'

if __name__ == "__main__":
	# env = gym.make(ENV_NAME)
	# np.random.seed(123)
	# env.seed(123)
	# nb_actions = env.action_space.n
	#
	# model = Sequential()
	# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
	# model.add(Dense(100))
	# model.add(Activation('sigmoid'))
	# model.add(Dense(nb_actions))
	# model.add(Activation('linear'))
	# model.load_weights(model_dir + model_name)
	#
	# manifold = []
	# manifold.append(['d_sb','delta_d','action', 'q'])
	# for i in range(100): # d_sb from 0 to 1
	# 	print(i)
	# 	state = [None]*12
	# 	d_sb = (100-i) / 100
	# 	state[0] = d_sb
	# 	state[1] = 1 - d_sb
	# 	for j in range(100):
	# 		delta_tol = (50-j)/100 # delta d_tol from -0.5 to 0.5
	# 		pred = np.linspace(d_sb, delta_tol+d_sb, 6)
	# 		state[2:7] = pred[1:]
	# 		state[7:12] = 1 - pred[1:]
	# 		action = model.predict(np.array(state).reshape((-1,1,12)), batch_size=1)
	# 		q = np.max(action)
	# 		action = np.argmax(action)
	# 		manifold.append([d_sb, delta_tol, action, q])
	#
	# manifold = np.array(manifold)
	# df = pd.DataFrame(columns=manifold[0, :], data=manifold[1:, :])
	# df.to_pickle('manifold'+'_'+exp_name+'.pickle')


	########## for manifold ###########
	pal = sns.cubehelix_palette(15, rot=-.25, light=.8)

	df = pd.read_pickle('manifold_'+exp_name+'.pickle')

	# df['action'] = df['action'].map(action_list)

	df = df.apply(pd.to_numeric, errors='ignore')
	x = df['d_sb']
	y = df['delta_d']
	q = df['q']

	f1 = sns.lmplot(x="d_sb", y="delta_d", hue='action', fit_reg=False, size=5, data=df, legend=True)
	f1.set_axis_labels("distance to barge (m)","L <-------------- changing in track in 1s (m) --------------> R")
	action_list = {'0':'hold/0', '1':'R12', '2':'L-12'}
	# action_list = {'0':'L-12', '1':'L-10', '2':'L-8', '3':'L-6', '4':'L-4', '5':'L-2', '6':'0', '7':'R2', '8':'R4',
	#                '9':'R6', '10':'R8', '11':'R10', '12':'R12'}
	for t, l in zip(f1._legend.texts, action_list):
		t.set_text(action_list[t._text])
	plt.xlim(1, 0)
	plt.ylim(-0.5, 0.5)
	plt.title('policy for constant speed mode')

	######### for value function #########
	f2, ax2 = plt.subplots()
	cmap = sns.light_palette('#1ba1e2', as_cmap=True)
	points = ax2.scatter(df['d_sb'], df['delta_d'], c=df['q'], s=50, cmap=cmap)
	f2.colorbar(points)
	ax2.set(xlabel="distance to barge (m)",ylabel="L <-------------- changing in track in 1s (m) --------------> R")

	plt.xlim(1, 0)
	plt.ylim(-0.5, 0.5)
	plt.title('value function for constant speed mode')
	plt.show()


