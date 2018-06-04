import gym
from gym import spaces
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from gym.utils import seeding
import spec_tools.spec_tools as st
import RNN.toolkit as toolkit


class SetDown_gym(gym.Env):

	def __init__(self):
		# self.init_h_s_ct = 10
		# self.init_hoist_len = 3
		self.init_h_s_ct = 5
		self.init_hoist_len = 4
		self.cur_limit = None
		self.init_limit = self.init_h_s_ct - self.init_hoist_len + 3
		# self.init_limit = self.init_h_s_ct - self.init_hoist_len + 5
		# self.init_limit = self.init_h_s_ct - self.init_hoist_len + 10

		self.limit_decay = 0.998
		self.limit_min = 3

		self.lowering_speed = 12 / 60
		self.lifting_speed = 12 / 60
		self.num_action = 13
		self.dt = 0.2
		self.timeout = 20
		self.hit_steps = 20

		self.obs_len = 0.2
		# self.initial_waiting_steps = int(self.obs_len / self.dt)  # waiting time for the first observation
		self.initial_waiting_steps = 600  # initial waiting time for using AR

		self.pred_len = 2
		self.predicting_steps = int(self.pred_len / self.dt)

		self.num_step = int(np.ceil(self.timeout / self.dt)) + 1 + self.initial_waiting_steps
		self.cur_step = None
		self.t = None

		self.holding_length = 0 # skip 5 frames per action
		self.holding_step = max(1, int(self.holding_length / self.dt))

		self.seed()

		self.use_AR = False

		self.rel_motion_sc = None
		self.rel_motion_sc_t = None

		self.resp = st.Spectrum.from_synthetic(spreading=None, Hs=2, Tp=7)
		self.rel_motion_sc_t, self.rel_motion_sc = self.resp.make_time_trace(self.num_step + 200, self.dt)

		self.cur_d_sb = None # current distance between supply boat and block (margin for the right)
		self.cur_d_blimit = None # margin to the left
		self.prev_d_sb = None
		self.cur_hoist_length = None

		self.high_limit = (self.init_h_s_ct - self.init_hoist_len + 10) * np.ones(
			2 * (int(self.obs_len / self.dt) + int(self.pred_len / self.dt)))
		# self.high_limit = (self.init_h_s_ct - self.init_hoist_len + 10) * np.ones(
		# 	1 * (int(self.obs_len / self.dt) + int(self.pred_len / self.dt)))
		self.action_space = spaces.Discrete(self.num_action)
		self.observation_space = spaces.Box(-self.high_limit, high=self.high_limit, dtype=np.float16)
		self.state = np.zeros([self.high_limit.shape[0]])
		self.sum_reward = 0

		self.hoist_len_track = []
		self.d_sb_track = []
		self.d_blimit_track = []

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		# random initial position
		# self.init_h_s_ct = 3.5
		# self.init_hoist_len = 3*np.random.rand()

		self.hoist_len_track = []
		self.d_sb_track = []
		self.d_blimit_track = []
		self.t = 0
		self.cur_step = 0
		self.cur_d_sb = self.init_h_s_ct - self.init_hoist_len
		self.cur_d_blimit = self.init_limit - self.cur_d_sb
		self.prev_d_sb = self.cur_d_sb
		self.cur_hoist_length = self.init_hoist_len
		self.final_imp_vel = 0
		self.sum_reward = 0
		self.cur_limit = self.init_limit

		self.cur_step += self.initial_waiting_steps
		self.t += round(self.cur_step * self.dt, 2)

		self.rel_motion_sc_t, self.rel_motion_sc = self.resp.make_time_trace(self.num_step + 1000, self.dt)
		# pred0 = self.rel_motion_sc[self.initial_waiting_steps-1:self.predicting_steps]
		pred0 = self.rel_motion_sc[self.initial_waiting_steps:self.initial_waiting_steps+self.predicting_steps+1]


		# prediction by AR
		AR_prediction = toolkit.computeAR(data=np.reshape(self.rel_motion_sc[0:self.initial_waiting_steps],
		                                                  (1,self.initial_waiting_steps,1)),
		                                  pred_len=self.predicting_steps+1)

		# first two elements are margin to right and left
		# self.state[self.initial_waiting_steps - 1] = self.cur_d_sb
		self.state[0] = self.cur_d_sb
		self.state[1] = self.cur_d_blimit

		# comment for both margin

		if not self.use_AR:
			# real wave
			# self.state[1:self.predicting_steps + 1] =\
			# 		self.cur_d_sb - (pred0[1:] - pred0[0])

			self.state[2:self.predicting_steps + 2] = self.cur_d_sb - (pred0[1:] - pred0[0])
			self.state[self.predicting_steps + 2:] = self.cur_d_blimit + (pred0[1:] - pred0[0])
		else:
			# predicted wave by AR
			# self.state[1:self.predicting_steps + 1] =\
			# 	self.cur_d_sb - (AR_prediction[1:] - AR_prediction[0])

			self.state[2:self.predicting_steps + 2] = self.cur_d_sb - (AR_prediction[1:] - AR_prediction[0])
			self.state[self.predicting_steps + 2:] = self.cur_d_blimit + (AR_prediction[1:] - AR_prediction[0])


		# uncomment for both margin
		# self.state[self.initial_waiting_steps] = self.cur_d_blimit

		#prediction of right margin (old)
		# self.state[self.initial_waiting_steps:self.predicting_steps+1] = self.cur_d_sb + (self.rel_motion_sc[0]-pred0)

		#prediction of left margin (old)
		# self.state[self.predicting_steps + 2:] = self.cur_d_blimit + (self.rel_motion_sc[1:self.predicting_steps+1]-pred0)


		# uncomment for both margin mode
		# prediction of right margin
		# self.state[self.initial_waiting_steps + 1:self.predicting_steps + 2] =\
		# 	self.cur_d_sb - (self.rel_motion_sc[1:self.predicting_steps + 1] - pred0[0])

		# prediction of left margin
		# self.state[self.predicting_steps + 2:] =\
		# 	self.cur_d_blimit + (self.rel_motion_sc[1:self.predicting_steps + 1] - pred0[0])

		self.state = np.reshape(self.state, [self.state.shape[0], ])
		return np.array(self.state)

	def get_reward(self, gameover):
		reward = 0

		if gameover:
			if self.cur_step < self.num_step and self.cur_d_sb < 0: # set-down
				vel = (self.prev_d_sb - self.cur_d_sb)/self.dt
				print(vel)
				if 0 < vel < 0.3: # good one
					reward = 10*1/vel
					self.final_imp_vel = vel
					if vel < 0.05:
						self.plot(show_motion=True)
				else: # bad one
					reward = -30
					# reward = 0
			if self.cur_d_sb > self.cur_limit: # hit boundary
				reward = -20

		# if not gameover:
		# 	reward = -0.01
		self.sum_reward += reward

		return reward

	def if_gameover(self):

		gameover = False
		# uncomment for both margins
		if self.cur_d_sb < 0 or self.cur_step == self.num_step or self.cur_d_sb > self.cur_limit:
		# if self.cur_d_sb < 0 or self.cur_step == self.num_step:

			gameover = True
		return gameover

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

		d_sb = self.cur_d_sb
		hoist_len = self.cur_hoist_length

		self.prev_d_sb = d_sb

		for k in range(self.holding_step):

			self.hoist_len_track.append(hoist_len)
			self.d_sb_track.append(d_sb)
			self.d_blimit_track.append(self.cur_d_blimit)

			speed = (action-int((self.num_action-1)//2))/30
			hoist_len = max(hoist_len + speed * self.dt, 0)

			self.cur_hoist_length = hoist_len

			h_sc = self.init_h_s_ct - np.interp(self.t, self.rel_motion_sc_t, self.rel_motion_sc)
			d_sb = h_sc - hoist_len

			self.cur_d_sb = d_sb
			self.cur_d_blimit = self.cur_limit - self.cur_d_sb

			if self.if_gameover():
				self.hoist_len_track.append(hoist_len)
				self.d_sb_track.append(d_sb)
				self.d_blimit_track.append(self.cur_d_blimit)
				break

			self.cur_step += 1
			self.t = round(self.cur_step * self.dt, 2)

		if self.use_AR:
			pred = toolkit.computeAR(data=np.reshape(self.rel_motion_sc[
			                                            self.cur_step-self.initial_waiting_steps:self.cur_step],
			                                                  (1,self.initial_waiting_steps,1)),
			                                  pred_len=self.predicting_steps+1)

		else:
			pred = self.rel_motion_sc[self.cur_step:self.cur_step + self.predicting_steps + 1]

		# self.state[self.initial_waiting_steps - 1] = self.cur_d_sb
		self.state[0] = self.cur_d_sb
		self.state[1] = self.cur_d_blimit


		# comment for both margin mode

		# this setting is wrong
		# self.state[self.initial_waiting_steps :self.predicting_steps + 1] = self.cur_d_sb - (pred[0] - pred[1:])

		# this setting is used for training 23,4
		# self.state[self.initial_waiting_steps :self.predicting_steps + 1] = self.cur_d_sb + (pred[0] - pred[1:])

		# single margin
		# self.state[1:self.predicting_steps + 1] = self.cur_d_sb - (pred[1:] - pred[0])

		self.state[2:self.predicting_steps + 2] = self.cur_d_sb - (pred[1:] - pred[0])
		self.state[self.predicting_steps + 2:] = self.cur_d_blimit + (pred[1:] - pred[0])


		# uncomment for both margin mode
		# self.state[self.initial_waiting_steps] = self.cur_d_blimit
		# self.state[self.initial_waiting_steps+1:self.predicting_steps+2] = self.cur_d_sb - (pred[0] - pred[1:])
		# self.state[self.predicting_steps + 2:] = self.cur_d_blimit + (pred[1:] - pred[0])

		done = self.if_gameover()
		reward = self.get_reward(done)

		if self.cur_limit > self.limit_min:
			self.cur_limit *= self.limit_decay

		return np.reshape(self.state, [self.state.shape[0], ]), reward, done, {}

	def plot(self, show_ani=False, show_motion=False):

		d_sb = np.array(self.d_sb_track)[-self.hit_steps:]
		h_len = np.array(self.hoist_len_track)[-self.hit_steps:]
		d_blimit = np.array(self.d_blimit_track)[-self.hit_steps:]

		if show_ani:
			fig, ax = plt.subplots()
			ax.set_ylim([-2, 7])
			lines = []
			lines.extend(ax.plot(self.init_h_s_ct - h_len[0], color='red', marker='x', markersize=15))
			lines.extend(ax.plot(self.init_h_s_ct - d_sb[0] - h_len[0], color='blue', marker='x', markersize=15))

			def animate(i):
				lines[0].set_ydata(self.init_h_s_ct - h_len[i])
				lines[1].set_ydata(self.init_h_s_ct - d_sb[i] - h_len[i])
				ax.set_title("time: %.1fs, d_sb: %.2fm" % (i * self.dt, d_sb[i]), fontsize=15)
				ax.legend(['block', 'barge'])
				return lines

			def init():
				lines[0].set_ydata(self.init_h_s_ct - h_len[0])
				lines[1].set_ydata(self.init_h_s_ct - d_sb[0] - h_len[0])
				return lines

			num_frame = len(d_sb)
			ani = animation.FuncAnimation(fig=fig, func=animate, frames=num_frame, init_func=init, interval=20,
			                              blit=False, repeat=False)
			plt.show()

		if show_motion:
			fig, ax = plt.subplots()
			ax.set_ylim([self.init_h_s_ct - h_len[-2]-1, self.init_h_s_ct - h_len[-2]+1])
			x = np.linspace(0, int(self.cur_step * self.dt), len(d_sb))[-self.hit_steps:]
			plt.plot(self.init_h_s_ct - h_len)
			plt.plot(self.init_h_s_ct - d_sb - h_len)
			plt.plot(self.init_h_s_ct - h_len + d_blimit)
			plt.xlabel('time(s)')
			plt.ylabel('distance (m)')
			plt.title("impact_velocity %.3f m/s" % self.final_imp_vel)
			plt.legend(['motion_block', 'motion_barge'])
			# plt.pause(3)
			# plt.close()
			plt.show()

	def save_file(self, file_dir, data):
		np.savetxt(file_dir, data, delimiter=',')

	def load_file(self, file_dir):
		data_log = np.loadtxt(file_dir, delimiter=',')
		return data_log

