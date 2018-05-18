
import gym
from gym import spaces
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from gym.utils import seeding
import spec_tools.spec_tools as st


class Following_gym(gym.Env):

	"""
	s: supply boat
	b: block
	c: crane top
	rel_motion_sc: motion between supply boat and crane (m)
	h_barge_hoist: inital distance between crane top and supply boat (m)
	cur_d_sb: current distance between supply boat and block (m)
	prev_d_sb: distance between supply boat and block at previous time step (m)
	obs_len: length of observation that agent will see (s)
	initial_waiting_steps: corresponding time steps of obs_len
	"""

	def __init__(self):
		self.init_h_s_ct = 5
		self.init_hoist_len = 4.8
		self.cur_limit = None
		self.init_limit = self.init_h_s_ct - self.init_hoist_len + 1.5
		self.limit_decay = 0.9
		self.limit_min = 0.2

		self.lowering_speed = 12/60
		self.lifting_speed = 12/60
		self.dt = 0.2
		self.timeout = 200

		self.num_step = int(np.ceil(self.timeout/self.dt))+1
		self.cur_step = None
		self.t = None

		self.holding_length = 0.6 # skip 3 frames per action
		self.holding_step = max(1, int(self.holding_length / self.dt))

		self.seed()

		self.obs_len = 0.2
		self.initial_waiting_steps = int(self.obs_len/self.dt) # waiting time for the first observation
		self.pred_len = 1
		self.predicting_steps = int(self.pred_len/self.dt)

		self.rel_motion_sc = None
		self.rel_motion_sc_t = None

		self.resp = st.Spectrum.from_synthetic(spreading=None, Hs=0.3, Tp=20)

		self.cur_d_sb = None # current distance between supply boat and block (margin for the right)
		self.cur_d_blimit = None # margin to the left
		self.prev_d_sb = None
		self.cur_hoist_length = None

		self.high_limit = (self.init_h_s_ct - self.init_hoist_len + 10)*np.ones(2*(int(self.obs_len/self.dt)+int(self.pred_len/self.dt)))
		# self.high_limit = (self.init_h_s_ct - self.init_hoist_len + 10)*np.ones(1*(int(self.obs_len/self.dt)+int(self.pred_len/self.dt)))
		# self.action_space = spaces.Discrete(3)
		self.action_space = spaces.Discrete(13)

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

		self.rel_motion_sc_t, self.rel_motion_sc = self.resp.make_time_trace(self.num_step + 200, self.dt)
		cur_motion_s = self.rel_motion_sc[0:self.predicting_steps]

		# first two elements are margin to right and left
		self.state[self.initial_waiting_steps-1] = self.cur_d_sb
		self.state[self.initial_waiting_steps] = self.cur_d_blimit

		#prediction of right margin
		self.state[self.initial_waiting_steps+1:self.predicting_steps+2] = self.cur_d_sb - cur_motion_s
		# self.state[self.initial_waiting_steps:] = self.cur_d_sb - cur_motion_s

		#prediction of left margin
		self.state[self.predicting_steps+2:] = self.cur_d_blimit + cur_motion_s

		self.state = np.reshape(self.state, [self.state.shape[0],])
		return np.array(self.state)

	def get_reward(self, gameover):
		reward = 0
		if not gameover:
			# reward = min(0.5*1 / np.abs(self.cur_d_sb), 10)
			reward = min(0.1*2/ np.abs(self.cur_d_sb-self.cur_d_blimit), 10)
		if gameover and self.cur_step < self.num_step:
			reward = -100
		self.sum_reward += reward
		return reward

	def if_gameover(self):

		gameover = False

		if self.cur_d_sb < 0 or self.cur_d_blimit < 0 or self.cur_step == self.num_step:
			gameover = True

		return gameover

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

		d_sb = self.cur_d_sb
		hoist_len = self.cur_hoist_length

		self.prev_d_sb = d_sb

		self.holding_step = abs(action)

		for k in range(self.holding_step):

			self.hoist_len_track.append(hoist_len)
			self.d_sb_track.append(d_sb)
			self.d_blimit_track.append(self.cur_d_blimit)

			# if action == 1: # right,payout
			# 	hoist_len = hoist_len + self.lowering_speed * self.dt
			#
			# elif action == 2: # left,haulin
			# 	hoist_len = max(hoist_len - self.lifting_speed * self.dt, 0)
			# else:
			# 	pass

			speed = (action - 6) / 30
			speed = np.linspace(0, speed, self.holding_step)[k]

			hoist_len = max(hoist_len + speed * self.dt, 0)

			self.cur_hoist_length = hoist_len

			h_sc = self.init_h_s_ct - np.interp(self.t, self.rel_motion_sc_t, self.rel_motion_sc)
			d_sb = h_sc - hoist_len

			self.cur_d_sb = d_sb
			self.cur_d_blimit = self.cur_limit - self.cur_d_sb

			if self.if_gameover():
				break

			self.cur_step += 1
			self.t = round(self.cur_step * self.dt, 2)

		pred = self.rel_motion_sc[self.cur_step + 1:self.cur_step + 1 + self.predicting_steps]

		self.state[self.initial_waiting_steps - 1] = self.cur_d_sb
		self.state[self.initial_waiting_steps] = self.cur_d_blimit

		self.state[2:self.predicting_steps + 2] = self.cur_d_sb - pred
		# self.state[1:self.predicting_steps + 1] = self.cur_d_sb - pred

		self.state[self.predicting_steps + 2:] = self.cur_d_blimit + pred

		done = self.if_gameover()
		reward = self.get_reward(done)

		if self.cur_limit > self.limit_min:
			self.cur_limit *= self.limit_decay
		return np.reshape(self.state, [self.state.shape[0], ]), reward, done, {}

	def plot(self, show_ani=False, show_motion=False):

		d_sb = np.array(self.d_sb_track)
		h_len = np.array(self.hoist_len_track)
		d_blimit = np.array(self.d_blimit_track)

		if show_ani:
			fig, ax = plt.subplots()
			ax.set_ylim([-2, 7])
			lines = []
			lines.extend(ax.plot(self.init_h_s_ct - h_len[0], color = 'red', marker = 'x', markersize=15))
			lines.extend(ax.plot(self.init_h_s_ct - d_sb[0] - h_len[0], color = 'blue', marker = 'x', markersize=15))
			lines.extend(ax.plot(self.init_h_s_ct - d_sb[0] + d_blimit[0], color = 'green', marker = 'x',
			                     markersize=15))

			def animate(i):
				lines[0].set_ydata(self.init_h_s_ct-h_len[i])
				lines[1].set_ydata(self.init_h_s_ct-d_sb[i]-h_len[i])
				lines[2].set_ydata(self.init_h_s_ct-h_len[i] + d_blimit[i])
				ax.set_title("time: %.1fs, d_sb: %.2fm" % (i*self.dt, d_sb[i]), fontsize=15)
				ax.legend(['block', 'barge', 'limit'])
				return lines

			def init():
				lines[0].set_ydata(self.init_h_s_ct-h_len[0])
				lines[1].set_ydata(self.init_h_s_ct-d_sb[0]-h_len[0])
				lines[2].set_ydata(self.init_h_s_ct - h_len[0]+d_blimit[0])
				return lines

			num_frame = len(d_sb)
			ani = animation.FuncAnimation(fig=fig, func=animate, frames=num_frame, init_func=init, interval=20,
			                              blit=False, repeat=False)
			plt.show()

		if show_motion:
			fig, ax = plt.subplots()
			ax.set_ylim([-0.5, 1.5])
			x = np.linspace(0, int(self.cur_step * self.dt), len(d_sb))
			plt.plot(x, self.init_h_s_ct - h_len)
			plt.plot(x, self.init_h_s_ct - d_sb - h_len)
			plt.plot(x, self.init_h_s_ct - h_len + d_blimit)
			plt.xlabel('time(s)')
			plt.ylabel('distance (m)')
			plt.title("followed %d s" % int(self.t))
			plt.legend(['motion_block', 'motion_barge', 'limit'])
			plt.show()

	def save_file(self, file_dir, data):
		np.savetxt(file_dir, data, delimiter=',')

	def load_file(self, file_dir):
		data_log = np.loadtxt(file_dir, delimiter=',')
		return data_log

