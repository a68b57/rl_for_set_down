
import gym
from gym import spaces
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class SetDown(gym.Env):

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


	# TODO:rendering needs to be determined later
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def __init__(self):
		self.init_h_s_ct = 10
		self.init_hoist_len = 3
		self.lowering_speed = 3/60   # 3 meters per minutes, 0.05 meters per second
		self.lifting_speed = 3/60
		self.dt = 0.2
		self.timeout = 3600

		self.num_step = int(np.ceil(self.timeout/self.dt))+1
		self.cur_step = None
		self.t = None

		self.holding_length = 2
		self.holding_step = int(self.holding_length / self.dt)

		self.act_time = None

		self.obs_len = 30
		self.initial_waiting_steps = int(self.obs_len/self.dt) # waiting time for the first observation
		self.pred_len = 0
		self.predicting_steps = int(self.pred_len/self.dt)

		# set the upper bound of the observation, the observation for now is the relative distance(current timestep)
		# between barge and cargo which is relative distance between cargo and hoist minus hoist length
		# another observation is hoist length

		self.rel_motion_sc = None # relative motion between supply boat and crane
		self.rel_motion_sc_t = None
		self.cur_d_sb = None # current distance between supply boat and block
		self.prev_d_sb = None
		self.cur_hoist_length = None
		self.final_imp_vel = None


		# 3 is arbitrary number as the compensation of the highest wave mag.

		# for one-step obs
		# self.high_limit = np.array([self.init_h_s_ct - self.init_hoist_len + 3, 10])

		# for raw motion obs
		self.high_limit = (self.init_h_s_ct - self.init_hoist_len + 10)*np.ones(int((self.obs_len+self.pred_len)/self.dt) + 1)

		# for velocity and position obs
		# self.high_limit = np.array([100, self.init_h_s_ct + 10, self.init_hoist_len + 10])

		self.action_space = spaces.Discrete(3)

		# for one-step obs
		# self.observation_space = spaces.Box(np.array([0, 3]), self.height_limit)

		# for vel/raw obs
		self.observation_space = spaces.Box(-self.high_limit, high=self.high_limit, dtype=np.float64)

		self.state = np.zeros([self.initial_waiting_steps + self.predicting_steps + 1])

	def restart_episode(self, resp, change_wave):

		if change_wave:
			self.rel_motion_sc_t, self.rel_motion_sc = resp.make_time_trace(self.num_step, self.dt)

		self.t = 0
		self.cur_step = 0
		self.cur_d_sb = self.init_h_s_ct - self.init_hoist_len
		self.prev_d_sb = self.cur_d_sb
		self.cur_hoist_length = self.init_hoist_len

		# for row obs
		# self.state[-2] = self.init_h_s_ct - self.init_hoist_len
		# self.state[-1] = self.init_hoist_len
		# return np.reshape(self.state, self.initial_waiting_steps + 1)

		# for row + pred
		self.state[self.initial_waiting_steps-1] = self.init_h_s_ct - self.init_hoist_len
		# self.state[self.initial_waiting_steps - 1] = 0
		self.state[-1] = self.init_hoist_len
		self.state[self.initial_waiting_steps:-1] = self.init_h_s_ct - self.rel_motion_sc[0:self.predicting_steps]
		# self.state[self.initial_waiting_steps:-1] = self.rel_motion_sc[0:self.prediting_steps]

		return np.reshape(self.state, [1,self.initial_waiting_steps + self.predicting_steps + 1])

		# for one-step obs
		# self.state = np.array([self.init_h_s_ct, self.init_hoist_len])
		# return np.reshape(self.state, [1, 2])

		# for vel obs
		# self.state = np.array([0, self.init_h_s_ct, self.init_hoist_len])
		# return np.reshape(self.state, [1,3])

	def get_act(self, agent):

		action = None

		if self.cur_step < self.initial_waiting_steps:
			action = 0
			if self.cur_step + 1 == self.initial_waiting_steps:
				self.act_time = self.cur_step + 1

		elif self.cur_step != self.act_time:
			action = agent.action_temp

		elif self.cur_step == self.act_time:
			action = agent.act(self.state)

			if action != 0:
				self.act_time = self.cur_step + self.holding_step
				agent.action_temp = action     #lock the new action at act_time
				agent.state_temp = self.state  # lock the state at act_time
			else:
				self.act_time = self.cur_step + 1

		return action

	def get_reward(self, done):
		#TODO: redefine reward function, proper reward each step
		reward = 0
		imp_vel = 0

		# penalty for each time step and no penalty for the first waiting period
		if not done and self.t > (self.init_h_s_ct - self.init_hoist_len) / self.lowering_speed:
			reward = -0.001

		if not done and self.cur_d_sb < self.prev_d_sb:
			reward = 0.001

		if done:
			# if it is really set-down,  +30 reward anyway, further reward/punishment depending on imp velocity
			if int(self.t) < self.timeout:
				reward = 20
			# here also evaluate d_sb for the case of timeout,
			# such that agent also get reward when he is close to success
			if self.t == self.timeout:
				if self.cur_d_sb <= 7:
					reward = 3/self.cur_d_sb - 10
				else:
					reward = -self.cur_d_sb - 10

			# regular evaluation on impact velocity
			else:
				imp_vel = (self.prev_d_sb - self.cur_d_sb)/self.dt
				if imp_vel > 0.4:
					reward = -10*imp_vel + reward
					# reward = -10
				elif imp_vel >= 0:
					reward = 10/imp_vel + reward
					# reward = 10
				self.final_imp_vel = round(imp_vel, 3)

		return imp_vel, reward

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
		state = self.state

		# for row motion obs
		# d_sb = state[-2]
		# hoist_len = state[-1]
		# self.prev_d_sb = d_sb

		# for row motion obs + pred
		d_sb = state[self.initial_waiting_steps-1]
		hoist_len = state[-1]
		self.prev_d_sb = d_sb
		# self.prev_d_sb = self.init_h_s_ct - d_sb - self.cur_hoist_length

		# for vel obs
		# prev_h_vel = state[0]
		# prev_h_sc = state[1]
		# hoist_len = state[2]
		# self.prev_d_sb = prev_h_sc - hoist_len

		if action == 1: # lower
			hoist_len = hoist_len + self.lowering_speed * self.dt

		elif action == 2: # lift
			hoist_len = max(hoist_len - self.lifting_speed * self.dt, 0)
			#TODO: assign small penalty to agent if he wants to over haul the hoist (hoist_len<0)
		else: # action == 0 # hold, do nothing
			pass

		self.cur_hoist_length = hoist_len

		h_sc = self.init_h_s_ct - np.interp(self.t, self.rel_motion_sc_t, self.rel_motion_sc)
		d_sb = h_sc - hoist_len

		self.cur_d_sb = d_sb

		# update states for row motion obs
		# self.state[:-2] = self.state[1:-1]
		# self.state[-1] = hoist_len
		# self.state[-2] = d_sb

		pred_period = np.linspace(self.t+self.dt,self.t+self.dt+self.pred_len, self.predicting_steps)
		pred = self.init_h_s_ct - np.interp(pred_period, self.rel_motion_sc_t, self.rel_motion_sc) - self.cur_hoist_length

		# update states for row + pred obs
		self.state[-1] = hoist_len
		self.state[0:self.initial_waiting_steps-1] = self.state[1:self.initial_waiting_steps]

		self.state[self.initial_waiting_steps-1] = d_sb
		# self.state[self.initial_waiting_steps - 1] = np.interp(self.t, self.rel_motion_sc_t, self.rel_motion_sc)

		self.state[self.initial_waiting_steps:-1] = pred


		# update states for vel obs
		# self.state = np.array([(h_sc - prev_h_sc)/self.dt, h_sc, hoist_len])

		done = d_sb < 0 or self.num_step == self.cur_step
		done = bool(done)
		imp_vel, reward = self.get_reward(done)

		self.cur_step += 1
		self.t = round(self.cur_step*self.dt, 2)

		return np.reshape(self.state, [1, self.initial_waiting_steps + self.predicting_steps + 1]), reward, done, imp_vel


	def plot(self, d_sb, h_len, show_ani=False, show_motion=False):

		if show_ani:
			fig, ax = plt.subplots()
			ax.set_ylim([-2, 12])
			lines = []
			lines.extend(ax.plot(self.init_h_s_ct - h_len[0], color = 'red', marker = 'x', markersize=15))
			lines.extend(ax.plot(self.init_h_s_ct - d_sb[0] - h_len[0], color = 'blue', marker = 'x', markersize=15))


			def animate(i):
				lines[0].set_ydata(self.init_h_s_ct-h_len[i])
				lines[1].set_ydata(self.init_h_s_ct-d_sb[i]-h_len[i])
				ax.set_title("time: %.1fs, d_sb: %.2fm" % (i*self.dt, d_sb[i]), fontsize=15)
				ax.legend(['block', 'barge'])
				return lines

			def init():
				lines[0].set_ydata(self.init_h_s_ct-h_len[0])
				lines[1].set_ydata(self.init_h_s_ct-d_sb[0]-h_len[0])
				return lines

			num_frame = len(d_sb)
			ani = animation.FuncAnimation(fig=fig, func=animate, frames=num_frame, init_func=init, interval=20,
			                              blit=False, repeat=False)
			plt.show()

		if show_motion:
			fig, ax = plt.subplots()
			ax.set_ylim([-2, 12])
			x = np.linspace(0, int(self.cur_step * self.dt), len(d_sb))
			plt.plot(x, self.init_h_s_ct - h_len)
			plt.plot(x, self.init_h_s_ct - d_sb - h_len)
			plt.xlabel('time(s)')
			plt.ylabel('distance (m)')
			plt.title("imp_vel: %.3f m/s in %d s" % (self.final_imp_vel, self.t))
			plt.legend(['motion_barge', 'motion_block'])
			plt.show()

	def save_file(self, file_dir, data):
		np.savetxt(file_dir, data, delimiter=',')

