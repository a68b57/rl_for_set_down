import gym
from gym import spaces
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from gym.utils import seeding
import spec_tools.spec_tools as st


class HRL_gym(gym.Env):

	def __init__(self, init_barge_ct, init_hoist_len, limit_decay, limit_min, dt, eps_timeout, goal_timeout, obs_len,
	             pred_len, hs, tp, num_actions, num_goals):

		# geometry
		self.init_h_s_ct = init_barge_ct
		self.init_hoist_len = init_hoist_len
		self.cur_limit = None
		self.init_limit = self.init_h_s_ct - self.init_hoist_len + 5
		self.limit_decay = limit_decay
		self.limit_min = limit_min

		self.cur_d_sb = None # current distance between supply boat and block (margin for the right)
		self.cur_d_blimit = None # margin to the left
		self.prev_d_sb = None
		self.cur_hoist_length = None
		self.final_imp_vel = 0

		# timing meta-controller
		self.dt = dt
		self.hit_steps = 5 # visualize last n steps before touching
		self.max_eps_num_step = int(np.ceil(eps_timeout / self.dt)) + 1
		self.max_goal_num_step = int(np.ceil(goal_timeout / self.dt)) + 1
		self.cur_step = None
		self.t = None

		# param meta-controller
		self.num_goals = num_goals
		self.num_survival_modes = (self.num_goals-1)//2
		self.eps_completed = None
		self.eps_over = None
		self.goal_space = spaces.Discrete(num_goals)
		self.goal = None # one-hot format

		# timing controller
		self.goal_t = None
		self.goal_cur_step = None
		self.goal_total_step = None

		self.holding_length = 0 # skip 5 frames per action
		self.holding_step = max(1, int(self.holding_length / self.dt))
		self.seed()

		# param controller
		self.num_actions = num_actions
		self.action_space = spaces.Discrete(num_actions)
		self.goal_completed = None
		self.goal_over = None

		# states
		self.obs_len = obs_len
		self.initial_waiting_steps = int(self.obs_len / self.dt) # waiting time for the first observation
		self.pred_len = pred_len
		self.predicting_steps = int(self.pred_len / self.dt)
		self.high_limit = (self.init_h_s_ct - self.init_hoist_len + 10) * np.ones(
			2 * (int(self.obs_len / self.dt) + int(self.pred_len / self.dt)))
		self.observation_space = spaces.Box(-self.high_limit, high=self.high_limit, dtype=np.float16)
		self.state = np.zeros([1, self.high_limit.shape[0]])

		# wave and motion
		self.resp = st.Spectrum.from_synthetic(spreading=None, Hs=hs, Tp=tp)
		self.rel_motion_sc_t, self.rel_motion_sc = self.resp.make_time_trace(self.max_eps_num_step + 200, self.dt)

		# logging
		self.hoist_len_track = []
		self.d_sb_track = []
		self.d_blimit_track = []

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		# logging
		self.hoist_len_track = []
		self.d_sb_track = []
		self.d_blimit_track = []

		# timing meta-controller
		self.t = 0
		self.cur_step = 0

		# geometry
		self.cur_d_sb = self.init_h_s_ct - self.init_hoist_len
		self.cur_d_blimit = self.init_limit - self.cur_d_sb
		self.prev_d_sb = self.cur_d_sb
		self.cur_hoist_length = self.init_hoist_len
		self.cur_limit = self.init_limit
		self.final_imp_vel = 0

		# wave and motion
		self.rel_motion_sc_t, self.rel_motion_sc = self.resp.make_time_trace(self.max_eps_num_step + 200, self.dt)
		cur_motion_s = self.rel_motion_sc[self.initial_waiting_steps-1:self.predicting_steps]

		# states
		# first two elements are margin to right and left
		self.state[self.initial_waiting_steps - 1] = self.cur_d_sb
		self.state[self.initial_waiting_steps] = self.cur_d_blimit
		#prediction of right margin
		self.state[self.initial_waiting_steps + 1:self.predicting_steps + 2] = self.cur_d_sb - (self.rel_motion_sc[1:self.predicting_steps+1]-cur_motion_s)
		#prediction of left margin
		self.state[self.predicting_steps + 2:] = self.cur_d_blimit + (self.rel_motion_sc[1:self.predicting_steps+1]-cur_motion_s)
		# self.state = np.reshape(self.state, [self.state.shape[0], ])
		self.state = np.reshape(self.state, [1, self.state.shape[0]])


		# controllers param
		self.goal_completed = False
		self.eps_completed = False
		self.goal_over = False
		self.eps_over = False

		return np.array(self.state)

	def goal_int(self):
		"""convert from one_hot to int"""
		return np.argmax(self.goal)

	def intp_goal(self):
		"""translate goal int to survival time"""
		max_t = (self.num_goals - 1) // 2
		if self.goal_int() != self.num_goals - 1:
			time = self.goal_int() % max_t + 1
			return time * self.dt
		else:
			return False

	def get_intrinsic_reward(self):
		"""
			intrinsic step reward of controller on games
			full-speed-down; following; set-down
		"""
		reward = 0
		goal_int = self.goal_int()
		if self.intp_goal():
			if goal_int < self.num_survival_modes:
				# full-speed down game
				if not self.goal_completed:
					reward = min(0.5 * 1 / np.abs(self.cur_d_sb), 10)
				if self.goal_over:
					reward = -100
			else:
				# following game
				if not self.goal_completed:
					reward = min(0.1*2/ np.abs(self.cur_d_sb-self.cur_d_blimit), 10)
				if self.goal_over:
					reward = -100
		else:
			# set-down game
			if self.goal_completed:
				vel = (self.prev_d_sb - self.cur_d_sb) / self.dt
				if 0 < vel < 0.3: # good one
					reward = 10*1/vel
					print(vel)
					self.final_imp_vel = vel
				else:
					reward = -30
			if self.goal_over:
				reward = -30
		return reward

	def get_extrinsic_reward(self):
		"""
			extrinsic step reward of meta-controller
			for completing goals and have imp_vel(identical to intrinsic reward)
		"""
		reward = 0
		if not self.intp_goal():
			reward = self.get_intrinsic_reward()
		if self.goal_completed:
			reward += 1
		return reward

	# goal is over if agent hit walls or exceeds goal timeout
	def is_goal_over(self):
		"""goal is over either timeout for set-down or touch limits for following and full-speed down"""
		goal_over = False
		if self.intp_goal():
			# for following and full speed down
			if self.cur_d_sb >= self.cur_d_blimit or self.cur_d_sb < 0:
				goal_over = True
		if not self.intp_goal():
			# for set-down
			if self.cur_step >= self.max_goal_num_step:
				goal_over = True
		self.goal_over = goal_over

	# eps is over as soon as goal is over
	def is_eps_over(self):
		"""episode is over then goal is over or episode step meets timeout"""
		eps_over = False
		if self.cur_step == self.max_eps_num_step or self.goal_over:
			eps_over = True
		self.eps_over = eps_over

	# goal is reached either meet survival time or set-down
	def is_goal_completed(self):
		"""goal is completed when goal step meets required survival time or set-down"""
		goal_reached = False
		goal_steps = self.intp_goal()
		if goal_steps:
			if self.goal_cur_step == goal_steps:
				goal_reached = True
		if not goal_steps:
			if self.cur_d_sb < 0:
				goal_reached = True
		self.goal_completed = goal_reached

	def is_eps_completed(self):
		"""eps is completed only when set-down"""
		eps_completed = False
		if self.cur_d_sb < 0 and not self.eps_over:
			eps_completed = True
		self.eps_completed = eps_completed

	def update_status(self):
		self.is_goal_over()
		self.is_goal_completed()
		self.is_eps_over()
		self.is_eps_completed()

	def set_goal(self, goal):
		self.goal = goal
		self.goal_total_step = self.intp_goal()

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

		d_sb = self.cur_d_sb
		hoist_len = self.cur_hoist_length
		self.prev_d_sb = d_sb

		for k in range(self.holding_step):
			# log geometry before action
			self.hoist_len_track.append(hoist_len)
			self.d_sb_track.append(d_sb)
			self.d_blimit_track.append(self.cur_d_blimit)

			# compute geometry after taking action
			speed = (action-(self.num_actions-1)//2)/30
			hoist_len = max(hoist_len + speed * self.dt, 0)
			self.cur_hoist_length = hoist_len
			h_sc = self.init_h_s_ct - np.interp(self.t, self.rel_motion_sc_t, self.rel_motion_sc)
			d_sb = h_sc - hoist_len
			self.cur_d_sb = d_sb
			self.cur_d_blimit = self.cur_limit - self.cur_d_sb

			# update status
			self.update_status()
			if self.goal_completed or self.goal_over:
				self.hoist_len_track.append(hoist_len)
				self.d_sb_track.append(d_sb)
				self.d_blimit_track.append(self.cur_d_blimit)
				break

			self.cur_step += 1
			self.t = round(self.cur_step * self.dt, 2)

		# prepare the transition for return
		# compute the next state after taking action at current state
		pred = self.rel_motion_sc[self.cur_step:self.cur_step + self.predicting_steps+1]

		# first and second elements are margins to both sides
		self.state[self.initial_waiting_steps - 1] = self.cur_d_sb
		self.state[self.initial_waiting_steps] = self.cur_d_blimit

		# changes of both margins
		self.state[self.initial_waiting_steps+1:self.predicting_steps+2] = self.cur_d_sb + (pred[0] - pred[1:])
		self.state[self.predicting_steps + 2:] = self.cur_d_blimit + (pred-self.rel_motion_sc[self.cur_step:self.cur_step + self.predicting_steps])

		intrinsic_reward = self.get_intrinsic_reward()

		if self.cur_limit > self.limit_min:
			self.cur_limit *= self.limit_decay

		# return transition
		return np.reshape(self.state, [self.state.shape[0], ]), intrinsic_reward, self.goal_completed, \
		       self.goal_over, self.eps_completed, self.eps_over

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
			ax.set_ylim([self.init_h_s_ct - h_len[-2]-0.2, self.init_h_s_ct - h_len[-2]+0.2])
			x = np.linspace(0, int(self.cur_step * self.dt), len(d_sb))[-self.hit_steps:]
			plt.plot(self.init_h_s_ct - h_len)
			plt.plot(self.init_h_s_ct - d_sb - h_len)
			plt.plot(self.init_h_s_ct - h_len + d_blimit)
			plt.xlabel('time(s)')
			plt.ylabel('distance (m)')
			plt.title("impact_velocity %.3f m/s" % self.final_imp_vel)
			plt.legend(['motion_block', 'motion_barge'])
			plt.pause(3)
			plt.close()

	def save_file(self, file_dir, data):
		np.savetxt(file_dir, data, delimiter=',')

	def load_file(self, file_dir):
		data_log = np.loadtxt(file_dir, delimiter=',')
		return data_log