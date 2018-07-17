from gym import spaces, Env
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from gym.utils import seeding
import spec_tools.spec_tools as st
import RL.tools.wave_tool as wavetool


class SetDown_reimpact_gym(Env):
	def __init__(self):

		## initial environment settings

		self.init_h_b_ct = 10  # initial distance between crane tip to barge
		self.init_hoist_len = 7 # initial hoist length of aux block
		# self.init_hoist_len = 9 # initial hoist length of aux block

		self.cur_limit = None # current limit on distance
		self.init_limit = self.init_h_b_ct - self.init_hoist_len + 3
		self.limit_decay = 0.998
		self.limit_min = 3

		self.max_speed = 9 / 60 # 100% RPM full speed down
		self.half_speed = 4.5 / 60 # 50% RPM
		self.accel = (4.5 / 60) / 3 # takes three second to increase speed to 50%; unit m/s2
		self.num_action = 3 # 0) reduce speed by 50%, 1) don't change speed, 2) increase speed by 50%
		self.dt = 0.2
		self.timeout = 600 # timeout at 600 seconds (3000 steps)
		self.hit_steps = 20 # steps of rewind before set-down

		self.obs_len = 0.2
		# self.initial_waiting_steps = 600  # initial waiting time for using AR
		self.initial_waiting_steps = 0  # initial waiting time for using AR

		self.pred_len = 15 # agent sees the actual change of distance for 15 second (appr. 2 cycles)
		# self.pred_len = 5 # agent sees the actual change of distance for 15 second (appr. 2 cycles)
		self.predicting_steps = int(self.pred_len / self.dt)
		self.num_input_wave_height = 8 # agent also sees the peak (crest and though) of rest 8 cycles
		# self.num_input_wave_height = 0 # agent also sees the peak (crest and though) of rest 8 cycles

		self.num_step = int(np.ceil(self.timeout / self.dt)) + 1 + self.initial_waiting_steps # total steps for the
		self.reimpact_step = 300

		# episode
		self.ramp_up = 3 # freezing time when action 1 and 2 are selected
		self.ramp_up_step = max(1, int(self.ramp_up / self.dt)) # corresponding holding steps

		self.rel_motion_bc = None # changes in distance between crane tip and barge generated from response of relative
		#  RAO
		# and input workable sea state
		self.rel_motion_bc_t = None # corresponding time frame

		# load RAO, get response
		self.wave = st.Spectrum.from_synthetic(spreading=None, Hs=1.5, Tp=8)
		self.relative_RAO = st.Rao.from_liftdyn('/home/michael/Desktop/workspace/rl_for_set_down/RL/RAO/relative_motion.stf_plt', 1, 1)
		self.resp = self.relative_RAO.get_response(self.wave)
		temp = self.resp.make_time_trace(self.num_step + 3000, self.dt)

		self.rel_motion_b_ct_t = temp['t']
		self.rel_motion_b_ct = temp['response']

		## instant variables
		self.cur_step = None # current step of the episode (will include initial waiting time)
		self.cur_t = None # actual time converted from cur_step
		self.seed()
		self.use_AR = False

		self.cur_d_cb = None # current distance between barge and block
		self.cur_d_climit = None # margin to the left
		self.prev_d_cb = None
		self.cur_hoist_length = None
		self.cur_speed = None # current speed of the hoist
		self.agent_active = False

		## experience variables
		# state[0]: cur_d_sb, state[1]: cur_speed, state[2:2+self.pred_len/self.dt]: distance changes,
		# state[-self.num_peaks:]: peaks of the following motion
		state_len = 2 + int(self.pred_len / self.dt) + self.num_input_wave_height
		self.high_limit = (self.init_h_b_ct - self.init_hoist_len + 10) * np.ones(state_len)
		self.action_space = spaces.Discrete(self.num_action)
		self.observation_space = spaces.Box(-self.high_limit, high=self.high_limit, dtype=np.float16)
		self.state = np.zeros([self.high_limit.shape[0]])
		self.sum_reward = 0
		self.all_wave_heights = None

		## logging variables
		self.hoist_len_track = []
		self.d_cb_track = []
		self.d_climit_track = []
		self.reimpact_b_track = None
		self.reimpact_c_track = None
		self.action_0_list = []
		self.action_1_list = []
		self.action_2_list = []

		## result variables
		self.final_imp_vel = None
		self.reimpact = False

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):

		self.hoist_len_track = []
		self.d_cb_track = []
		self.d_climit_track = []
		self.reimpact_b_track = None
		self.reimpact_c_track = None
		self.action_list = []

		self.cur_t = 0
		self.cur_step = 0

		self.cur_d_cb = self.init_h_b_ct - self.init_hoist_len
		self.cur_d_climit = self.init_limit - self.cur_d_cb

		self.cur_speed = 0

		self.prev_d_cb = self.cur_d_cb
		self.cur_hoist_length = self.init_hoist_len

		self.final_imp_vel = 0
		self.sum_reward = 0
		self.reimpact = False

		self.cur_limit = self.init_limit

		self.cur_step += self.initial_waiting_steps
		self.cur_t += round(self.cur_step * self.dt, 2)

		# generate the time trace of changes of distance between crane tip and barge
		temp = self.resp.make_time_trace(self.num_step+3000, self.dt)
		self.rel_motion_b_ct_t = temp['t']
		self.rel_motion_b_ct = temp['response'][0]


		pred0 = self.rel_motion_b_ct[self.cur_step:self.cur_step + self.predicting_steps]
		peaks = self.rel_motion_b_ct[self.cur_step + self.predicting_steps:self.cur_step+self.predicting_steps+1000]
		all_wave_heights = wavetool.return_wave_heights(peaks)

		# prediction by AR
		# AR_prediction = toolkit.computeAR(data=np.reshape(self.rel_motion_sc[0:self.initial_waiting_steps],
		#                                                   (1,self.initial_waiting_steps,1)),
		#                                   pred_len=self.predicting_steps+1)

		# first two elements are margin to right and left
		self.state[0] = self.cur_d_cb
		self.state[1] = self.cur_speed

		if not self.use_AR:
			# what is the distance in 2 cycles (15s, 75 steps) if no action
			self.state[2:self.predicting_steps + 2] = self.cur_d_cb + pred0
			# the absolute wave height for the next 8 cycles (8 values) if no action
			self.state[self.predicting_steps + 2:] = all_wave_heights[0:self.num_input_wave_height]

		# else:
		# 	predicted wave by AR
		# self.state[2:self.predicting_steps + 2] = self.cur_d_sb - (AR_prediction[1:] - AR_prediction[0])
		# self.state[self.predicting_steps + 2:] = self.cur_d_blimit + (AR_prediction[1:] - AR_prediction[0])

		self.state = np.reshape(self.state, [self.state.shape[0], ])
		return np.array(self.state)

	def reset_MCTS_env(self, cur_d_cb, cur_d_climit, cur_speed, prev_d_cb, cur_hoist_length,
	                   cur_limit,
	               rel_motion_b_ct_t, rel_motion_b_ct):

		# use this method to setup the env of MCTS sim
		# note that AR(initial waiting steps) is not implemented yet
		self.cur_t = 0
		self.cur_step = 0
		self.cur_d_cb = cur_d_cb
		self.cur_d_climit = cur_d_climit
		self.cur_limit = cur_limit
		self.cur_speed = cur_speed
		self.prev_d_cb = prev_d_cb
		self.cur_hoist_length = cur_hoist_length
		self.rel_motion_b_ct_t = rel_motion_b_ct_t
		self.rel_motion_b_ct = rel_motion_b_ct # relative motion onwards

		self.hoist_len_track = []
		self.d_cb_track = []
		self.d_climit_track = []
		self.reimpact_b_track = None
		self.reimpact_c_track = None
		self.action_list = []

		self.final_imp_vel = 0
		self.sum_reward = 0
		self.reimpact = False

		# generate the time trace of changes of distance between crane tip and barge
		# self.rel_motion_b_ct_t, self.rel_motion_b_ct = self.resp.make_time_trace(self.num_step + 3000, self.dt)
		pred0 = self.rel_motion_b_ct[self.cur_step:self.cur_step + self.predicting_steps]
		peaks = self.rel_motion_b_ct[self.cur_step + self.predicting_steps:self.cur_step + self.predicting_steps + 1000]
		all_wave_heights = wavetool.return_wave_heights(peaks)

		# first two elements are margin to right and cur speed
		self.state[0] = self.cur_d_cb
		self.state[1] = self.cur_speed

		# what is the distance in 2 cycles (15s, 75 steps) if no action
		self.state[2:self.predicting_steps + 2] = self.cur_d_cb + pred0
		# the absolute wave height for the next 8 cycles (8 values) if no action
		self.state[self.predicting_steps + 2:] = all_wave_heights[0:self.num_input_wave_height]

		return np.array(self.state)

	def get_extra_payout(self, start):
		vel = -np.linspace(self.cur_speed, self.max_speed, (self.max_speed-self.cur_speed)/(self.accel*self.dt)+1)
		x = np.linspace(0, int(self.reimpact_step*self.dt)-self.dt, self.reimpact_step)

		acc_period = x[0:len(vel)]
		y_acc = -0.5 * self.accel * acc_period * acc_period - self.cur_speed * acc_period + start

		con_period = x[len(vel):]-x[len(vel)-1]
		y_con = -self.max_speed*con_period + y_acc[-1]
		y = np.concatenate((y_acc,y_con))
		return y

	def is_reimpact(self):
		yb = - self.rel_motion_b_ct[self.cur_step: self.cur_step+self.reimpact_step]
		yc = self.get_extra_payout(self.cur_d_cb + (-self.rel_motion_b_ct[self.cur_step]))

		self.reimpact_b_track = yb
		self.reimpact_c_track = yc

		is_reimpact = ((yb - yc) < 0).any()
		return is_reimpact

	def get_reward(self, gameover):
		reward = 0
		if gameover:
			if self.cur_step < self.num_step and self.cur_d_cb < 0: # set-down
				vel = (self.prev_d_cb - self.cur_d_cb) / self.dt
				self.final_imp_vel = vel
				if 0 < vel < 0.5: # good one
					reward = min(10 * 1 / vel, 200)
					self.final_imp_vel = vel

				else: # bad one
					reward = -30

					# reward = max(-200*(vel-0.5), -99)

			if self.cur_d_cb > self.cur_limit: # hit boundary
				reward = -20

			self.reimpact = self.is_reimpact()



			if self.reimpact:
				reward = -100

		self.sum_reward += reward

		return reward

	def if_gameover(self):

		gameover = False
		# uncomment for both margins
		if self.cur_d_cb < 0 or self.cur_step == self.num_step or self.cur_d_cb > self.cur_limit:
			# if self.cur_d_sb < 0 or self.cur_step == self.num_step:
			gameover = True
		return gameover

	def get_holding_step(self, action):
		if action == 1:
			return 1
		elif self.cur_speed == self.max_speed and action == 2:
			return 1
		elif self.cur_speed == 0 and action == 0:
			return 1
		else:
			return 15

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

		self.action_list.append([self.cur_step, action])

		d_cb = self.cur_d_cb
		hoist_len = self.cur_hoist_length
		self.prev_d_cb = d_cb

		holding_step = self.get_holding_step(action)

		for k in range(holding_step):

			self.hoist_len_track.append(hoist_len)
			self.d_cb_track.append(d_cb)
			self.d_climit_track.append(self.cur_d_climit)

			# action=0:reduce; action=1:no action; action=2:gas
			# action-1 == [-1, 0, 1]
			if self.cur_speed == self.max_speed:
				self.cur_speed = min(self.cur_speed + (action - 1) * self.accel * self.dt, self.max_speed)
			else:
				self.cur_speed = max(self.cur_speed + (action - 1) * self.accel * self.dt, 0)

			self.cur_speed = min(self.cur_speed, self.max_speed)
			hoist_len = max(hoist_len + self.cur_speed * self.dt, 0)

			self.cur_hoist_length = hoist_len

			h_b_ct = self.init_h_b_ct + np.interp(self.cur_t, self.rel_motion_b_ct_t, self.rel_motion_b_ct)
			d_cb = h_b_ct - hoist_len

			self.cur_d_cb = d_cb
			self.cur_d_climit = self.cur_limit - self.cur_d_cb

			if self.if_gameover():
				self.hoist_len_track.append(hoist_len)
				self.d_cb_track.append(d_cb)
				self.d_climit_track.append(self.cur_d_climit)
				break

			self.cur_step += 1
			self.cur_t = round(self.cur_step * self.dt, 2)

		# if self.use_AR:
		# 	pred = toolkit.computeAR(data=np.reshape(self.rel_motion_sc[
		# 	                                            self.cur_step-self.initial_waiting_steps:self.cur_step],
		# 	                                                  (1,self.initial_waiting_steps,1)),
		# 	                                  pred_len=self.predicting_steps+1)

		# else:
		pred = self.rel_motion_b_ct[self.cur_step:self.cur_step + self.predicting_steps]

		peak_wave = self.rel_motion_b_ct[self.cur_step + self.predicting_steps:
											self.cur_step + self.predicting_steps + 1000]

		wave_heights = wavetool.return_wave_heights(peak_wave)

		self.state[0] = self.cur_d_cb
		self.state[1] = self.cur_speed

		self.state[2:self.predicting_steps + 2] = self.cur_d_cb - pred
		self.state[self.predicting_steps + 2:] = wave_heights[0:self.num_input_wave_height]

		done = self.if_gameover()
		reward = self.get_reward(done)

		if self.cur_limit > self.limit_min:
			self.cur_limit *= self.limit_decay

		return np.reshape(self.state, [self.state.shape[0], ]), reward, done, {}

	def plot(self, show_ani=False, show_motion=False):

		d_cb = np.array(self.d_cb_track)
		h_len = np.array(self.hoist_len_track)
		reimpact_b = self.reimpact_b_track[1:75]
		reimpact_c = self.reimpact_c_track[1:75]

		action_list = np.array(self.action_list)

		action_0 = action_list[np.where(action_list[:, 1] == 0)]
		action_1 = action_list[np.where(action_list[:, 1] == 1)]
		action_2 = action_list[np.where(action_list[:, 1] == 2)]

		if show_ani:
			fig, ax = plt.subplots()
			ax.set_ylim([-2, 12])
			lines = []
			lines.extend(ax.plot(self.init_h_b_ct - h_len[0], color='red', marker='x', markersize=15))
			lines.extend(ax.plot(self.init_h_b_ct - d_cb[0] - h_len[0], color='blue', marker='x', markersize=15))

			def animate(i):
				lines[0].set_ydata(self.init_h_b_ct - h_len[i])
				lines[1].set_ydata(self.init_h_b_ct - d_cb[i] - h_len[i])
				ax.set_title("time: %.1fs, d_sb: %.2fm" % (i * self.dt, d_cb[i]), fontsize=15)
				ax.legend(['block', 'barge'])
				return lines

			def init():
				lines[0].set_ydata(self.init_h_b_ct - h_len[0])
				lines[1].set_ydata(self.init_h_b_ct - d_cb[0] - h_len[0])
				return lines

			num_frame = len(d_cb)
			ani = animation.FuncAnimation(fig=fig, func=animate, frames=num_frame, init_func=init, interval=20,
			                              blit=False, repeat=False)
			plt.show()

		if show_motion:
			fig, ax = plt.subplots()
			# ax.set_ylim([self.init_h_b_ct - h_len[-2] - 1, self.init_h_b_ct - h_len[-2] + 1])
			# x = np.linspace(0, int(self.cur_step * self.dt), len(d_cb))[-self.hit_steps:]
			ax.set_ylim(-2,4)
			y_cargo = np.concatenate((self.init_h_b_ct - h_len, reimpact_c))
			y_barge = np.concatenate((self.init_h_b_ct-d_cb-h_len, reimpact_b))

			y_action_0 = y_cargo[action_0[:,0]]
			y_action_1 = y_cargo[action_1[:, 0]]
			y_action_2 = y_cargo[action_2[:, 0]]

			x = np.linspace(0, int(len(y_barge))*self.dt, len(y_barge))
			plt.plot(x, y_cargo)
			plt.plot(x, y_barge)
			plt.plot(action_0[:, 0] * self.dt, y_action_0, 'o')
			plt.plot(action_1[:, 0] * self.dt, y_action_1, 'o')
			plt.plot(action_2[:, 0] * self.dt, y_action_2, 'o')

			# plt.plot(self.init_h_b_ct - h_len + d_climit)
			plt.xlabel('time(s)')
			plt.ylabel('distance (m)')
			plt.title("impact_velocity %.3f m/s" % self.final_imp_vel)
			plt.legend(['motion_block', 'motion_barge', 'brake', 'no action', 'gas'])
			# plt.pause(3)
			# plt.close()
			plt.show()
