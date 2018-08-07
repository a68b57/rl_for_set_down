from gym import spaces, Env
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from gym.utils import seeding
import spec_tools.spec_tools as st
import RL.tools.wave_tool as wavetool

from RL.envs.set_down_hmc_physics import SimpleSetDownGame

class SetDown_hmc_insite(Env):

	def __init__(self):
		self.game = SimpleSetDownGame()
		wave = st.Spectrum.from_synthetic(spreading=None, Hs=1.5, Tp=8)
		relative_RAO = st.Rao.from_liftdyn('./RAO/relative_motion.stf_plt', 1, 1)

		self.game.spectrum = relative_RAO.get_response(wave)

		self.game.s_timeout = 200  # seconds
		self.game.initial_hoist_length = 9
		self.game.n_action_hold = 1
		self.game.lowering_speed = 0.01
		self.game.dt = 0.2

		self.pred_len = 5 # in seconds
		self.pred_steps = int(self.pred_len/self.game.dt)

		# current distance between crane tip to barge, current speed, current length, relative distance in 2 seconds
		state_len = 1 + 1 + 1 + self.pred_steps
		self.high_limit = np.inf * np.ones(state_len)
		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Box(-self.high_limit, high=self.high_limit, dtype=np.float16)
		self.state = np.zeros([self.high_limit.shape[0]])

	def reset(self):
		self.game.new_game()
		self.state[0] = self.game.distance_crane_to_barge - self.game.hoist_length - self.game.relative_motion_h[
			self.game.i] # distance
		self.state[1] = self.game.hoist_length # hoist length
		self.state[2] = self.game.cur_speed # current speed
		self.state[3:] = -self.game.relative_motion_h[self.game.i + 1: self.game.i + 1 + self.pred_steps]

		return np.array(self.state)

	def get_reward(self):
		reward = 0

		if len(self.game.load_dist_log)>2:

			if self.is_terminal() and self.game.i * self.game.dt < self.game.s_timeout:
				impact_velocity = (self.game.load_dist_log[-2] - self.game.load_dist_log[-1]) / self.game.dt
				print(impact_velocity)
				if 0 < impact_velocity < 0.2:
					reward += min(200, (100*(0.2-impact_velocity))**2)
				else:
					reward += -100*(impact_velocity - 0.2)
		return reward

	def is_terminal(self):
		is_done = False
		distance = self.game.distance_crane_to_barge - self.game.hoist_length - self.game.relative_motion_h[
			self.game.i]
		if distance < 0 or self.game.i * self.game.dt > self.game.s_timeout:
			is_done = True
		return is_done

	def step(self, action):

		self.game.timestep(action+1)
		reward = self.get_reward()
		done = self.is_terminal()
		self.state[0] = self.game.distance_crane_to_barge - self.game.hoist_length - self.game.relative_motion_h[self.game.i]
		self.state[1] = self.game.hoist_length
		self.state[2] = self.game.cur_speed
		self.state[3:] = -self.game.relative_motion_h[self.game.i + 1: self.game.i + 1 + self.pred_steps]

		return np.reshape(self.state, [self.state.shape[0], ]), reward, done, {}


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]



