from gym import spaces, Env
from gym.utils import seeding
import spec_tools.spec_tools as st
from .game_2d_physics import game2d
import numpy as np
import pygame
from copy import deepcopy

class SetDown_2d_gym(Env):
	def __init__(self):
		self.wave_spectrum = st.Spectrum.from_synthetic(Hs=2, coming_from=0, omega=np.arange(0, 4, 0.01), Tp=8,
		                                                gamma=2.0,
		                                                spreading_type=None)

		self.fps = 60   # this many frames per second for drawing
		self.rtf = 5    # time scale
		dt_inner_target = 0.004
		self.n_inner = int(np.ceil(self.rtf / (dt_inner_target * self.fps)))
		self.engine = game2d()
		self.engine.wave_spectrum = self.wave_spectrum
		self.engine.magic_pitch_factor = 0.2 # <1 means lower pitch
		self.engine.n_inner = self.n_inner
		pygame.init()

		self.sim_start = 0
		self.t_simulation = 0

		# state_len = 4 #for exp 25

		state_len = 13

		self.high_limit = np.inf * np.ones(state_len)
		self.action_space = spaces.Discrete(5)
		# self.action_space = spaces.Discrete(3)

		self.observation_space = spaces.Box(-self.high_limit, high=self.high_limit, dtype=np.float16)
		self.state = np.zeros([self.high_limit.shape[0]])
		self.tol_steps = 0
		self.engine.setup()

	# def get_reward(self):
	# 	"""this is for exp25"""
	#
	# 	hook_position = self.state[0]
	# 	theta = self.state[1]
	# 	vel = self.state[3]
	# 	reward = 0
	#
	# 	# if np.abs(theta) < 5 and np.abs(vel) < 2:
	# 	# 	reward = min(1/(np.abs(vel)), 5) # exp 25.1
	# 	#   reward = 1 # exp 25
	#
	# 	if np.abs(theta) < 1 and np.abs(vel) < 1: # exp 25.2
	# 		reward = 1
	#
	# 	if np.abs(hook_position) > 5:
	# 		reward += 5 - np.abs(hook_position)
	#
	# 	if self.engine.hoist_length < 20:
	# 		reward += -1
	#
	# 	return reward

	def get_reward(self):
		"""this is for exp 26"""
		reward = 0

		min_distance = self.engine.min_distance
		height = np.abs(self.state[12] - self.state[6])

		drop = self.state[12] - self.state[8]

		# if 0 < min_distance < 2 and self.state[3] < 1 and self.state[1] < 1 and height < 5:
		# 	reward += self.state[12]/100
			## reward += 1
		# else:
		# 	reward -= 0.1
		# if self.is_terminal():
		# 	print(min_distance, height)
		# 	if self.engine.is_done:
		# 		reward += min(200, 10 / min_distance)

		done = self.is_terminal()

		reward = 0
		if not done:
			if 0 < min_distance < 2 and self.state[3] < 1 and self.state[1] < 1 and drop > 3:
				# reward += self.state[12]/100
				reward += 1

			if self.engine.has_barge_contact:
				reward -= 100
			# 	print('bump contact')
		#
		# elif self.engine.max_impact != 0:
		# 	if 0 < min_distance < 2.5:
		# 		reward += np.power(20000/self.engine.max_impact, 2)

		return reward, done

	# def is_terminal(self):
	# 	"""this is for exp 25"""
	# 	hook_position = self.state[0]
	# 	theta = self.state[1]
	# 	vel = self.state[3]
	# 	is_terminal = False
	# 	# if self.tol_steps == 2000 or np.abs(hook_position[0]) > 50: # 25 and 25.1
	# 	state = np.abs(theta) < 1 and np.abs(vel) < 1
	# 	if self.tol_steps == 1500 or np.abs(hook_position) > 15 or self.engine.hoist_length < 10 or self.engine.is_done:
	# 		is_terminal = True
	# 	return is_terminal

	def is_terminal(self):
		"""this is for exp 26"""
		is_terminal = False
		if self.tol_steps+1 == 1500 or self.engine.is_done or np.abs(self.state[0]) > 20 or self.engine.has_barge_contact:
			is_terminal = True
			# valid = np.abs(self.state[12] - self.state[6]) < 0.2 and np.abs(self.state[10] - self.state[
			# 		6]) < 0.2 and 0 < self.engine.min_distance < 2.5
			# print(self.engine.max_impact, self.engine.min_distance, np.abs(self.state[12] - self.state[6]),
			#       self.tol_steps+1, valid)
		return is_terminal

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):

		self.engine.prep_new_run()
		self.engine.is_done = False

		self.sim_start = self.rtf * pygame.time.get_ticks() / 1000

		self.tol_steps = 0

		self.compute_state()
		self.state = np.reshape(self.state, [self.state.shape[0], ])
		return np.array(self.state)

	def compute_state(self):

		hook_position = self.engine.hook.local_to_world((0, 0))
		poi_position = self.engine.load.local_to_world(self.engine.poi)
		poi_v_x = self.engine.load.velocity_at_world_point(self.engine.poi)[0]
		poi_v_y = self.engine.load.velocity_at_world_point(self.engine.poi)[1]

		bumper_position_lower = self.engine.barge.local_to_world(self.engine.bumper_lower)
		bumper_position_upper = self.engine.barge.local_to_world(self.engine.bumper_upper)

		load_position_lower_right = self.engine.load.local_to_world(self.engine.load_lower_right)
		load_position_lower_left = self.engine.load.local_to_world(self.engine.load_lower_left)

		x = (poi_position[0] - hook_position[0])
		y = -(poi_position[1] - hook_position[1])
		theta = np.rad2deg(np.arctan2(y, x)) + 90

		self.state[0] = hook_position[0]  # hook x
		self.state[1] = theta             # angle cable
		self.state[2] = poi_position[0]   # x at end of cable
		self.state[3] = poi_v_x           # vel x at end of cable

		# followings are new states for exp26
		self.state[4] = poi_v_y           # vel y at end of cable
		self.state[5] = bumper_position_lower[0]      # x bumper lower
		self.state[6] = bumper_position_lower[1]      # y bumper lower
		self.state[7] = bumper_position_upper[0]      # x bumper upper
		self.state[8] = bumper_position_upper[1]      # y bumper upper
		self.state[9] = load_position_lower_left[0]   # x load lower left
		self.state[10] = load_position_lower_left[1]  # y load lower left
		self.state[11] = load_position_lower_right[0] # x load lower right
		self.state[12] = load_position_lower_right[1] # y load lower right

	def step(self, action):

		## this is for exp 25
		# if action == 0:
		# 	key = 'left'
		# elif action == 1:
		# 	key = 'hold'
		# else:
		# 	key = 'right'

		# for exp 26 and 25.3
		# if action == 0:
		# 	key = 'hold'
		# elif action == 1:
		# 	key = 'right'
		# elif action == 2:
		# 	key = 'left'
		# elif action == 3:
		# 	key = 'down'
		# else:
		# 	key = 'up'


		# for dagger on 25.2.2 and exp 26.4
		if action == 0:
			key = 'left'
		elif action == 1:
			key = 'hold'
		elif action == 2:
			key = 'right'
		elif action == 3:
			key = 'down'
		else:
			key = 'up'


		self.t_simulation = self.rtf * pygame.time.get_ticks() / 1000 # convert to seconds
		self.t_simulation -= self.sim_start
		self.engine.step(self.t_simulation, self.rtf / self.fps, key)

		hook_position = self.engine.hook.local_to_world((0, 0))
		poi_position = self.engine.load.local_to_world(self.engine.poi)
		poi_v_x = self.engine.load.velocity_at_world_point(self.engine.poi)[0]
		poi_v_y = self.engine.load.velocity_at_world_point(self.engine.poi)[1]

		bumper_position_lower = self.engine.barge.local_to_world(self.engine.bumper_lower)
		bumper_position_upper = self.engine.barge.local_to_world(self.engine.bumper_upper)

		load_position_lower_right = self.engine.load.local_to_world(self.engine.load_lower_right)
		load_position_lower_left = self.engine.load.local_to_world(self.engine.load_lower_left)

		x = (poi_position[0] - hook_position[0])
		y = -(poi_position[1] - hook_position[1])
		theta = np.rad2deg(np.arctan2(y, x)) + 90

		# reward = self.get_reward(hook_position, theta, poi_v_x) # for exp 25
		# done = self.is_terminal(hook_position, theta, poi_v_x) # for exp 25

		self.state[0] = hook_position[0]  # hook x
		self.state[1] = theta             # angle cable
		self.state[2] = poi_position[0]   # x at end of cable
		self.state[3] = poi_v_x           # vel x at end of cable

		# followings are new states for exp26
		self.state[4] = poi_v_y           # vel y at end of cable
		self.state[5] = bumper_position_lower[0]      # x bumper lower
		self.state[6] = bumper_position_lower[1]      # y bumper lower
		self.state[7] = bumper_position_upper[0]      # x bumper upper
		self.state[8] = bumper_position_upper[1]      # y bumper upper
		self.state[9] = load_position_lower_left[0]   # x load lower left
		self.state[10] = load_position_lower_left[1]  # y load lower left
		self.state[11] = load_position_lower_right[0] # x load lower right
		self.state[12] = load_position_lower_right[1] # y load lower right

		reward, done = self.get_reward()

		self.tol_steps += 1

		return np.reshape(self.state, [self.state.shape[0], ]), reward, done, {}
