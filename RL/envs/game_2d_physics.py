import pymunk
from pymunk.vec2d import Vec2d
import numpy as np
import spec_tools.spec_tools as st

# Center of the scene is at x=0


from os.path import abspath, dirname


class game2d(object):

	def __init__(self):

		# game end
		self.time_in_position_till_end = 5  # seconds


		self.sway_speed = 0.5  # m/s
		self.hoist_speed = 0.5  # m/s
		self.n_inner = 100  # this many time-steps per frame

		self.wave_spectrum = []

		# time-traces
		self.dt_motions = 0.1
		self.t_motions = 1000

		self.friction = 100

		# geometry
		self.barge_upper_left = Vec2d(-302 * 0.3 / 2, -4 )
		self.barge_upper_right = Vec2d(302 * 0.3 / 2, -4 )
		self.barge_lower_left = Vec2d(-302 * 0.3 / 2, 2 )
		self.barge_lower_right = Vec2d(302 * 0.3 / 2, 2 )

		self.load_upper_left = Vec2d(-4, -4)
		self.load_upper_right = Vec2d(4, -4)
		self.load_lower_left = Vec2d(-4, 4)
		self.load_lower_right = Vec2d(4, 4)

		self.poi = Vec2d(0, -13)

		self.EA = 50000

		self.water_level = 90

		self.magic_pitch_factor = 1 # for better looking physics

		# initial game settings
		self.hoist_length = 60
		self.crane_sway = 0
		self.barge_impulse = []
		self.bumper_impulse = []
		self.has_barge_contact = False
		self.has_bumper_contact = False
		self.setdown_counter = 0
		self.is_done = False

		self.time_lookup_index = 0
		self.max_impact = 0


	## Collision handler
	def barge_contact(self, arbiter, space, data):
		self.barge_impulse.append(arbiter._get_total_impulse())
		if np.max(np.abs(arbiter._get_total_impulse())) > self.max_impact:
			self.max_impact = np.max(np.abs(arbiter._get_total_impulse()))
		self.has_barge_contact = True


	def bumper_contact(self, arbiter, space, data):
		self.bumper_impulse.append(arbiter._get_total_impulse())
		if np.max(np.abs(arbiter._get_total_impulse())) > self.max_impact:
			self.max_impact = np.max(np.abs(arbiter._get_total_impulse()))
		self.has_bumper_contact = True

	def prep_new_run(self):
		""""resets the engine to a new game"""

		# Make world
		self.space = pymunk.Space()
		# self.load = pymunk.Body(1000, 1000 * 30 *30)
		self.load = pymunk.Body(1000, 10000 * 30 * 30)

		self.barge = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
		self.hook = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
		self.space.add(self.load)
		self.space.add(self.barge)

		## Contact shapes

		# give contact shapes a thickness for better stability

		th = 2
		barge_deck = pymunk.Segment(self.barge, [self.barge_upper_left[0], self.barge_upper_left[1] + th],
		                            [self.barge_upper_right[0], self.barge_upper_right[1] + th], th)
		barge_deck.collision_type = 1
		barge_deck.friction = self.friction
		self.space.add(barge_deck)

		# Load contact shape
		radius = 0.1
		shape1 = pymunk.Circle(self.load, radius, self.load_lower_left)
		shape2 = pymunk.Circle(self.load, radius, self.load_lower_right)
		shape1.collision_type = 2
		shape2.collision_type = 2
		shape1.friction = self.friction
		shape2.friction = self.friction
		self.space.add(shape1)
		self.space.add(shape2)

		# Load contact shape bottom
		load_bottom_shape = pymunk.Segment(self.load, self.load_lower_left, self.load_lower_right, 0)
		load_bottom_shape.collision_type = 2
		load_bottom_shape.friction = self.friction
		self.space.add(load_bottom_shape)

		# Guide contact shape
		self.bumper_lower = Vec2d(10, -4)
		self.bumper_upper = Vec2d(10, -10)
		bumper = pymunk.Segment(self.barge, [self.bumper_lower[0] + 2, self.bumper_lower[1]],
		                        [self.bumper_upper[0] + 2, self.bumper_upper[1]], 2)
		bumper.collision_type = 3
		self.space.add(bumper)

		# spring-damper between hook and load
		damper = pymunk.DampedSpring(self.hook, self.load, (0, 0), self.poi, 0, 0, 5000)
		adamper = pymunk.DampedRotarySpring(self.hook, self.load, 0, 0, 400)
		self.space.add(damper, adamper)

		handler_barge_contact = self.space.add_collision_handler(1, 2)  # barge  is type 1, load is type 2
		handler_barge_contact.post_solve = self.barge_contact

		handler_bumper_contact = self.space.add_collision_handler(2, 3)  # bumper is type 3, load is type 2
		handler_bumper_contact.post_solve = self.bumper_contact

		self.space.gravity = Vec2d(0, 9.81)
		self.space.damping = 0.98


		n = int(self.t_motions / self.dt_motions)

		temp = self.wave_spectrum.make_time_trace(associated=self.associated, n=n, dt=self.dt_motions,
												  locations=self.wave_location)
		self.wave_elevation = temp['response']
		self.motions_t = temp['t']
		R = temp['associated']


		self.motions_sway_block = R[0]
		self.motions_heave_block = R[1]
		self.motions_302_surge = R[2]
		self.motions_302_heave = R[3]
		self.motions_302_pitch = R[4] * self.magic_pitch_factor


		# # TODO: Temp start with stationary env
		# self.motions_sway_block = np.zeros((10000,))
		# self.motions_heave_block = np.zeros((10000,))
		# self.motions_302_surge = np.zeros((10000,))
		# self.motions_302_heave = np.zeros((10000,))
		# self.motions_302_pitch = np.zeros((10000,))


		self.hoist_length = np.random.uniform(45, 55)
		self.crane_sway = 0
		self.barge_impulse = []
		self.bumper_impulse = []
		self.has_barge_contact = False
		self.setdown_counter = 0

		self.is_done = False

		initial_x = np.random.uniform(10, 15) * np.random.choice([-1, 1])
		self.load.position = Vec2d(initial_x, self.hoist_length - self.poi[1])
		self.time_lookup_index = 0

		self.max_impact = 0

	def setup(self):
		current_directory = abspath(__file__)
		pkg_dir = dirname(dirname(current_directory))

		block_sway_motion = st.Rao.from_liftdyn(
			pkg_dir + '/RAO/model_crane_susp.stf_plt', iMode=2)
		block_heave_motion = st.Rao.from_liftdyn(
			pkg_dir + '/RAO/model_crane_susp.stf_plt', iMode=3)

		barge_surge_motion = st.Rao.from_liftdyn(
			pkg_dir + '/RAO/model_h302_cog.stf_plt', iMode=1)
		barge_heave_motion = st.Rao.from_liftdyn(
			pkg_dir + '/RAO/model_h302_cog.stf_plt', iMode=3)
		barge_pitch_motion = st.Rao.from_liftdyn(
			pkg_dir + '/RAO/model_h302_cog.stf_plt', iMode=5)

		self.associated = [block_sway_motion, block_heave_motion, barge_surge_motion, barge_heave_motion, barge_pitch_motion]


		self.wave_location = np.arange(-100, 100, 10)



	def get_external_movements(self, t):

		i = self.time_lookup_index

		if t > self.motions_t[i]:
			i += 1

		i = min(i, len(self.motions_t) - 2)

		self.time_lookup_index = i

		t_last = self.motions_t[i]
		t_next = self.motions_t[i + 1]

		dt = t_next - t_last

		fac_next = (t - t_last) / dt
		fac_last = (t_next - t) / dt

		crane_x = fac_last * self.motions_sway_block[i] + fac_next * self.motions_sway_block[i + 1]
		crane_y = fac_last * self.motions_heave_block[i] + fac_next * self.motions_heave_block[i + 1]
		barge_x = fac_last * self.motions_302_surge[i] + fac_next * self.motions_302_surge[i + 1]
		barge_y = fac_last * self.motions_302_heave[i] + fac_next * self.motions_302_heave[i + 1] + self.water_level
		barge_rotation = fac_last * self.motions_302_pitch[i] + fac_next * self.motions_302_pitch[i + 1]

		return (crane_x, crane_y, barge_x, barge_y, barge_rotation)


	def step(self, t_simulation, dt, action):

		dt_physics = dt / self.n_inner

		for i in range(self.n_inner):

			if action == 'right':
				self.crane_sway += self.sway_speed * dt_physics
			elif action == 'left':
				self.crane_sway -= self.sway_speed * dt_physics
			elif action == 'down':
				self.hoist_length += self.hoist_speed * dt_physics
			elif action == 'up':
				self.hoist_length -= self.hoist_speed * dt_physics
			elif action == 'hold':
				pass
			else:
				raise ValueError('Unknown action')

			t = t_simulation + i * dt_physics

			(crane_x, crane_y, barge_x, barge_y, barge_rotation) = self.get_external_movements(t)
			# crane_x = np.interp(t, self.motions_t, self.motions_sway_block)
			# crane_y = np.interp(t, self.motions_t, self.motions_heave_block)

			self.hook.position = Vec2d(crane_x + self.crane_sway, crane_y)  # motions are already scaled

			# apply barge motions
			# barge_x = np.interp(t, self.motions_t, self.motions_302_surge)
			# barge_y = np.interp(t, self.motions_t, self.motions_302_heave) + self.water_level
			# barge_rotation = np.interp(t, self.motions_t, self.motions_302_pitch)

			self.barge.position = (barge_x, barge_y)
			self.barge.angle = np.deg2rad(barge_rotation)

			# cable runs between
			# hoist_position
			# and
			#
			c = self.hook.local_to_world((0, 0)) - self.load.local_to_world(self.poi)

			actual_length = c.length
			dir = c.normalized()

			if actual_length <= self.hoist_length:
				t = 0
			else:
				t = self.EA * actual_length / self.hoist_length

			force = t * dir

			global_poi_position = self.load.local_to_world(self.poi)
			self.load.apply_force_at_world_point(force, global_poi_position)


			# Time-stepping and damping
			self.has_barge_contact = False
			self.has_bumper_contact = False

			self.space.step(dt_physics)


			# for the game to end we need a minimum period of continuous contact
			if self.has_barge_contact:
				self.setdown_counter += 1
				if self.setdown_counter > ((self.time_in_position_till_end / dt) * self.n_inner):
					# print('Ready')
					self.is_done = True
					# break

			else:
				self.setdown_counter -= 1
				self.setdown_counter = max(self.setdown_counter,0) # never less and 0

			# print(self.setdown_counter)



