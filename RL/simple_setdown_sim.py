# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:21:58 2018

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# one single simulation

# initial condition


dt = 0.2 # time-step
lunchtime = 1000

lowering_speed = 3 / 60 # m/s
raising_speed = 3 / 60  # m/s

h = 10 # distance between barge and crane-top

hoist_length = 3 # initial hoist length

# relative motion between crane-hook and barge

relative_motion_t = np.arange(0, 1000, dt)

dir = 'dataSet/d0.2_f0.7-0.9/X_o5000_p50_d0.2_f0.7-0.9_1.mat'
wave_dic = loadmat(dir)

# plt.plot(relative_motion_t,relative_motion_h)


def get_action(hdist_log, actual_dist):

	if hdist_log[-1] < hdist_log[-2]:
		decreasing = True
	else:
		decreasing = False

	increasing = not decreasing

	# if we are sill more than 5 meters above the deck, then lower
	if actual_dist > 5:
		return 1

	# if we are close the the deck, and the distance is increasing, then lower
	if increasing:
		return 1

	# if we are close to the deck and the distane is decreasing, then raise
	if decreasing and actual_dist < 1:
		return 3

	# else hold

	return 2



result = []
for i in range(1):

	wave = np.transpose(wave_dic['X'])
	length = len(wave)
	idx = np.random.randint(length)

	# relative_motion_h = np.sin(2 * relative_motion_t * 2* np.pi / 100) + 1 * np.sin(relative_motion_t * 2* np.pi / 70)
	relative_motion_h = wave[i]

	# Start the simulation

	log_r = []
	log_h = []

	hdist_log = []
	rdist_log = []

	t = 0 # time in seconds

	while True:

		# get the relative distance between hook and barge at this moment in time

		hdist = h - np.interp(t, relative_motion_t, relative_motion_h)
		hdist_log.append(hdist)

		rdist = hdist - hoist_length # distance between bottom of cargo and barge
		rdist_log.append(rdist)

		if rdist < 0:
			# we have impact, stop
			impact_velocity = (rdist_log[-2] - rdist_log[-1]) / dt
			break

		# determine action

		# action = 1 : lower
		# action = 2 : hold
		# action = 3 : raise

		# first look the cat our of the tree
		if t < 10:
			action = 2 # hold
		else:
			action = get_action(hdist_log[-10:], rdist)

		# and take action

		if action == 1:
			hoist_length = hoist_length + lowering_speed * dt
		elif action == 2:
			pass
		else:
			hoist_length = hoist_length - raising_speed * dt

		t = t + dt

		log_r.append(rdist)
		log_h.append(hoist_length)

		if t > lunchtime:
			break

	print(impact_velocity)
	result.append(impact_velocity)


result = np.array(result)
# plt.hist(result, bins='auto')

# plt.title('Histogram of Impact velocity: 100 random cases')
# plt.xlabel('Impact velocity')
# plt.ylabel('Probability')
# plt.show()


# make the time-axis
x = np.arange(0,t,dt)


plt.plot(x,log_r)
plt.plot(x,log_h)
plt.title('Impact velocity: %.3f' % impact_velocity)
plt.ylabel('distance(m)')
plt.xlabel('time(s)')
plt.legend(['dis_barge_cargo','hoist_length'])
plt.show()
