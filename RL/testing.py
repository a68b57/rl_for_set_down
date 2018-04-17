import spec_tools.spec_tools as st
import numpy as np
import os


class Testing():

	def __init__(self, num_epi, env, agent, log_file, training_epi):
		self.agent = agent
		self.env = env
		self.episodes = num_epi
		self.log_file = log_file
		if not os.path.isfile(self.log_file):
			open(self.log_file, 'w').close()
		self.training_epi = training_epi

	def run_testing(self, hs, tp):
		logs = np.loadtxt(open(self.log_file, "r"), delimiter=",")
		resp = st.Spectrum.from_synthetic(spreading=None, Hs=hs, Tp=tp)
		for e in range(self.episodes):
			print('{}/{}'.format(e+1, self.episodes))
			states = []
			tol_reward = 0
			state = self.env.restart_episode(resp, testing=True)

			for time_t in range(self.env.num_step):
				action = self.env.get_act(self.agent)
				states.append(list(state[-2:])) # append history of d_sb and hoist len of the last step
				next_state, reward, done, imp_vel = self.env.step(action)
				state = next_state
				tol_reward = tol_reward + reward

				# if done and time_t < self.env.num_step - 1:
				# 	print("set-down, episode: {}/{}, time: {}s, imp_vel: {} m/s, reward: {}"
				# 	      .format(e+1, self.episodes, int(self.env.t), self.env.final_imp_vel, tol_reward))
				# 	states = np.array(states)
				# 	d_sb_log = states[:, -2]
				# 	hoist_len_log = states[:, -1]
				# 	if self.env.final_imp_vel < 0.1:
				# 		self.env.plot(d_sb_log, hoist_len_log, show_ani=False, show_motion=False)
				# 	break

			if self.env.t > self.env.timeout:
				set_down = False
			log = np.array([self.training_epi, round(tol_reward, 3), self.env.timeout, imp_vel, int(set_down)])

			if logs.size:
				logs = np.vstack((logs, log))
			else:
				logs = log
		np.savetxt(self.log_file, logs, delimiter=',')


