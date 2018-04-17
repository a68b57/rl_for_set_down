import numpy as np
from RL import rlagent
import gym
from RL import testing
import spec_tools.spec_tools as st
import matplotlib.pyplot as plt

num_episodes = 50000

Hs = 1.5
Tp = 20

testing_Hs = 3
testing_Tp = 8

model_dir = '/home/michael/Desktop/workspace/rl_for_set_down/RL/model/'
log_dir = '/home/michael/Desktop/workspace/rl_for_set_down/RL/log/'

exp_name = 'agent_exp_21'

testing_episodes = 10
testing_freq = 50
model_update_freq = 5000
save_reward_log_freq = 200

change_resp_freq = 50000
change_wave_freq = 50000

if __name__ == "__main__":

	# env = gym.make('SetDown-v0')
	env = gym.make('Following-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = rlagent.RLAgent(state_size, action_size, testing=False)
	resp = st.Spectrum.from_synthetic(spreading=None, Hs=Hs, Tp=Tp)
	np.random.seed(11)
	env.rel_motion_sc_t, env.rel_motion_sc = resp.make_time_trace(env.num_step, env.dt)
	env.rel_motion_sc = env.load_file(log_dir + 'training_motion_exp18.csv')

	# env.save_file(log_dir + 'training_motion_exp21.csv', env.rel_motion_sc)
	tol_reward_log = []

	for e in range(num_episodes):
		states = []
		q_log = []

		# if e > 1000:
		# 	if agent.epsilon > agent.epsilon_min:
		# 		agent.epsilon *= 0.9995
		#
		# 	if agent.gamma < agent.gamma_max:
		# 		agent.gamma = 1-0.9995*(1-agent.gamma)
		#
		# 	agent.learning_rate *= 0.9995

		if e % change_resp_freq == 0 and e:
			hs = np.random.uniform(low=1, high=5)
			tp = np.random.uniform(low=5, high=20)
			resp = st.Spectrum.from_synthetic(spreading=None, Hs=hs, Tp=tp)

		if e % change_wave_freq == 0 and e:
			state = env.restart_episode(resp, change_wave=True)
		else:
			state = env.restart_episode(resp, change_wave=False)

		hoist_len = env.init_hoist_len
		for step in range(env.num_step):

			state = np.copy(state)
			action, q = env.get_act(agent)

			q_log.append(q)

			states.append(list([env.init_h_s_ct - hoist_len - state[0][env.initial_waiting_steps],
			                    hoist_len])) # append history of d_sb and hoist len of the last step
			# hoist_len, next_state, reward, done, imp_vel = env.step(action)
			hoist_len, next_state, reward, done = env.step(action)

			# agent.remember(agent.state_temp, action, reward, next_state, done)
			agent.remember(state, action, reward, np.copy(next_state), done)

			state = next_state

			# if done and step < env.num_step-1:
			# 	states = np.array(states)
			# 	d_sb_log = states[:, -2]
			# 	hoist_len_log = states[:, -1]
			# 	print("episode: {}/{}, d_sb:{}, hoist_len:{}, reward:{}, time: {}s, imp_vel: {} m/s"
			# 	      .format(e+1, num_episodes, round(d_sb_log[-1], 3), round(hoist_len_log[-1], 3), round(env.sum_reward, 3), int(env.t), env.final_imp_vel))
			# 	env.plot(d_sb_log, hoist_len_log, show_ani=False, show_motion=False)
			# 	break
			#
			# elif step == env.num_step - 1:
			# 	states = np.array(states)
			# 	print("episode: {}/{}, d_sb:{}, hoist_len:{}, reward:{}"
			# 	      .format(e+1, num_episodes, round(states[-1][-2],3), round(states[-1][-1],3), round(env.sum_reward,3)))
			# 	d_sb_log = states[:, -2]
			# 	hoist_len_log = states[:, -1]
			# 	if env.sum_reward > 5:
			# 		env.plot(d_sb_log, hoist_len_log, show_ani=False, show_motion=True)

			if done:
				states = np.array(states)
				d_sb_log = states[:, -2]
				hoist_len_log = states[:, -1]
				print("episode: {}/{}, d_sb:{}, hoist_len:{}, alpha:{}, gamma:{}, e:{}, reward:{}, mean_q:{}, time: {}s"
				      .format(e + 1, num_episodes,
				              round(d_sb_log[-1], 3),
				              round(hoist_len_log[-1], 3),
				              round(agent.learning_rate, 5),
				              round(agent.gamma, 3),
				              round(agent.epsilon, 3),
				              round(env.sum_reward, 3),
				              np.round(np.mean(q_log),3),
				              int(env.t)))
				# if env.sum_reward > 300:
				# env.plot(d_sb_log, hoist_len_log, show_ani=False, show_motion=True)
				break

			# elif step == env.num_step - 1:
			# 	states = np.array(states)
			# 	d_sb_log = states[:, -2]
			# 	hoist_len_log = states[:, -1]
			# 	print("episode: {}/{}, d_sb:{}, hoist_len:{}, alpha:{}, gamma:{}, e:{}, reward:{}, mean_q:{}"
			# 	      .format(e + 1, num_episodes,
			# 	              round(d_sb_log[-1], 3),
			# 	              round(hoist_len_log[-1], 3),
			# 	              round(agent.learning_rate, 5),
			# 	              round(agent.gamma, 3),
			# 	              round(agent.epsilon, 3),
			# 	              round(env.sum_reward, 3),
			# 	              np.round(np.mean(q_log),3),
			# 	              ))
			# env.plot(d_sb_log, hoist_len_log, show_ani=False, show_motion=True)

		tol_reward_log.append([env.sum_reward, agent.learning_rate, agent.gamma, agent.epsilon, np.mean(q_log), int(env.t)])

		if e > 1000:
			agent.epsilon = 0.2
			agent.replay(32)
		if e % 100 == 0 and e:
			agent.target_train()

		if e % model_update_freq == 0 and e:
			agent.saveModel(model_dir, exp_name)

		# if e % testing_freq == 0 and e:
		# 	t = testing.Testing(testing_episodes, env, agent, log_dir + exp_name + '_testing.csv', e+1)
		# 	t.run_testing(hs=testing_Hs, tp=testing_Tp)

		if e % save_reward_log_freq == 0 and e:
			env.save_file(log_dir + exp_name + '_reward.csv', tol_reward_log)

		#TODO: log for training: end d_sb wrt episode; reward wrt episode; 3D gaussian dist time & impact_vel, testing hist
