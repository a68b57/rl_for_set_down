import numpy as np
from RL import rlagent
import gym
import spec_tools.spec_tools as st

num_episodes = 1000
Hs = 0.5
Tp = 10
model_dir = '/home/michael/Desktop/workspace/rl_for_set_down/RL/model/'

change_resp_freq = 50000
change_wave_freq = 1
resp = st.Spectrum.from_synthetic(spreading=None, Hs=Hs, Tp=Tp)

if __name__ == "__main__":

	env = gym.make('Following-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = rlagent.RLAgent(state_size, action_size, testing=True)
	agent.loadModel(model_dir, 'agent_exp_20')

	resp = st.Spectrum.from_synthetic(spreading=None, Hs=Hs, Tp=Tp)
	np.random.seed(11)
	env.rel_motion_sc_t, env.rel_motion_sc = resp.make_time_trace(env.num_step, env.dt)
	tol_reward_log = []

	for e in range(num_episodes):
		states = []
		q_log = []

		if e % change_resp_freq == 0 and e:
			hs = np.random.uniform(low=1, high=5)
			tp = np.random.uniform(low=5, high=20)
			resp = st.Spectrum.from_synthetic(spreading=None, Hs=hs, Tp=tp)

		if e % change_wave_freq == 0:
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

			hoist_len, next_state, reward, done = env.step(action)
			agent.remember(state, action, reward, np.copy(next_state), done)
			state = next_state

			if done and step < env.num_step - 1:
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
				              np.round(np.mean(q_log), 3),
				              int(env.t)))
				if env.sum_reward > 300:
					env.plot(d_sb_log, hoist_len_log, show_ani=False, show_motion=True)
				break

			elif step == env.num_step - 1:
				states = np.array(states)
				d_sb_log = states[:, -2]
				hoist_len_log = states[:, -1]
				print("episode: {}/{}, d_sb:{}, hoist_len:{}, alpha:{}, gamma:{}, e:{}, reward:{}, mean_q:{}"
				      .format(e + 1, num_episodes,
				              round(d_sb_log[-1], 3),
				              round(hoist_len_log[-1], 3),
				              round(agent.learning_rate, 5),
				              round(agent.gamma, 3),
				              round(agent.epsilon, 3),
				              round(env.sum_reward, 3),
				              np.round(np.mean(q_log), 3),
				              ))
				env.plot(d_sb_log, hoist_len_log, show_ani=False, show_motion=True)
