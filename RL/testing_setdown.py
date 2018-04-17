import numpy as np
from RL import rlagent
import gym
import spec_tools.spec_tools as st

num_episodes = 1000
Hs = 1.5
Tp = 15
high_pass = 0.3
low_pass = 0.42
model_dir = '/home/michael/Desktop/workspace/rl_for_set_down/RL/model/'

resp = st.Spectrum.from_synthetic(spreading=None, Hs=Hs, Tp=Tp)

if __name__ == "__main__":

	env = gym.make('SetDown-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = rlagent.RLAgent(state_size, action_size, testing=True)
	agent.loadModel(model_dir, 'agent_exp8')
	env.rel_motion_sc_t, env.rel_motion_sc = resp.make_time_trace(env.num_step, env.dt) #

	for e in range(num_episodes):
		states = []
		tol_reward = 0
		state = env.restart_episode(resp, change_wave=True)

		for time_t in range(env.num_step):

			action = env.get_act(agent)
			states.append(list([state[0,env.initial_waiting_steps-1], state[0,-1]])) # append history of d_sb and hoist len of the last step
			next_state, reward, done, imp_vel = env.step(action)
			state = next_state
			tol_reward = tol_reward + reward

			if done and time_t < env.num_step - 1:
				print("set-down, episode: {}/{}, time: {}s, imp_vel: {} m/s, reward: {}"
				      .format(e, num_episodes, int(env.t), env.final_imp_vel, tol_reward))
				states = np.array(states)
				d_sb_log = states[:, -2]
				hoist_len_log = states[:, -1]
				if env.final_imp_vel < 0.1:
					env.plot(d_sb_log, hoist_len_log, show_ani=True,show_motion=True)
				break
