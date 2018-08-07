import numpy as np
import matplotlib.pyplot as plt

class SimpleSetDownGame(object):
    def __init__(self):

        # response spectrum
        self.spectrum = None

        # time-step in seconds
        self.dt = 0.25  # time-step
        self.s_timeout = 1000  # maximum number of steps in a simulation

        self.n_future_wave_samples = 200 # make the wave until xx samples past time-out

        self.lowering_speed = 3 / 60  # m/s
        self.raising_speed = 3 / 60  # m/s

        self.distance_crane_to_barge = 10  # distance between barge and crane-top
        self.initial_hoist_length = 3  # initial hoist length
        self.n_action_hold = 1  # hold every action for xxx time-steps. After that ask for the next

        self.action = 2 # initialize action to "hold" - will be changed in "timestep"

        self.cur_speed = 0

        self.max_speed = 9 / 60  # 100% RPM full speed down
        self.half_speed = self.max_speed // 2  # 50% RPM

        self.ramp_up = 3  # freezing time when action 1 and 2 are selected, in seconds
        self.ramp_up_step = max(1, int(self.ramp_up / self.dt))  # corresponding holding steps
        self.accel = (4.5 / 60) / self.ramp_up  # takes three second to increase speed to 50%; unit m/s2

    def reset_game(self):
        """Resets game, but keeps waves the same"""

        # empty log
        self.load_dist_log = []
        self.hoist_length_log = []
        self.motion_log = []
        self.action_log = []

        self.i = 0
        self.cur_speed = 0
        self.i_action = self.n_action_hold + 1  # enforce naksing for new action
        self.hoist_length = self.initial_hoist_length

    def new_game(self):
        """Prepares a new game with new random waves, 
        automatically calls "reset_game" as well
        """

        self.n_timeout = int(self.s_timeout / self.dt)  # number of timesteps in the calculation

        # initialize random motion-train based on spectrum
        temp = \
            self.spectrum.make_time_trace(self.n_timeout + self.n_future_wave_samples, self.dt)

        self.relative_motion_h = temp['response'][0]

        self.reset_game()


    def timestep(self, action_agent):
        """
        Marches the simulation one step forwards in time.
        
        Requests "action_agent" for a new action if required.
        
        
        :param action_agent:
         action_agent should match the following signature:
         action_agent(motion, wirelength, relative), this function shall return 1,2 or 3
        
        :return: Positive: end of simulation -> impact-velocity
                 negative: end of simulation -> load distance
                 None: simulation has not yet ended
                 
        """

        # update book-keeping
        t = self.i * self.dt
        wave_elevation = self.relative_motion_h[self.i]
        load_distance = self.distance_crane_to_barge - self.hoist_length - wave_elevation

        # update the log
        self.load_dist_log.append(load_distance)
        self.motion_log.append(wave_elevation)
        self.hoist_length_log.append(self.hoist_length)
        self.action_log.append(self.action)

        if np.mod(self.i, self.n_action_hold) == 0:
            self.action = action_agent

        # action=0:reduce; action=1:no action; action=2:gas
        # action-1 == [-1, 0, 1]
        action = self.action - 1

        # 0 = down
        # 1 = hold
        # 2 = up

        if self.cur_speed == self.max_speed:
            self.cur_speed = min(self.cur_speed + (action - 1) * self.accel * self.dt, self.max_speed)
        else:
            self.cur_speed = max(self.cur_speed + (action - 1) * self.accel * self.dt, -self.max_speed)

        self.cur_speed = min(self.cur_speed, self.max_speed)
        self.hoist_length = max(self.hoist_length + self.cur_speed * self.dt, 0)

        self.i = self.i + 1

        return None

    def plot_game(self):
        """
        Plots the last played game, or the current game status, in the current matplotlib window
        :return: None
        """

        t = self.dt * np.arange(0, len(self.motion_log))
        plt.plot(t, self.motion_log)

        # t = self.dt * np.arange(0, self.i)
        plt.plot(t, self.distance_crane_to_barge - np.array(self.hoist_length_log))

    def save_results(self):

        r = np.random.randint(100000000)
        filename = r'\\ALECTO\techylei\Dep\MarineEngineering\Software\AI\game_results\results_' + str(r) + '.txt'

        t = self.dt * np.arange(0, len(self.motion_log))
        m = self.motion_log
        d = self.distance_crane_to_barge - np.array(self.hoist_length_log)
        al = self.action_log

        with open(filename,'w') as f:
            for [a,b,c,d] in zip(t,m,d,al):
                f.write('{}, {}, {} {}\n'.format(a,b,c,d))

    def play_round(self, agent):
        """
        Runs a game until game-over. Does not re-initialize the waves.
         To re-init waves with new random phase angles call new_game() before
         calling this method.
        
        :param agent: see timestep
        :return: see timestep
        """
        self.reset_game()

        result = None
        while result is None:
            result = self.timestep(agent)

        return result

    def tournament(self, agent, number_of_times):
        """
        Play the game a number_of_times times using agent
        Plots the results as a histogram in the currently active figure.
        
        New waves are used for every round.
        
        :return: impact_velocities , number of failed games (time-out) 
        """
        results = []
        failed = 0

        for i in range(number_of_times):

            self.new_game()

            result = self.play_round(agent)

            if result > 0:
                results.append(result)
            else:
                failed += 1

                # self.plot_game()

        # plt.figure()
        plt.hist(results)

        return results, failed
