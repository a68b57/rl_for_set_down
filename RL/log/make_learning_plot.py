import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

filename = '/exp22/following_2.csv'


data_log = np.loadtxt(filename, delimiter=',')

no_replay = 1000
average_window = 500
r = data_log[no_replay:, 0]
alpha = data_log[no_replay:, 1]
gamma = data_log[no_replay:, 2]
eps = data_log[no_replay:, 3]
q = data_log[no_replay:, 4]
# time = data_log[no_replay:, 5]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

r = running_mean(r, average_window)
alpha = alpha[0:r.shape[0]]
gamma = gamma[0:r.shape[0]]
eps = eps[0:r.shape[0]]
# q = q[0:r.shape[0]]
q = running_mean(q, average_window)
# time = running_mean(time,average_window)
t = np.arange(0, r.shape[0], 1)

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()
# par3 = host.twinx()

# learning rate
new_fixed_axis_1 = par1.get_grid_helper().new_fixed_axis
par1.axis["left"] = new_fixed_axis_1(loc="right",
                                    axes=par1,
                                    offset=(0, 0))

par1.axis["left"].toggle(all=True)



# # gamma
# new_fixed_axis_2 = par2.get_grid_helper().new_fixed_axis
# par2.axis["left"] = new_fixed_axis_2(loc="right",
#                                     axes=par2,
#                                     offset=(80, 0))
# #
# par2.axis["left"].toggle(all=True)
#
#
#
# # eps
# new_fixed_axis_3 = par3.get_grid_helper().new_fixed_axis
# par3.axis["right"] = new_fixed_axis_3(loc="left",
#                                     axes=par3,
#                                     offset=(-60, 0))
#
# par3.axis["right"].toggle(all=True)

host.set_ylim(-100, 300)

host.set_xlabel("episode")
host.set_ylabel("reward")
par1.set_ylabel("Q")
# par2.set_ylabel("survive time")
# par1.set_ylabel("learning rate")
# par2.set_ylabel("discount factor")
# par3.set_ylabel("exploration")


p1, = host.plot(t, r, label="reward")
p2, = par1.plot(t, q, label="Q")
# p3, = par2.plot(t,time,label="time")
# p2, = par1.plot(t, alpha, label="learning rate")
# p3, = par2.plot(t, gamma, label="discount factor")
# p4, = par3.plot(t, eps, label="exploration")

par1.set_ylim(0, 30)

par2.set_ylim(0, 200)

# par1.set_ylim(0, 0.001)
# par2.set_ylim(0, 1.5)
# par3.set_ylim(0, 1.5)

host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["left"].label.set_color(p2.get_color())
# par2.axis["left"].label.set_color(p3.get_color())
# par3.axis["right"].label.set_color(p4.get_color())

plt.draw()
plt.show()
