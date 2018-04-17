import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


# sns.set(color_codes=True)
sns.set(style="darkgrid")


#### for overlapped histo#########
# sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
#
# dt_23_7_1 = pd.read_json('following_23.7+1000000_log.csv')
# dt_23_7_2 = pd.read_json('following_23.7+1000000+1000000_log.csv')
# dt_23_7_3 = pd.read_json('following_23.7+1000000+1000000+1000000_log.csv')
#
# d1 = np.array(dt_23_7_1['mean_q'])
# d2 = np.array(dt_23_7_2['mean_q'])
# d3 = np.array(dt_23_7_3['mean_q'])
#
# print(d1)
# print(d2)
# print(d3)

# test_all_data = np.concatenate((d1, d2, d3, d4, d5, d6, d7))
# test_group = np.concatenate((10*np.ones(len(d1)), 30*np.ones(len(d2)), 50*np.ones(len(d3)), 70*np.ones(len(d4)), 90*np.ones(len(d5)),110*np.ones(len(d6)), 130*np.ones(len(d7))))
# df = pd.DataFrame(dict(g=test_group.astype(np.int), impact_velocity=test_all_data))
#
# pal = sns.cubehelix_palette(10, rot=-.25, light=.4)
# g = sns.FacetGrid(df, row="g", hue="g", aspect=2.5, size=1, palette=pal)
#
# g.map(sns.kdeplot, "impact_velocity", clip_on=False, shade=True, alpha=0.5, lw=1.5, bw=.2)
# g.map(sns.kdeplot, "impact_velocity", clip_on=False, color="w", lw=2, bw=.2)
# g.map(plt.axhline, y=0, lw=2, clip_on=False)
#
#
# def label(x, color, label):
#     ax = plt.gca()
#     ax.text(-.15, 0.1, label+'k steps', fontweight="bold", color=color,
#             ha="left", va="center", transform=ax.transAxes)
#
# g.map(label, "impact_velocity")
# g.fig.subplots_adjust(hspace=-0.4)
#
# g.set_titles("")
# g.set(yticks=[])
# g.despine(bottom=True, left=True)
# plt.xlim(-0.1,1)
# plt.show()
####################

##################### overlapped histros in one figure



#
# sns.kdeplot(d1, label='pred:1s', shade=True)
# sns.kdeplot(d2, label='pred:3s', shade=True)
# sns.kdeplot(d3, label='pred:5s', shade=True)
#
#
# # sns.kdeplot(d3, label='pred: 5s', shade=True)
#
# # label = 'mean:{:.4f}, std:{:.4f}'.format(np.mean(d1), np.std(d1))
# # sns.kdeplot(d1, label=label, shade=True)
#
# # plt.xlim(-0.1,0.6)
# plt.xlabel('survival time')
# plt.title('Hs:1.5, Tp:20, speed:3/60, limit:1.5m')
#
# plt.show()
###################################################

######show tsplot###########
window = 300


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


cur1 = pd.read_json('following_23.7_log.csv')['mean_q']
cur2 = pd.read_json('following_23.8_log.csv')['mean_q']

cur1 = running_mean(np.array(cur1), window)
std_cur1 = np.std(cur1)

cur2 = running_mean(np.array(cur2), window)
std_cur2 = np.std(cur2)

l = min(len(cur1),len(cur2))
C_params = np.linspace(1, l, l)
cur1 = cur1[0:l]
cur2 = cur2[0:l]

cur1_min = -50
cur1_max = 50

cur2_min = -50
cur2_max = 50

sns.set_style("darkgrid")


host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()

new_fixed_axis_1 = par1.get_grid_helper().new_fixed_axis
par1.axis["left"] = new_fixed_axis_1(loc="right",
                                    axes=par1,
                                    offset=(0, 0))

par1.axis["left"].toggle(all=True)


host.set_ylim(cur1_min, cur1_max)
host.set_xlabel("episode")
host.set_ylabel("mean Q-value")
# par1.set_ylabel("mean Q-value")

p1, = host.plot(C_params, cur1,label="23.7", color='red')
p2, = par1.plot(C_params, cur2, label="23.8", color='green')
par1.fill_between(C_params, cur1 - std_cur1, cur1 + std_cur1, alpha = 0.1, color="red")
host.fill_between(C_params, cur2 - std_cur2, cur2 + std_cur2, alpha = 0.1, color="green")
par1.set_ylim(cur2_min, cur2_max)
host.legend()

# host.axis["left"].label.set_color(p1.get_color())
# par1.axis["left"].label.set_color(p2.get_color())

# host.axis["left"].major_ticklabels.set_color(p1.get_color())
# par1.axis["left"].major_ticklabels.set_color(p2.get_color())

# plt.title('Hs:1.5, Tp:15, pred:5s, speed:3/60')
plt.draw()
plt.show()
#





