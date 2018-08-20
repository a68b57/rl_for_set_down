import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import json
import matplotlib.patches as mpatches


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


# d1 = np.array([])
# d2 = np.array([])
# d3 = np.array([])
# d4 = np.array([])
#
# sns.kdeplot(d1, label='Agent_4.5/60', shade=True)
# sns.kdeplot(d2, label='Agent_9/60', shade=True, color='g')
# sns.kdeplot(d3, label='Agent_2_speeds', shade=True, color='r')
# sns.kdeplot(d4, label='Monkey', shade=True, color='orange')
#
# # label = 'mean:{:.4f}, std:{:.4f}'.format(np.mean(d1), np.std(d1))
# # sns.kdeplot(d1, label=label, shade=True)
#
# plt.xlim(0,1)
# plt.xlabel('impact velocity(m/s)')
# plt.title('Distribution of impact velocity')
# plt.show()
###################################################

########### count plots ############


# data = pd.read_csv('exp_one_step.csv')
#
# data['integral_reimpact'] = pd.cut(data['reward'], bins=[g for g in range(-100, 10, 10)],right=False,
#                                 precision=0, include_lowest=True)
#
#
#
# ax = sns.countplot(x='mode', hue='integral_reimpact', data=data)
# plt.title('Distribution of integral of reimpact')
# plt.show()
#################################

######## correlation plot ###########

# data = pd.read_csv('exp_one_step.csv')

# data = data.loc[(data['mode'] != 'monkey') & data['vel'] > 0]


# g = sns.lmplot(x="step", y="vel", hue="mode",
#                truncate=True, data=data)
# g.set_axis_labels("time steps", "imp_vel(m/s)")
# plt.show()
####################################

######## bar plot　###########

# data = pd.read_csv('exp_one_step.csv')
#
# data = data.loc[(data['mode'] == 'combined')]
#
# data = data.loc[(data['action'] != 0) & data['vel'] > 0]
#
#
# # g = sns.lmplot(x="step", y="vel", hue="action",
# #                truncate=False, data=data, legend=['2','3'])
# # g.set_axis_labels("time steps", "imp_vel(m/s)")
#
# action1 = data.query("action == 1")
# action2 = data.query("action == 2")
# labels = []
#
# ax = sns.kdeplot(action2['step'], action2['vel'],
#                  cmap="Greens", shade=True, shade_lowest=False, legend=True)
# label_patch1 = mpatches.Patch(color='lightgreen', label='4.5/60')
# labels.append(label_patch1)
# ax = sns.kdeplot(action1['step'], action1['vel'],
#                  cmap="Blues", shade=True, shade_lowest=False, legend=True)
# label_patch2 = mpatches.Patch(color='lightblue', label='9/60')
# labels.append(label_patch2)
#
#
# ax.legend(handles=labels, loc='upper left')
# plt.title('speed choice by "combined" agent')
# plt.show()


############################


######show tsplot###########
window = 300


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

import json

with open('following_25.2.2_log.csv') as f:
   data = json.load(f)

data = pd.DataFrame(data)

cur1 = data['mean_q']
cur2 = data['episode_reward']

# cur1 = pd.read_json('following_23.1.2.4_log.csv')['episode_reward']
# cur2 = pd.read_json('following_23.1.2.3_log.csv')['episode_reward']

# cur1 = np.array([])
# cur2 = np.array([])

cur1 = running_mean(np.array(cur1), window)
std_cur1 = np.std(cur1)

cur2 = running_mean(np.array(cur2), window)
std_cur2 = np.std(cur2)

l = min(len(cur1),len(cur2))
C_params = np.linspace(1, l, l)
cur1 = cur1[0:l]
cur2 = cur2[0:l]

cur1_min = 0
cur1_max = 1000

cur2_min = 0
cur2_max = 1000

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
par1.set_ylabel("episode_len")
host.set_ylabel("reward")

p1, = host.plot(C_params, cur1,label="episode length", color='red')
p2, = par1.plot(C_params, cur2, label="reward", color='green')
host.fill_between(C_params, cur1 - std_cur1, cur1 + std_cur1, alpha = 0.1, color="red")
par1.fill_between(C_params, cur2 - std_cur2, cur2 + std_cur2, alpha = 0.1, color="green")
par1.set_ylim(cur2_min, cur2_max)
host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["left"].label.set_color(p2.get_color())

host.axis["left"].major_ticklabels.set_color(p1.get_color())
par1.axis["left"].major_ticklabels.set_color(p2.get_color())

# plt.title('Approaching bumper')
plt.draw()
plt.show()






