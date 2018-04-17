import numpy as np
import csv

pred_len = 25
obs_len = 2000

high_pass_f = 0.5
low_pass_f = 1.5
dt = 0.2

model_dir = "o%d_p%d_d%.1f_f%.1f-%.1f/prediction/" % (obs_len, pred_len, dt, high_pass_f, low_pass_f)


decoded = np.genfromtxt(model_dir+'predicted__110140.csv',delimiter=',')
target = np.genfromtxt(model_dir+'target__110140.csv',delimiter=',')

error = np.abs(decoded - target)
mean = np.mean(error)
print(mean)

