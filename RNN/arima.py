import spec_tools.spec_tools as st
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.ar_model import AR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pred_len = 20
obs_len = 200 # 10 periods
# tol_length = 220
tol_length = 20000

dt = 0.2
hs = 3
tp = 10

resp = st.Spectrum.from_synthetic(spreading=None, Hs=hs, Tp=tp)

data_set = []
for i in range(10):
	print(i)
	rel_motion_sc_t, data = resp.make_time_trace(tol_length, dt)
	data_set.append(data)


data_set = np.array(data_set)
result = adfuller(data_set[0])

#p-value > 0.05: Accept the null hypothesis (H0), the data has a unit root and is non-stationary.
#p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

np.random.seed(10)
data = st.Spectrum.from_synthetic(spreading=None, Hs=hs, Tp=tp).make_time_trace(tol_length, dt)[1]

acf_1 = acf(data,nlags=500)
pacf_1 = pacf(data,nlags=500)

acf_2 = acf(st.Spectrum.from_synthetic(spreading=None, Hs=3, Tp=20).make_time_trace(tol_length, dt)[1],nlags=int(15/dt))
acf_3 = acf(st.Spectrum.from_synthetic(spreading=None, Hs=1.5, Tp=10).make_time_trace(tol_length, dt)[1],nlags=int(8/dt))
acf_4 = acf(st.Spectrum.from_synthetic(spreading=None, Hs=3, Tp=10).make_time_trace(tol_length, dt)[1],nlags=int(8/dt))
plt.plot(np.arange(0,len(pacf_1), 1), pacf_1)
plt.plot(np.arange(0,len(pacf_1), 1), (2/np.sqrt(len(pacf_1)))*np.ones(len(pacf_1)))
plt.plot(np.arange(0,len(pacf_1), 1), -(2/np.sqrt(len(pacf_1)))*np.ones(len(pacf_1)))
plt.plot(np.arange(0,len(pacf_1), 1), acf_1)

# plt.plot(acf_3)
# plt.plot(acf_4)
# plt.legend(['Hs:1,5,Tp:20','Hs:3,Tp:20','Hs:1,5,Tp:10','Hs:3,Tp:10'])
plt.title('Acf')
plt.ylabel('correlation')
plt.show()


model = ARIMA(endog=np.array(data_set[0]), order=(10,0,0))
model_fit = model.fit(disp=0, transparams=False)
print(model_fit.summary())

# model = AR(endog=np.array(data_set[0]))
# model_fit = model.fit(disp=True)
# print('Lag: %s' % model_fit.k_ar)
# print('Coefficients: %s' % model_fit.params)


