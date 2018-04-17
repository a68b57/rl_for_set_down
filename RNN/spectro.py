from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import spec_tools.spec_tools as st



resp = st.Spectrum.from_synthetic(spreading=None, Hs=3, Tp=8)
np.random.seed(11)
_, wave = resp.make_time_trace(100, 0.2)
window = 500

# for i in range(100):
# 	segment = wave[i*5:i*5+window]
#
# 	# yf = np.fft.fft(wave)
# 	# xf = np.linspace(0.0, 1.0/(2.0*0.2), 1000//2)
# 	# plt.plot(xf, 2.0/1000 * np.abs(yf[0:1000//2]))
# 	# plt.grid()
# 	# plt.show()
#
# 	f, t, Sxx = signal.spectrogram(segment, fs=5, window=('hamming'),nperseg=20, noverlap=10, nfft=20)
# 	plt.pcolormesh(t, f, Sxx)
# 	plt.ylabel('Frequency [Hz]')
# 	plt.xlabel('Time [sec]')
# 	plt.show()


f, t, Sxx = signal.stft(wave, fs=5, nperseg=10)
# plt.pcolormesh(t, f, np.abs(Sxx))
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

_, xrec = signal.istft(Sxx, 5)
plt.plot(wave)
plt.plot(xrec)
plt.show()
