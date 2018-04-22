from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

raw = np.loadtxt("./Data/all2.txt", delimiter=' ')



N = 250
t = np.arange(N)/float(N)
f1 = 10
f2 = 30
f3 = 50
f4 = 70
x1 = 1*np.sin(2*np.pi*t[:30]*f1)
x2 = 2*np.sin(2*np.pi*t[30:60]*f2)
x3 = 4*np.sin(2*np.pi*t[60:90]*f3)
x4 = 8*np.sin(2*np.pi*t[90:]*f4)
x = np.concatenate((x1,x2,x3,x4))


f, t, Zxx = signal.stft(x, N, nperseg=10)
plt.specgram(x, NFFT=64, Fs=250, noverlap=50)
# plt.pcolormesh(t,f,np.abs(Zxx))
plt.title("STFT Magnitude")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.show()