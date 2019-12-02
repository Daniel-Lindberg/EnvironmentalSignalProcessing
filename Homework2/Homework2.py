#Author: Daniel Lindberg
# Native python modules
import scipy.io
import random
import math
import numpy as np
import numpy.fft as fft
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import scipy.signal as signal


# Native python submodules
from spectrum.arma import arma_estimate, arma2psd
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
from scipy.fftpack import rfft, irfft, fftfreq

mat_2 = scipy.io.loadmat('hw2_2.mat')
mat_3 = scipy.io.loadmat('hw2_3.mat')

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

def acorrBiased(y):
	"""Obtain the biased autocorrelation and its lags
	"""
	r = signal.correlate(y, y) / len(y)
	l = np.arange(-(len(y)-1), len(y))
	return r,l

def blackmanTukey(y, w, Nfft, fs=1):
	# Code from https://gist.github.com/ul/05dc52bd6121335c5a8e1acadf31dbd6
	"""Evaluate the Blackman-Tukey spectral estimator

	Parameters
	----------
	y : array_like
	  Data
	w : array_like
	  Window, of length <= y's
	Nfft : int
	  Desired length of the returned power spectral density estimate. Specifies
	  the FFT-length.
	fs : number, optional
	  Sample rate of y, in samples per second. Used only to scale the returned
	  vector of frequencies.

	Returns
	-------
	phi : array
	  Power spectral density estimate. Contains ceil(Nfft/2) samples.
	f : array
	  Vector of frequencies corresponding to phi.
	"""
	M = len(w)
	N = len(y)
	if M>N:
		raise ValueError('Window cannot be longer than data')
	r, lags = acorrBiased(y)
	r = r[np.logical_and(lags >= 0, lags < M)]
	rw = r * w
	phi = 2 * fft.fft(rw, Nfft).real - rw[0];
	f = np.arange(Nfft) / Nfft;
	return (phi[f < 0.5], f[f < 0.5] * fs)


# Problem 2.2
frequency_points = []
Hr_points = []
Hi_points = []
total_points = []
for sub_triple in mat_2['d']:
	frequency_points.append(sub_triple[0])
	Hr_points.append(sub_triple[1])
	Hi_points.append(sub_triple[2])
	total_points.append(sub_triple[1]+(complex()*sub_triple[2]))
time_stamps = []
for sub_frequency in frequency_points:
	"""
	w = 1/T
	Tw = 1
	T = 1/w
	"""
	time_stamps.append((1/sub_frequency))
# Part 2.2(b)
for j in range(1, 100):
	part_b = []
	for i in range(len(total_points)):
		temp_val = total_points[i] + np.sum(total_points) * (random.random()*len(total_points))+complex()*(random.random()*len(total_points))/math.sqrt(j)
		part_b.append(temp_val)
	coefs_b = poly.polyfit(time_stamps, part_b, j)
	ffit = poly.polyval(time_stamps, coefs_b)
	transfer_b = signal.TransferFunction(coefs_b, [1.0], dt=time_stamps)
	print j
	print "Zeros_"+str(j)+":",transfer_b.zeros, transfer_b.zeros.shape 
	print "Poles_"+str(j)+":",transfer_b.poles,transfer_b.poles.shape

coefs = poly.polyfit(time_stamps, total_points, 77)
ffit = poly.polyval(time_stamps, coefs)
plt.figure()
plt.plot(time_stamps, ffit)
transfer_f = signal.TransferFunction(coefs, [1.0], dt=time_stamps)
print "Zeros:",transfer_f.zeros, transfer_f.zeros.shape 
print "Poles",transfer_f.poles,transfer_f.poles.shape
np.savetxt('Zeros.txt', transfer_f.zeros)
np.savetxt('Poles.txt', transfer_f.poles)

# Problem 2.3
x_coordinates_3 = []
y_coordinates_3 = []
for sub_tuple in mat_3['d']:
	x_coordinates_3.append(sub_tuple[0])
	y_coordinates_3.append(sub_tuple[1])
# Part 2.3(A)
plt.figure()
f, Pxx_den = signal.welch(y_coordinates_3)
plt.semilogy(f, Pxx_den)
plt.title('Welch Periodogram')
plt.xlabel('Frequencies')
plt.ylabel('Power Spectral Density')

#Part 2.3(B)
btWin = signal.hamming(len(y_coordinates_3))
db20 = lambda x: np.log10(np.abs(x)) * 20
phi, f = blackmanTukey(y_coordinates_3, btWin, len(y_coordinates_3))
plt.figure()
plt.plot(x_coordinates_3, db20(phi))
plt.title('Blackman-Tukey')
plt.xlabel('Frequencies')
plt.ylabel('Spectral Estimate')

#Part 2.3(C)
plt.figure()
ar, ma, rho = arma_estimate(y_coordinates_3, 15, 15, 30)
psd = arma2psd(ar, ma, rho=rho, NFFT=4096)
plt.plot(10*np.log10(psd/max(psd)))
plt.axis([0, 4096, -80, 0])
plt.title('AutoRegressive')
plt.xlabel('Frequencies')
plt.ylabel('Spectral Estimate')

plt.figure()
plt.subplot(2,1,1)
# Part 2.5
frequencies = 1000.0  # Sampling frequency
time_stamps_2 = np.arange(1000) / frequencies
signal_1 = np.sin(2*np.pi*100*time_stamps_2) # with frequency of 100
plt.plot(time_stamps_2, signal_1, color='r', label='Signal_1')

signal_2 = np.sin(2*np.pi*20*time_stamps_2) # frequency 20
plt.plot(time_stamps_2, signal_2, color='b', label='Signal_2')

signal_3 = signal_1 + signal_2
plt.plot(time_stamps_2, signal_3, color='g', label='Signal_3')
plt.title('FirstOrder Buttworth')
cut_off = 30  # Cut-off frequency for the low pass
critical_frequencies = cut_off / (frequencies / 2) # Normalize the frequency
denominator_signal, numerator_signal = signal.butter(5, critical_frequencies, 'low')
output = signal.filtfilt(denominator_signal, numerator_signal, signal_3)
plt.plot(time_stamps_2, output, color='m', label='filtered')
plt.legend()
plt.subplot(2,1,2)
auto_correlation = autocorr(output)
#plt.axis([0, 4096, -80, 0])
plt.plot(auto_correlation)
plt.title('AutoCorrelation')

# Part 2.5(2)
plt.figure()
plt.subplot(2,1,1)
f0 = 7 # center frequency
B = 6 # bandwidth
time_stamp_3 = np.linspace(0,10,2000)
signal_2 = np.cos(5*np.pi*time_stamp_3) + np.cos(f0*np.pi*time_stamp_3)

critical_frequencies_2 = fftfreq(signal_2.size, d=time_stamp_3[1]-time_stamp_3[0])
f_signal = rfft(signal_2)

# If our original signal_2 time_stamp_3 was in seconds, this is now in Hz    
cut_f_signal = f_signal.copy()
cut_f_signal[(critical_frequencies_2<B)] = 0
plt.plot(time_stamp_3, signal_2, color='g', label='Original')

filtered_signal = irfft(cut_f_signal)
plt.plot(time_stamp_3, filtered_signal, color='r', label='bandpass')
plt.legend()
plt.title('Ideal Bandpass')
plt.subplot(2,1,2)
plt.title('AutoCorrelation')
auto_correlation_2 = autocorr(filtered_signal)
plt.plot(auto_correlation_2)

plt.show()
