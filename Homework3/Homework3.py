#Author: Daniel Lindberg
# Native python modules
import scipy.io
import random
import math
import statistics
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Native python submodules
from sklearn.decomposition import PCA

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


# Part 3.3
mat_3 = scipy.io.loadmat('hw3_3.mat')
Y = mat_3["Y"]
matrix_1 = []
matrix_2 = []
matrix_3 = []
matrix_4 = []
matrix_5 = []
matrix_6 = []
for i in range(Y.shape[0]):
	matrix_1.append(Y[i][0])
	matrix_2.append(Y[i][1])
	matrix_3.append(Y[i][2])
	matrix_4.append(Y[i][3])
	matrix_5.append(Y[i][4])
	matrix_6.append(Y[i][5]) 
full_matrix = np.array([matrix_1, matrix_2, matrix_3, matrix_4, matrix_5, matrix_6])
correlation_coefficient = np.corrcoef(full_matrix)
covariance_matrix = np.cov(full_matrix)
eigen_matrix = (covariance_matrix * covariance_matrix.transpose())
eigen_values, eigen_vectors = np.linalg.eig(eigen_matrix)
print "Eigen Values:", eigen_values
print "Eigen Vectors:", eigen_vectors
print "Diagonal Eigen Matrix:", eigen_vectors[0][0], eigen_vectors[1][1], eigen_vectors[2][2], eigen_vectors[3][3], eigen_vectors[4][4], eigen_vectors[5][5]  
print "Principal Components Eigens (5, 4, 3):", eigen_vectors[4][4], eigen_vectors[3][3], eigen_vectors[2][2] 

three_matrix = np.array([matrix_3, matrix_4, matrix_5])
pca = PCA(n_components=3)
pca.fit(three_matrix)
pca_components = pca.components_.transpose()
# Plotting functions
plt.figure()
plt.subplot(2,1,1)
plt.plot(correlation_coefficient[0], label="Signal1")
plt.plot(correlation_coefficient[1], label="Signal2")
plt.plot(correlation_coefficient[2], label="Signal3")
plt.plot(correlation_coefficient[3], label="Signal4")
plt.plot(correlation_coefficient[4], label="Signal5")
plt.plot(correlation_coefficient[5], label="Signal6")
plt.xlabel("ith elements")
plt.ylabel('jth lements')
#plt.plot(correlation_coefficient)
plt.title("Correlation Coefficient Matrix")
plt.legend()
plt.subplot(2,1,2)
plt.plot(covariance_matrix[0], label="Signal1")
plt.plot(covariance_matrix[1], label="Signal2")
plt.plot(covariance_matrix[2], label="Signal3")
plt.plot(covariance_matrix[3], label="Signal4")
plt.plot(covariance_matrix[4], label="Signal5")
plt.plot(covariance_matrix[5], label="Signal6")
plt.xlabel("ith elements")
plt.ylabel('jth lements')
#plt.plot(covariance_matrix)
plt.title("Covariance Matrix")
plt.legend()
plt.figure()
plt.subplot(2,1,1)
plt.title("Eigen Values")
plt.plot(eigen_values)
plt.xlabel("Signal units")
plt.ylabel('EigenValues')
plt.subplot(2,1,2)
plt.title("Eigen Vectors")
plt.plot(eigen_vectors)
plt.xlabel("Signal units")
plt.ylabel('EigenVectors')
plt.figure()
[s_1, s_2, s_3] = plt.plot(pca_components)
plt.title("PCA Values")
plt.xlabel("Time(t)")
plt.legend([s_1,s_2,s_3], ("Signal3","Signal4","Signal5"))
plt.figure()
plt.subplot(3,1,1)
plt.plot(range(140,170), pca.components_[0][140:170], label="Signal3")
plt.plot(range(140,170), pca.components_[1][140:170], label="Signal4")
plt.plot(range(140,170), pca.components_[2][140:170], label="Signal5")
plt.xlabel("Time(t)")
plt.ylabel("Principal Components")
plt.legend()
plt.subplot(3,1,2)
plt.plot(range(250,280), pca.components_[0][250:280], label="Signal3")
plt.plot(range(250,280), pca.components_[1][250:280], label="Signal4")
plt.plot(range(250,280), pca.components_[2][250:280], label="Signal5")
plt.xlabel("Time(t)")
plt.ylabel("Principal Components")
plt.legend()
plt.subplot(3,1,3)
plt.plot(range(60,90), pca.components_[0][60:90], label="Signal3")
plt.plot(range(60,90), pca.components_[1][60:90], label="Signal4")
plt.plot(range(60,90), pca.components_[2][60:90], label="Signal5")
plt.xlabel("Time(t)")
plt.ylabel("Principal Components")
plt.legend()


# Part 3.4
mat_4 = scipy.io.loadmat('hw3_4.mat')
y_3 = mat_3['Y']

d_4 = mat_4['d']
dc_4 = mat_4['dc']
dpn_4 = mat_4['dpn']
dcpn_4 = mat_4['dcpn']
pattern_4 = mat_4['pattern']
rxx_est_4 = mat_4['Rxx_est']
sigman_4 = mat_4['sigman']
pre_snr = signaltonoise(dpn_4)

some_result_4 = signal.wiener(dpn_4, noise=3)
post_snr = signaltonoise(some_result_4)
plt.figure()
plt.subplot(2,1,1)
plt.title("Pre Wiener Filter")
plt.imshow(dpn_4)
plt.subplot(2,1,2)
plt.title("Post Wiener Filter")
plt.imshow(some_result_4)
plt.figure()
plt.subplot(1,1,1)
plt.plot(pre_snr, label="Pre Wiener")
plt.plot(post_snr, label="Post Wiener")
plt.ylabel("Signal Units")
plt.xlabel("Time(t)")
plt.title("SNR")
plt.legend()

some_result_4_2 = signal.wiener(dpn_4, noise=np.var(dpn_4))
# np.var(dpn_4) = 547.75189
post_snr_2 = signaltonoise(some_result_4_2)

plt.figure()
plt.subplot(2,1,1)
plt.title("Pre Wiener Filter")
plt.imshow(dpn_4)
plt.subplot(2,1,2)
plt.title("Post Wiener Filter")
plt.imshow(some_result_4_2)
plt.figure()
plt.subplot(1,1,1)
plt.plot(pre_snr, label="Pre Wiener")
plt.plot(post_snr_2, label="Post Wiener")
plt.ylabel("Signal Units")
plt.xlabel("Time(t)")
plt.title("SNR")
plt.legend()

######################################################################
pre_dcpn_snr = signaltonoise(dcpn_4)

some_result_dcpn_4 = signal.wiener(dcpn_4, noise=3)
post_dcpn_snr = signaltonoise(some_result_dcpn_4)
plt.figure()
plt.subplot(2,1,1)
plt.title("Pre Wiener Filter")
plt.imshow(dcpn_4)
plt.subplot(2,1,2)
plt.title("Post Wiener Filter")
plt.imshow(some_result_dcpn_4)
plt.figure()
plt.subplot(1,1,1)
plt.plot(pre_dcpn_snr, label="Pre Wiener")
plt.plot(post_dcpn_snr, label="Post Wiener")
plt.ylabel("Signal Units")
plt.xlabel("Time(t)")
plt.title("SNR")
plt.legend()

some_result_dcpn_4_2 = signal.wiener(dcpn_4, noise=np.var(dcpn_4))
# np.var(dcpn_4) = 534.306057497
post_dcpn_snr_2 = signaltonoise(some_result_dcpn_4_2)

plt.figure()
plt.subplot(2,1,1)
plt.title("Pre Wiener Filter")
plt.imshow(dcpn_4)
plt.subplot(2,1,2)
plt.title("Post Wiener Filter")
plt.imshow(some_result_dcpn_4_2)
plt.figure()
plt.subplot(1,1,1)
plt.plot(pre_dcpn_snr, label="Pre Wiener")
plt.plot(post_dcpn_snr_2, label="Post Wiener")
plt.ylabel("Signal Units")
plt.xlabel("Time(t)")
plt.title("SNR")
plt.legend()

plt.show()

