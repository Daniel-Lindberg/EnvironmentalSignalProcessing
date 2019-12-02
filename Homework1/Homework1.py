#Author: Daniel Lindberg
# Native python modules
import scipy.io
import random
import math
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import scipy.signal as signal


# Native python submodules
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid, repeat, arange, cumsum, subplots_adjust


# This is for part 4
mat = scipy.io.loadmat('hw1_4.mat')

method = 'cholesky'
num_samples = 500
desired_mean = 0.0
desired_std_dev_1 = 1
desired_std_dev_2 = 2
desired_std_dev_3 = 3

correlation_cov = np.array([
        [ 1, 0.5, 0],
        [ 0.5, 4, -0.5],
        [ 0, -0.5, 9]
    ])

prony_outliers = [(0, 100000000000000),
				  (1, 100000000000000),
				  (189, 10),
				  (190, 10)]


def genDataSet(samples, std_given):
	actual_mean = np.mean(samples)
	actual_std = np.std(samples)
	zero_mean_samples = samples - (actual_mean)
	zero_mean_mean = np.mean(zero_mean_samples)
	zero_mean_std = np.std(zero_mean_samples)
	scaled_samples = zero_mean_samples * (std_given/zero_mean_std)
	scaled_mean = np.mean(scaled_samples)
	scaled_std = np.std(scaled_samples)
	return scaled_samples

def getDecomposition(cov_matrix):
	c = None
	if method == 'cholesky':
		c = cholesky(cov_matrix, lower=True)
	else:
		evals, evecs = eigh(cov_matrix)
		c = np.dot(evecs, np.diag(np.sqrt(evals)))
	return c

def calculateCost(theta, x, y):
	m = len(y)
	predictions = x.dot(theta)
	cost = (1/2*m) * np.sum(np.square(predictions=y))
	return cost

def gradientDescent(alpha, x_values, y_values, ep=0.0001, max_iter=1000):
    converged = False
    iter = 0
    m = len(x_values) # number of samples
    # initial theta
    theta0 = np.random.random(1)
    theta1 = np.random.random(2)
    # total error, error_matrix(theta)
    error_matrix = sum([(theta0 + theta1*x_values[i] - y_values[i])**2 for i in range(m)])
    all_thetas = [theta1]
    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta error_matrix(theta))
        grad0 = 1.0/m * sum([(theta0 + theta1*x_values[i] - y_values[i]) for i in range(m)]) 
        grad1 = 1.0/m * sum([(theta0 + theta1*x_values[i] - y_values[i])*x_values[i] for i in range(m)])
        # update the theta_temp
        temp0 = theta0 - alpha * grad0
        temp1 = theta1 - alpha * grad1
        # update theta
        theta0 = temp0
        theta1 = temp1
        all_thetas.append([theta0,theta1])
        iter += 1  # update iter
        if iter == max_iter:
            #print 'Max interactions exceeded!'
            converged = True
        # mean squared error
        e = sum( [ (theta0 + theta1*x_values[i] - y_values[i])**2 for i in range(m)] ) 
        if abs(error_matrix[0]-e[0]) <= ep or abs(error_matrix[1]-e[1]) <= ep:
            converged = True
        error_matrix = e   # update error 
    return all_thetas

def updateReference(T, k):
    return T - 0.001

def getNeighbors(i, L):
    assert L > 1 and i >= 0 and i < L
    if i == 0:
        return [1]
    elif i == L - 1:
        return [L - 2]
    else:
        return [i - 1, i + 1]

def makeMove(x, y_coordinates, T):
	# Move around the function , decide a random spot along the y_coordinates
    nhb = random.choice(xrange(0, len(y_coordinates))) # choose from all points
    delta = y_coordinates[nhb] - y_coordinates[x]
    if delta < 0:
        return nhb
    else:
        p = math.exp(-delta / T)
        return nhb if random.random() < p else x

def simulatedAnnealing(y_model, y_coordinates, chi_squared, max_iter=500):
	#computer chi squared. single number. change parameter vector A, 
	#use simulated annealing.
	# recompute chi
    y_return_model = y_model[:]
    L = len(y_model)
    x0 = random.choice(xrange(0, L))
    T = 1.
    k = 1
    i = x0
    y_best = x0
    matrix_of_values = []
    while T > 1e-3 and len(matrix_of_values) < max_iter:
        i = makeMove(i, y_model, T)
        if(y_model[i] < y_model[y_best]):
            y_best = i
            temp_chi_squared = (y_coordinates[i] - y_model[i])**2
            multiplier = 1
            if (y_coordinates[i] - y_model[i]) < 0:
            	multiplier = -1
            y_return_model[i] = y_model[i] - (multiplier*temp_chi_squared)
        T = updateReference(T, k)
        k += 1
        matrix_of_values.append(y_coordinates[y_best])
    return y_return_model

def pronysMethod(y_model, y_coordinates, exponent):
  	"""Input  : real arrays y_model, y_coordinates of the same size (y_model(i), y_coordinates(i)
            : integer exponent - the number of modes in the exponential fit
    Output : arrays numerator and demoneator estimates such that y_coordinates(x_coordinate) ~ sum ai exp(bi*x_coordinates)"""
	N    = len(y_model)
	numerator_matrix = np.zeros((N-exponent, exponent))
	denomenator_matrix = y_coordinates[exponent:N]
	for jcol in range(exponent):
		numerator_matrix[:, jcol] = y_coordinates[exponent-jcol-1:N-1-jcol]
	sol = np.linalg.lstsq(numerator_matrix, denomenator_matrix)
	d = sol[0]
	c = np.zeros(exponent+1)
	c[exponent] = 1.
	for i in range(1,exponent+1):
		c[exponent-i] = -d[i-1]
	u = poly.polyroots(c)
	denominator_est = np.log(u)/(y_model[1] - y_model[0])
	numerator_matrix = np.zeros((N, exponent))
	denomenator_matrix = y_coordinates
	for irow in range(N):
		numerator_matrix[irow, :] = u**irow
	sol = np.linalg.lstsq(numerator_matrix, denomenator_matrix)
	numerator_est = sol[0]
	for prony_index, prony_value in prony_outliers:
		numerator_est[prony_index]/=prony_value
	return numerator_est, denominator_est

if __name__=="__main__":
	# Problem 1.2
	# Generate the random matrix of 3 by 500
	# First matrix is for the ones with std and rho
	first_matrix = norm.rvs(size=(3, num_samples))	
	# First set the first_matrix to have the desired standard deviation
	first_c = genDataSet(first_matrix[0], desired_std_dev_1)
	second_c = genDataSet(first_matrix[1], desired_std_dev_2)
	third_c = genDataSet(first_matrix[2], desired_std_dev_3)
	first_matrix[0] = first_c 
	first_matrix[1] = second_c
	first_matrix[2] = third_c
	# Next get the covariance
	cov_matrix = np.cov(first_matrix)
	# Get the decomposition
	decom = getDecomposition(cov_matrix)
	# Get the resultant matrix 2
	resultant = decom.dot(first_matrix)

	second_cov = np.cov(resultant)

	plt.figure()

	# Swap out first_matrix for second_matrix , dependent on which one you want to print
	plt.subplot(2,2,1)
	plt.plot(resultant[0], resultant[1], 'b.')
	plt.ylabel('resultant[1]')
	plt.axis('equal')
	plt.grid(True)

	plt.subplot(2,2,2)
	plt.plot(resultant[0], resultant[2], 'b.')
	plt.xlabel('resultant[0]')
	plt.ylabel('resultant[2]')
	plt.axis('equal')
	plt.grid(True)

	plt.subplot(2,2,3)
	plt.plot(resultant[1], resultant[2], 'b.')
	plt.xlabel('resultant[1]')
	plt.axis('equal')
	plt.grid(True)

	# Problem 1.4
	plt.figure()
	x_coordinates = np.transpose(mat['x'])[0]
	y_coordinates = np.transpose(mat['y'])[0]
	x_new = np.linspace(x_coordinates[0], x_coordinates[-1], num=len(x_coordinates)*10)
	coefs = np.polynomial.polynomial.polyfit(x_coordinates, y_coordinates, 4)
	ffit = np.polynomial.polynomial.polyval(x_new, coefs)
	y_model = []
	for i in range(0, len(y_coordinates)):
		y_model.append(ffit[i*10])
	plt.subplot(2,2,1)
	plt.plot(x_new, ffit, 'g.')
	plt.xlabel('PolyFit')
	plt.ylim(-2,2)
	plt.axis('equal')
	plt.grid(True)
	m = np.shape(x_coordinates)
	n = np.shape(y_coordinates)
	model_chi_squared = 0.0
	for i in range(0, len(y_coordinates)):
		#print i-1, (i-1)*10
		model_chi_squared += (y_coordinates[i]-y_model[i])**2
	model_chi_squared = model_chi_squared/np.std(y_coordinates)
	alpha = 0.00005
	every_theta = gradientDescent(alpha, y_model, y_coordinates, max_iter = m[0])
	y_predict = []
	gradient_chi_squared = 0.0
	for i in range(len(every_theta)):
		if i == 0:
			continue
		current_val = (every_theta[i][0][0] + every_theta[i][1][1]*y_coordinates[i-1])
		y_predict.append(current_val)
		gradient_chi_squared += (y_coordinates[i-1]-current_val) ** 2
	plt.subplot(2,2,2)
	plt.plot(x_coordinates, y_predict, 'r')
	plt.xlabel('gradient_descent')
	plt.ylim(-2,2)
	plt.axis('equal')
	plt.grid(True)
	y_model_simulated = simulatedAnnealing(y_model, y_coordinates, model_chi_squared)
	sa_chi_squared  = 0.0
	for i in range(0, len(y_model_simulated)):
		sub_chi = (y_coordinates[i]-y_model_simulated[i]) ** 2
		sa_chi_squared += sub_chi
	sa_chi_squared /= np.std(y_coordinates)
	plt.subplot(2,2,3)
	plt.plot(np.arange(0.0, 10.0, 0.02), y_model_simulated, 'b')
	plt.xlabel('simulated_annealing')
	plt.axis('equal')
	plt.grid(True)
	numerator_est, denominator_est = pronysMethod(y_model, y_coordinates, len(x_coordinates)-1)
	prony_y_coordinates = []
	prony_y_coordinates.append(y_model[0])
	prony_chi_squared = 0.0
	for i in range(1, len(y_coordinates)):
		prony_chi_squared += (y_coordinates[i]-numerator_est[i-1])**2
		prony_y_coordinates.append(numerator_est[i-1])
	prony_chi_squared /= np.std(y_coordinates)
	plt.subplot(2,2,4)
	plt.plot(np.arange(0.0, 10.0, 0.02), prony_y_coordinates, 'k')
	plt.xlabel('Pronys Method')
	plt.axis('equal')
	plt.grid(True)
	print "Gradient-Chi^2:"+str(gradient_chi_squared)
	print "SimmulatedAnnealing-Chi^2:"+str(sa_chi_squared)
	print "PronysMethod-Chi^2:"+str(prony_chi_squared)
	# the thetas are the 
	plt.show()


	
	

