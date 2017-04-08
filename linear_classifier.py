###################################################################################################################################
#@Author Yasin YILDIRIM                                                                                                           #
#@Licence MIT License                                                                                                             #
																																  #
#Copyright 2017 Yasin YILDIRIM																									  #
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation		  #
#files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use,             #
#copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons                  #
#to whom the Software is furnished to do so, subject to the following conditions:                                                 #
																																  #
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.   #
																															      #
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 			  #
#WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR            #
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,      #
#ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                            #
# see https://opensource.org/licenses/MIT for details.  																	      #
###################################################################################################################################

import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import BayesClassifier
from scipy import optimize
from scipy.optimize import leastsq

def find_boundary(x, y, n, plot_pts=1000):

    def sines(theta):
        ans = np.array([np.sin(i*theta)  for i in range(n+1)])
        return ans

    def cosines(theta):
        ans = np.array([np.cos(i*theta)  for i in range(n+1)])
        return ans

    def residual(params, x, y):
        x0 = params[0]
        y0 = params[1]
        c = params[2:]

        r_pts = ((x-x0)**2 + (y-y0)**2)**0.5

        thetas = np.arctan2((y-y0), (x-x0))
        m = np.vstack((sines(thetas), cosines(thetas))).T
        r_bound = m.dot(c)

        delta = r_pts - r_bound
        delta[delta>0] *= 10

        return delta

    # initial guess for x0 and y0
    x0 = x.mean()
    y0 = y.mean()

    params = np.zeros(2 + 2*(n+1))
    params[0] = x0
    params[1] = y0
    params[2:] += 1000

    popt, pcov = leastsq(residual, x0=params, args=(x, y),
                         ftol=1.e-12, xtol=1.e-12)

    thetas = np.linspace(0, 2*np.pi, plot_pts)
    m = np.vstack((sines(thetas), cosines(thetas))).T
    c = np.array(popt[2:])
    r_bound = m.dot(c)
    x_bound = x0 + r_bound*np.cos(thetas)
    y_bound = y0 + r_bound*np.sin(thetas)

    return x_bound, y_bound
def discr_func(x, y, cov_mat, mu_vec):
    """
    Calculates the value of the discriminant function for a dx1 dimensional
    sample given covariance matrix and mean vector.

    Keyword arguments:
            x_vec: A dx1 dimensional numpy array representing the sample.
            cov_mat: numpy array of the covariance matrix.
            mu_vec: dx1 dimensional numpy array of the sample mean.

        Returns a float value as result of the discriminant function.

        """
    x_vec = np.array([[x],[y]])

    W_i = (-1/2) * np.linalg.inv(cov_mat)
    assert(W_i.shape[0] > 1 and W_i.shape[1] > 1), 'W_i must be a matrix'

    w_i = np.linalg.inv(cov_mat).dot(mu_vec)
    assert(w_i.shape[0] > 1 and w_i.shape[1] == 1), 'w_i must be a column vector'

    omega_i_p1 = (((-1/2) * (mu_vec).T).dot(np.linalg.inv(cov_mat))).dot(mu_vec)
    omega_i_p2 = (-1/2) * np.log(np.linalg.det(cov_mat))
    omega_i = omega_i_p1 - omega_i_p2
    assert(omega_i.shape == (1, 1)), 'omega_i must be a scalar'

    g = ((x_vec.T).dot(W_i)).dot(x_vec) + (w_i.T).dot(x_vec) + omega_i
    return float(g)

         
 
#application code 
C1_x, C1_y = [], []
C2_x, C2_y = [], []
with open('C1_pts.txt','r') as f:
    reader = csv.reader(f,delimiter=' ')
    index = 0
    
    ## get the points from file
    for row in reader:
        index += 1
        if(index <5):
            continue
        firstItemAppended = False
        #print row
        for rowItem in row:
            
            if(rowItem == ''):
                continue
            elif ( firstItemAppended == False) :
                
                C1_x.append(float(rowItem))
                firstItemAppended = True
            elif( firstItemAppended == True):
                C1_y.append(float(rowItem))
    #print C1_x, C1_y

with open('C2_pts.txt','r') as f:
    reader = csv.reader(f,delimiter=' ')
    index = 0
    
    ## get the points from file
    for row in reader:
        index += 1
        if(index <5):
            continue
        firstItemAppended = False
        #print row
        for rowItem in row:
            
            if(rowItem == ''):
                continue
            elif ( firstItemAppended == False) :
                
                C2_x.append(float(rowItem))
                firstItemAppended = True
            elif( firstItemAppended == True):
                C2_y.append(float(rowItem))
C1_features = list()
C2_features = list()
C1_z = np.array(C1_x)**2 + np.array(C1_y)**2
C2_z = np.array(C2_x)**2 + np.array(C2_y)**2 
C1_features.append(C1_x)
C1_features.append(C1_y)
#C1_features.append(C1_z)
C2_features.append(C2_x)
C2_features.append(C2_y)
#C2_features.append(C2_z)

classifier = BayesClassifier.GaussianBayesClassifier()

C1_distribution = classifier.getClassDistribution(C1_features)
C2_distribution = classifier.getClassDistribution(C2_features)
C1_x_variance = C1_distribution[1][0] * C1_distribution[1][0]
C1_y_variance = C1_distribution[1][1] * C1_distribution[1][1]
C1_cov_matrix = classifier.calc_2d_covariance_matrix(C1_x, C1_y,
 C1_distribution[0][0],C1_distribution[0][1],C1_x_variance,C1_y_variance)

C2_x_variance = C2_distribution[1][0] * C2_distribution[1][0]
C2_y_variance = C2_distribution[1][1] * C2_distribution[1][1]
C2_cov_matrix = classifier.calc_2d_covariance_matrix(C2_x, C2_y,
 C2_distribution[0][0],C2_distribution[0][1],C2_x_variance,C2_y_variance)

x_est50 = list(np.arange(-6, 6, 0.1))
y_est50 = []
mu_vec_1 = np.zeros((2,1))
mu_vec_1[0][0] = C1_distribution[0][0]
mu_vec_1[1][0] = C1_distribution[0][1]

mu_vec_2 = np.zeros((2,1))
mu_vec_2[0][0] = C2_distribution[0][0]
mu_vec_2[1][0] = C2_distribution[0][1]

np.reshape(mu_vec_2, 2)
"""
for i in x_est50:
    y_est50.append(scipy.optimize.bisect(lambda y: discr_func(i, y, cov_mat=C1_cov_matrix, mu_vec=mu_vec_1) - 
    discr_func(i, y, cov_mat=C2_cov_matrix, mu_vec=mu_vec_2), -10,10))
    y_est50 = [float(i) for i in y_est50]
    """
x1, y1 = np.random.multivariate_normal(C1_distribution[0], C1_cov_matrix, 5000).T
x2, y2 = np.random.multivariate_normal(C2_distribution[0], C2_cov_matrix, 5000).T

z1 = (1/(2*np.pi*math.sqrt(C1_x_variance)*math.sqrt(C1_x_variance)) * np.exp(-(x1**2/(2*C1_x_variance)
     + y1**2/(2*C1_x_variance))))
z2 = (1/(2*np.pi*math.sqrt(C2_x_variance)*math.sqrt(C2_x_variance)) * np.exp(-(x2**2/(2*C2_x_variance)
     + y2**2/(2*C2_x_variance))))
     

plt.plot(C1_x, C1_y, 'bx')
plt.plot(C2_x, C2_y, 'ro')
#plt.plot(x1, y1, '--b')  
#plt.plot(x2, y2, '--r')
x1b, y1b = find_boundary(x1, y1, 2)
x2b, y2b = find_boundary(x2, y2, 2)
#xval = np.roots(y1b - y2b)
#yval = np.polyval(y1b, xval)
#plt.plot(xval, yval, '--k', lw=2.)
plt.plot(x1b, y1b, '--k', lw=2.)
plt.plot(x2b, y2b, '--k', lw=2.)
plt.xlabel('x axis ')
plt.ylabel('y axis ')
plt.title('input points')
plt.legend(['Class 1', 'Class 2'])
plt.show()

