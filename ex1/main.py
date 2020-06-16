import numpy as np
import matplotlib.pyplot as plt

from plotData import plotData
from featureNormalize import featureNormalize
from computeCostMulti import computeCostMulti
from gradientDescentMulti import gradientDescentMulti

data = np.loadtxt('ex1/ex1data1.txt', delimiter=',')
m, n = data.shape
# plotData('scatter', data[:,0], data[:,1], 'data1', 'population', 'profit')
# plt.show()
X = np.concatenate((np.ones((m,1)), data[:,0:n-1]), axis=1)
y = data[:,1].reshape(m,1)
theta = np.zeros((n, 1))

iterations = 1500
alpha = 0.01

J = computeCostMulti(X, y, theta)
print('\nWith theta = [0 ; 0]\nCost computed = ', J,'\nExpected cost value (approx) 32.07\n')
J = computeCostMulti(X, y, np.array([[-1], [2]]))
print('\nWith theta = [-1 ; 2]\nCost computed = ', J,'\nExpected cost value (approx) 54.24\n')


theta, J_history = gradientDescentMulti(X, y, theta, alpha, iterations)

print('Theta found by gradient descent:\n',theta,'\nExpected theta values (approx)\n -3.6303\n  1.1664\n\n')


# plotData('scatter', data[:,0],data[:,1], 'data1', 'population', 'profit')
# plotData('plot', data[:,0], X.dot(theta), 'data1', 'Training data', 'Linear regression', color='blue')
# plt.show()

predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of \n', predict1*10000)
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of \n', predict2*10000)

# plotData('plot', np.array(range(iterations)), J_history, 'Cost function', 'iterations', 'J')
# plt.show()


data2 = np.loadtxt('ex1/ex1data2.txt', delimiter=',')
m, n = data2.shape

x_norm, mu, sigma = featureNormalize (data2[:,0:n-1],True)
y = data2[:,1].reshape(m,1)
iterations = 800
alpha = 0.01
theta = np.zeros((n, 1))


theta, J_history = gradientDescentMulti(x_norm, y, theta, alpha, iterations)
plotData('plot', np.array(range(iterations)), J_history, 'Cost function', 'iterations', 'J')
plt.show()
