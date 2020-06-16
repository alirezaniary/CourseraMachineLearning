import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from plotData import plotData
from sigmoid import sigmoid
from costFunction import costFunction
from gradFunction import gradFunction

data = np.loadtxt('ex2/ex2data1.txt', delimiter=',')
m, n = data.shape
print(m,n)
n = n-1
X = data[:, 0:2]
print(X.shape)
y = data[:, 2].reshape(m, 1)
print(y.shape)
# x1 = X[:,0]
# x2 = X[:,1]
# plotData('scatter', x1[np.nonzero(y == 1)[0]], x2[np.nonzero(y == 1)[0]], 'data1', 'Exam 1 score', 'Exam 2 score')
# plotData('scatter', x1[np.nonzero(y == 0)[0]], x2[np.nonzero(y == 0)[0]], 'data1', 'Exam 1 score', 'Exam 2 score',marker='o')
# plt.show()

X = np.concatenate((np.ones((m,1)), X), axis=1)
print(X.shape)

init_theta = np.zeros((n+1,))
print(init_theta.shape)
cost = costFunction(init_theta, X, y)
print(cost, cost.shape)
grad = gradFunction(init_theta, X, y)
print(grad.shape)
result = opt.minimize(costFunction, x0=init_theta, method='BFGS', jac=gradFunction, args=(X, y))
theta = result.x
print('Cost at theta found by fmin_bfgs: ', result.fun)
print('theta: ', theta)




