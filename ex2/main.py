import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotData import plotData
from sigmoid import sigmoid

# print(sigmoid(np.array([-1,0,1])))
data = pd.read_csv('ex2/ex2data1.txt',header=None)
data = np.array(data)
m, n = data.shape

X = data[:, 0:2].reshape(m, n-1)
y = data[:, 2].reshape(m, 1)
x1 = X[:,0]
x2 = X[:,1]
print(x1.shape)
plotData('scatter', x1[np.nonzero(y == 1)[0]], x2[np.nonzero(y == 1)[0]], 'data1', 'Exam 1 score', 'Exam 2 score')
plotData('scatter', x1[np.nonzero(y == 0)[0]], x2[np.nonzero(y == 0)[0]], 'data1', 'Exam 1 score', 'Exam 2 score',marker='o')
plt.show()
