def costFunction(theta, X, y):
    import numpy as np
    from sigmoid import sigmoid
    m = len(y)
    # h = sigmoid(x.dot(theta))
    # if np.sum(1-h < 1e-10) != 0:
    #     return np.inf
    J = (-y.T @ (np.log(sigmoid(X @ theta))) - (1 - y).T @ (np.log(1 - sigmoid(X @ theta))))/m
    
    return J.reshape(1,)

