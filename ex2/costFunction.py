def costFunction(theta, X, y):
    import numpy as np
    from sigmoid import sigmoid
    m = len(y)
    J = sum(-y * np.log(sigmoid(X.dot(theta))) - (1 - y) * np.log(1 - sigmoid(X.dot(theta))))/m
    grad = np.transpose(X).dot(sigmoid(X.dot(theta)) - y) / m
    
    return J, grad