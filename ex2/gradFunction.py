def gradFunction(theta, X, y):
    import numpy as np
    from sigmoid import sigmoid
    m, n = X.shape
    grad = X.T @ (sigmoid(X @ theta) - y) / m
    
    return grad.reshape(n, )

