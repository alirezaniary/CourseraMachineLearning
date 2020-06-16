def computeCostMulti(X, y, theta):
    import numpy as np
    m = X.shape[0]
    J = 0.5 * np.transpose(X @ theta - y).dot((X @ theta - y)) / m

    return J