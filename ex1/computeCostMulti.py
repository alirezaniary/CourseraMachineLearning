def computeCostMulti(X, y, theta):
    import numpy as np
    m = X.shape[0]
    J = 0.5 * np.transpose(X.dot(theta) - y).dot((X.dot(theta) - y)) / m

    return J