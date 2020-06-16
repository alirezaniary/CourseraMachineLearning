def gradientDescentMulti(X, y, theta, alpha, iterations):
    import numpy as np
    from computeCostMulti import computeCostMulti
    m = X.shape[0]
    J_history = np.zeros((iterations,1))
    for i in range(iterations):
        theta = theta - alpha * np.transpose(X) @ (X.dot(theta) - y) / m  
        J_history[i] = computeCostMulti(X, y, theta)
    return theta, J_history