def sigmoid(x):
    import numpy as np
    m = x.shape[0]
    try:
        n = x.shape[1]
    except IndexError:
        n = 1
    return (1 / (1 + np.exp(-x))).reshape(m,n)