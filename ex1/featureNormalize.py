def featureNormalize(x, addOnes = False):
    import numpy as np
    m = x.shape[0]
    try:
        n = x.shape[1]
    except IndexError:
        n = 1
    sigma = np.std(x,axis=0)
    mu = np.mean(x,axis=0)
    x_norm = x - mu
    x_norm = x_norm / sigma
    if addOnes == True:
        x_norm = np.concatenate((np.ones((m,1)), x_norm.reshape(m,n)), axis=1)
    return x_norm, mu, sigma
