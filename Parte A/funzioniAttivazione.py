import numpy as np

def sigmoide(x):
    y = (1/(1+np.exp(-x)))
    return y

def identity(x):
    y = x
    return y

def sumOfSquares(y,t):
    e = (1/2) * sum(np.power(y-t,2))
    return e

def derivCrossEntropy(Y,T):
    e = -sum(T / Y)
    return e

def derivSumOfSquares(y,t):
    e = y-t
    return e

def derivSigmoide(x):
    z = sigmoide(x)
    y = z * (1-z)
    return y

def derivIdentity(x):
    y = np.zeros(x.size,x.size)
    return y