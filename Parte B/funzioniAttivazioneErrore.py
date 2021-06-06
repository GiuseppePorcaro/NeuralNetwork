import numpy as np

def sigmoide(x): #ok
    y = (1/(1 + np.exp(-x)))
    return y

def identity(x): #ok
    y = x
    return y

def sumOfSquares(y,t):
    e = (1/2) * (np.power(y-t,2).sum())
    return e

def crossEntropy(Y,T): #ok
    e=-(T * np.log(Y)).sum()
    return e

def crossEntropySoftmax(y,t):
    z = np.exp(y)/np.exp(y).sum() #softmax
    e = z - t
    return e

def derivCrossEntropy(Y,T):
    e = -(T / Y).sum()
    return e

def derivSumOfSquares(y,t): #ok
    e = y-t
    return e

def derivSigmoide(x): #ok
    z = sigmoide(x)
    y = z * (1-z)
    return y

def derivIdentity(x):
    y = np.zeros(len(x[0]),len(x[0]))
    return y