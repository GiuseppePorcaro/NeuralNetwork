import numpy as np

def sigmoide(x):
    y = (1/(1+np.exp(-x)))
    return y

def identity(x):
    y = x
    return y