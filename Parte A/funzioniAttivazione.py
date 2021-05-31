import numpy as np

def discesaDelGradiente(rete,eta,derivHidden,derivOut,derivBiasHidden,derivBiasOut):
    rete.W1 = rete.W1 - eta * derivHidden #(m,d) - val * (m,d)
    rete.WOutput = rete.WOutput - eta * derivOut
    rete.b1 = rete.b1 - eta * derivBiasHidden
    rete.bOut = rete.bOutput - eta * derivBiasOut

    return rete

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