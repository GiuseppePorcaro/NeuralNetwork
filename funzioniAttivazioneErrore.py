import numpy as np

def sigmoide(x): 
    y = (1/(1 + np.exp(-x)))
    return y

def identity(x): 
    y = x
    return y

def tanh(x):
    return np.tanh(x)

def derivTanh(x):
    return 1-np.power(tanh(x),2)

def RELU(x):
    return x * (x > 0)

def derivRELU(x):
    return np.greater(x,0).astype(int)

def LRELU(x,beta):
    return beta * x * (x > 0)

def sumOfSquares(y,t): 
    e = (1/2) * (np.power(y-t,2).sum())
    return e

def crossEntropy(Y,T): 
    e=-(T * np.log(Y)).sum()
    return e

def softmax(x):
    e = np.exp(x)
    return e/np.sum(e,axis=0,keepdims=True)

def stableSoftmax(x):
    #Questo softmax corregge i problemi legati ai floating point
    s = np.exp(x-np.max(x))/np.sum(np.exp(x),axis=0,keepdims=True)
    return s

def derivSoftmax(x,t):
    return x - t

def crossEntropySoftmax(y,t):
    return (np.log(np.exp(y).sum())) - ((t*y).sum())

def derivCrossEntropySoftmax(y,t):
    #Se si usa softmax come funzione di attivazione per il layer output, è possibile, 
    #unire il calcolo della derivata della cross entropy con quello del softmax, il quale diventa y-t
    #dove y è l'output al quale è stato applicato il softmax
    e = y - t
    return e

def derivCrossEntropy(Y,T): 
    e = -(T / Y).sum(axis=1)
    e = e.reshape(len(e),1)
    return e

def derivSumOfSquares(y,t): 
    e = y-t
    return e

def derivSigmoide(x): 
    z = sigmoide(x)
    y = z * (1-z)
    return y

def derivIdentity(x):
    y = np.zeros(len(x[0]),len(x[0]))
    return y