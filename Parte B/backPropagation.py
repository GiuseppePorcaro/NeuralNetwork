import numpy as np

def backPropagation(rete,x,t,derivFunHidden, derivFunOutupt, derivFunErr, softmax):

    #Passo di forwardStep
    [y,a1,z1,a2] = forwardStep(rete,x)

    if softmax == 0:
        #Passo di calcolo derivate
        deltaOut = derivFunOutupt(a2) 
        deltaOut = deltaOut * derivFunErr(y,t)
    else:
        deltaOut = derivFunErr(y,t)

    deltaHidden = np.dot(np.transpose(rete.WOutput),deltaOut)
    deltaHidden = deltaHidden * derivFunHidden(a1) 

    derivWOut = np.dot(deltaOut,np.transpose(z1)) 
    derivWhidden = np.dot(deltaHidden,np.transpose(x)) 

    derivBiasOut = np.sum(deltaOut, axis=1)
    derivBiasHidden = np.sum(deltaHidden, axis=1)
    derivBiasHidden = derivBiasHidden.reshape(len(derivBiasHidden),1)
    derivBiasOut = derivBiasOut.reshape(len(derivBiasOut),1)

    return [derivWhidden, derivWOut, derivBiasHidden, derivBiasOut]

def forwardStep(rete,x):

    a1 = np.dot(rete.W1,x) + rete.b1 
    z1 = rete.f(a1) 
    a2 = np.dot(rete.WOutput,z1) + rete.bOutput
    y = rete.g(a2) 

    return (y,a1,z1,a2)

def simulaRete(rete,x):
    
    a1 = np.dot(rete.W1,x) + rete.b1 
    z1 = rete.f(a1) 
    a2 = np.dot(rete.WOutput,z1) + rete.bOutput
    y = rete.g(a2) 

    return y
