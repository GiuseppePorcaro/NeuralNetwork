import numpy as np

def backPropagation(rete,x,t,derivFunHidden, derivFunOutupt, derivFunErr):

    #Passo di forwardStep
    [y,a1,z1,a2] = forwardStep(rete,x) #y:(c,1), a1:(m,1), z1:(m,1), a2:(c,1)

    #Passo di calcolo derivate
    deltaOut = derivFunOutupt(a2) #(c,N)
    deltaOut = deltaOut * derivFunErr(y,t) #(c,N)x(N,) = (c,N)

    deltaHidden = np.dot(np.transpose(rete.WOutput),deltaOut) #(m,c)x(c,N) = (m,N)
    deltaHidden = deltaHidden * derivFunHidden(a1) #(m,N)x(m,N) = (m,N)

    derivWOut = np.dot(deltaOut,np.transpose(z1)) #(c,N)x(N,m) = (c,m)
    derivWhidden = np.dot(deltaHidden,np.transpose(x)) #(m,N)x(N,d) = (m,d)

    derivBiasOut = sum(deltaOut)
    derivBiasHidden = sum(deltaHidden)

    return [derivWhidden,derivWOut,derivBiasHidden,derivBiasOut]

def forwardStep(rete,x):

    a1 = np.dot(rete.W1,x) + rete.b1 #(m,d)x(d,) + (m,1) = (m,) + (m,1) = (m,)
    z1 = rete.f(a1) #(m,)
    a2 = np.dot(rete.WOutput,z1) + rete.bOutput #(c,m)x(m,) + (c,1) = (c,) + (c,1) = (c,)
    y = rete.g(a2) #(c,)

    return (y,a1,z1,a2)

def simulaRete(rete,x):
    
    a1 = np.dot(rete.W1,x) + rete.b1 #(m,d)x(d,N) + (m,1) = (m,N) + (m,1) = (m,N)
    z1 = rete.f(a1) #(m,N)
    a2 = np.dot(rete.WOutput,z1) + rete.bOutput #(c,m)x(m,N) + (c,1) = (c,N) + (c,1) = (c,N)
    y = rete.g(a2) #(c,N)

    return y
