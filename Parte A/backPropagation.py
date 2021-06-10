import numpy as np

def backPropagation(rete,x,t,derivFunHidden, derivFunOutupt, derivFunErr):

    if rete.nStrati == 1:
        dhInput = 0 #Siccome non è usato quando la rete ha un singolo layer, lo si inizializza per evitare che sia undefined
        dbhInput = 0

        [y,a1,z1,a2] = forwardStep(rete,x)

        #Passo di calcolo derivate
        [derivWhidden,derivWOut,derivBiasHidden,derivBiasOut, dh] = calcoloDerivate(y,a1,z1,a2,derivFunHidden, derivFunOutupt, derivFunErr,x,t,rete)
    else:

        derivWhidden = np.empty((1,rete.nStrati-1), dtype = object)
        derivBiasHidden = np.empty((1,rete.nStrati-1), dtype = object)

        [y,a1,z1,a2] = forwardStep(rete,x)

        k = rete.nStrati-2

        #Calcolo derivate tra layer output e ultimo layer hidden
        [derivWhidden[0][k], derivWOut, derivBiasHidden[0][k], derivBiasOut,dh] = calcoloDerivate(y,a1[0][k+1],z1[0][k+1],a2,derivFunHidden, derivFunOutupt, derivFunErr,z1[0][k],t,rete)

        #Calcolo derivate tra pen-ultimo layer hidden e secondo layer hidden
        k = k-1
        for i in range(k,-1,-1):
            
            dh = np.dot(np.transpose(rete.W[0][i+1]),dh)
            dh = dh * derivFunHidden(a1[0][i+1])

            derivWhidden[0][i] = np.dot(dh,np.transpose(z1[0][i]))
            derivBiasHidden[0][i] = dh.sum()

        #Calcolo derivata primo layer hidden
        dh = np.dot(np.transpose(rete.W[0][0]),dh)
        dh = dh * derivFunHidden(a1[0][0])

        dhInput = np.dot(dh,np.transpose(x))
        dbhInput = dh.sum()

    return [derivWhidden,derivWOut,derivBiasHidden,derivBiasOut,dhInput,dbhInput]

def calcoloDerivate(y,a1,z1,a2,derivFunHidden, derivFunOutupt, derivFunErr,x,t,rete):

    deltaOut = derivFunOutupt(a2) #(c,N)
    deltaOut = deltaOut * derivFunErr(y,t)

    deltaHidden = np.dot(np.transpose(rete.WOutput),deltaOut) 
    deltaHidden = deltaHidden * derivFunHidden(a1) 

    derivWOut = np.dot(deltaOut,np.transpose(z1)) 
    derivWhidden = np.dot(deltaHidden,np.transpose(x))

    derivBiasOut = deltaOut.sum()
    derivBiasHidden = deltaHidden.sum()

    return (derivWhidden,derivWOut,derivBiasHidden,derivBiasOut,deltaHidden)

def forwardStep(rete,x):

    if rete.nStrati == 1:
        a1 = np.dot(rete.W1,x) + rete.b1 
        z1 = rete.f[0](a1) 
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput 
        y = rete.g(a2) 
    else:

        a1 = np.empty((1,rete.nStrati), dtype = object)
        z1 = np.empty((1,rete.nStrati), dtype = object)

        a1[0][0] = np.dot(rete.W1,x) + rete.b1
        z1[0][0] = rete.f[0](a1[0][0])
        k = 1
        for i in range(1,rete.nStrati):
            a1[0][i] = np.dot(rete.W[0][i-1],z1[0][i-1]) + rete.b[0][i-1]
            z1[0][i] = rete.f[i-1](a1[0][i])
            k = k+1
        a2 = np.dot(rete.WOutput,z1[0][k-1]) + rete.bOutput
        y = rete.g(a2)

    return (y,a1,z1,a2)

def simulaRete(rete,x):

    if rete.nStrati == 1:
        a1 = np.dot(rete.W1,x) + rete.b1 
        z1 = rete.f[0](a1) 
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput 
        y = rete.g(a2)
    else:
        a1 = np.dot(rete.W1,x) + rete.b1
        z1 = rete.f[0](a1)
        k = 1
        for i in range(0,rete.nStrati-1):
            a1 = np.dot(rete.W[0][i],z1) + rete.b[0][i]
            z1 = rete.f[k](a1)
            k = k + 1
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput
        y = rete.g(a2)
    return y
