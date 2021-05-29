import numpy as np

def backPropagation(rete,x,t,derivFunHidden, derivFunOutupt, derivFunErr):
    #Per adesso la rete effettua la back propagation bene solo per rete shallow

    #Passo di forward step
    [y,a1,z1,a2] = forwardStep(rete,x)

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

    #ATTENZIONE IL PASSO FORWARD COSì SCRITTO ANDREBBE BENE SE LA RETE FOSSE SHALLOW
    #MA NEL CASO DI PIù STRATI INTERNI DEVO VEDERE COME FARE LA BACKPROPAGATION
    #PER TUTTI GLI ALTRI Y,A1,A2,Z1, DATO CHE ORA OTTERREI I DELTA UTILI SOLO PER
    #GLI ULTIMI DUE STRATI DELLA RETE

    #Magari si potrebbe cambiare andando a 

    if rete.nStrati == 1:
        a1 = np.dot(rete.W1,x) + rete.b1 #(m,d)x(d,N) + (m,1) = (m,N) + (m,1) = (m,N)
        z1 = rete.f[0](a1) #(m,N)
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput #(c,m)x(m,N) + (c,1) = (c,N) + (c,1) = (c,N)
        y = rete.g(a2) #(c,N)
    else:
        a1 = np.dot(rete.W1,x) + rete.b1
        z1 = rete.f[0](a1)
        for i in range(0,rete.nStrati-1):
            a1 = np.dot(rete.W[i,:,:],z1) + rete.b[i,:,:]
            z1 = rete.f[i](a1)
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput
        y = rete.g(a2)

    return (y,a1,z1,a2)

def simulaRete(rete,x):
    if rete.nStrati == 1:
        a1 = np.dot(rete.W1,x) + rete.b1 #(m,d)x(d,N) + (m,1) = (m,N) + (m,1) = (m,N)
        z1 = rete.f[0](a1) #(m,N)
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput #(c,m)x(m,N) + (c,1) = (c,N) + (c,1) = (c,N)
        y = rete.g(a2) #(c,N)
    else:
        a1 = np.dot(rete.W1,x) + rete.b1
        z1 = rete.f[0](a1)
        for i in range(0,rete.nStrati-1):
            a1 = np.dot(rete.W[i,:,:],z1) + rete.b[i,:,:]
            z1 = rete.f[i](a1)
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput
        y = rete.g(a2)
    return y