import numpy as np

def backPropagation(rete,x,t,derivFunHidden, derivFunOutupt, derivFunErr):
    #Passo di forward step
    [y,a1,z1,a2] = forwardStep(rete,x)

    #Passo di calcolo derivate
    


def forwardStep(rete,x):

    #ATTENZIONE IL PASSO FORWARD COSì SCRITTO ANDREBBE BENE SE LA RETE FOSSE SHALLOW
    #MA NEL CASO DI PIù STRATI INTERNI DEVO VEDERE COME FARE LA BACKPROPAGATION
    #PER TUTTI GLI ALTRI Y,A1,A2,Z1, DATO CHE ORA OTTERREI I DELTA UTILI SOLO PER
    #GLI ULTIMI DUE STRATI DELLA RETE

    if rete.nStrati == 1:
        a1 = np.dot(rete.W1,x) + rete.b1
        z1 = rete.f[0](a1)
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput
        y = rete.g(a2)
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
    a1 = np.dot(rete.W1,x) + rete.b1
    z1 = rete.f[0](a1)
    a2 = np.dot(rete.WOutput,z1) + rete.bOutput
    y = rete.g(a2)
    return y