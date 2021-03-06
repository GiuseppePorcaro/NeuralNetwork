import numpy as np

def backPropagation(rete,x,t, derivFunOutupt, derivFunErr,softmax):

    if rete.nStrati == 1:
        dhInput = 0 #Siccome non è usato quando la rete ha un singolo layer, lo si inizializza per evitare che sia undefined
        dbhInput = 0

        [y,a1,z1,a2] = forwardStep(rete,x)

        #Passo di calcolo derivate
        [derivWhidden,derivWOut,derivBiasHidden,derivBiasOut, dh] = calcoloDerivate(y,a1,z1,a2,rete.derivF[0], derivFunOutupt, derivFunErr,x,t,rete,softmax)
    else:
        #Array che memorizzaranno tutte le derivate degli strati interni
        derivWhidden = np.empty((1,rete.nStrati-1), dtype = object)
        derivBiasHidden = np.empty((1,rete.nStrati-1), dtype = object)

        [y,a1,z1,a2] = forwardStep(rete,x)

        k = rete.nStrati-2

        #Calcolo derivate tra layer output e ultimo layer hidden
        [derivWhidden[0][k], derivWOut, derivBiasHidden[0][k], derivBiasOut,dh] = calcoloDerivate(y,a1[0][k+1],z1[0][k+1],a2,rete.derivF[k], derivFunOutupt, derivFunErr,z1[0][k],t,rete,softmax)

        #Calcolo derivate tra pen-ultimo layer hidden e secondo layer hidden
        k = k-1
        for i in range(k,-1,-1):
            
            dh = np.dot(np.transpose(rete.W[0][i+1]),dh)
            dh = dh * rete.derivF[i](a1[0][i+1])

            derivWhidden[0][i] = np.dot(dh,np.transpose(z1[0][i]))
            derivBiasHidden[0][i] = dh.sum(axis=1)
            derivBiasHidden[0][i] = derivBiasHidden[0][i].reshape(len(derivBiasHidden[0][i]),1)

        #Calcolo derivata primo layer hidden
        dh = np.dot(np.transpose(rete.W[0][0]),dh)
        dh = dh * rete.derivF[0](a1[0][0])

        dhInput = np.dot(dh,np.transpose(x))
        dbhInput = dh.sum(axis=1)
        dbhInput = dbhInput.reshape(len(dbhInput),1)

    return [derivWhidden,derivWOut,derivBiasHidden,derivBiasOut,dhInput,dbhInput]

def calcoloDerivate(y,a1,z1,a2,derivFunHidden, derivFunOutupt, derivFunErr,x,t,rete,softmax):

    #Se si usa il softmax come funzione attivazione output si può unire il calcolo della derivata
    #alla derivata della funzione di errore crossEntropy. Se 1 si usa direttamente tale derivata
    #Se 0 allora si effettua il calcolo della backprop normalmente
    if softmax == 0:
        deltaOut = derivFunOutupt(a2)
        deltaOut = deltaOut * derivFunErr(y,t)
    else:
        deltaOut = derivFunErr(y,t)

    deltaHidden = np.dot(np.transpose(rete.WOutput),deltaOut) 
    deltaHidden = deltaHidden * derivFunHidden(a1)

    derivWOut = np.dot(deltaOut,np.transpose(z1)) 
    derivWhidden = np.dot(deltaHidden,np.transpose(x))

    derivBiasOut = deltaOut.sum(axis=1)
    derivBiasHidden = deltaHidden.sum(axis=1)
    derivBiasOut = derivBiasOut.reshape(len(derivBiasOut),1)
    derivBiasHidden = derivBiasHidden.reshape(len(derivBiasHidden),1)

    return (derivWhidden,derivWOut,derivBiasHidden,derivBiasOut,deltaHidden)

def forwardStep(rete,x):

    if rete.nStrati == 1:
        a1 = np.dot(rete.W1,x) + rete.b1 
        z1 = rete.f[0](a1) 
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput 
        y = rete.g(a2)
    else:
        #Inizializzazione array per memorizzare output e somme pesate da usare nella 
        a1 = np.empty((1,rete.nStrati), dtype = object)
        z1 = np.empty((1,rete.nStrati), dtype = object)

        #Calcolo e memorizzazione di tutti gli output e 'a' degli strati hidden
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
