import numpy as np

def backPropagation(rete,x,t,derivFunHidden, derivFunOutupt, derivFunErr):

    if rete.nStrati == 1:
        dhInput = 0 #Siccome non è usato quando la rete ha un singolo layer, lo si inizializza per evitare che sia undefined
        [y,a1,z1,a2] = forwardStep(rete,x) #y:(c,1), a1:(m,1), z1:(m,1), a2:(c,1)

        #Passo di calcolo derivate
        [derivWhidden,derivWOut,derivBiasHidden,derivBiasOut, dh] = calcoloDerivate(y,a1,z1,a2,derivFunHidden, derivFunOutupt, derivFunErr,x,t,rete)
    else:

        derivWhidden = np.zeros((rete.nStrati,rete.m,rete.m))
        derivBiasHidden = np.zeros((rete.nStrati,len(x[0]),))

        [y,a1,z1,a2] = forwardStep(rete,x)

        k = rete.nStrati-1

        [derivWhidden[k,:,:], derivWOut, derivBiasHidden[k,:], derivBiasOut,dh] = calcoloDerivate(y,a1[k,:,:],z1[k,:,:],a2,derivFunHidden, derivFunOutupt, derivFunErr,z1[k-1,:,:],t,rete)
        k = k-1
        for i in range(k,0,-1):

            dh = np.dot(np.transpose(rete.W[i,:,:]),dh)
            dh = dh * derivFunHidden(a1[i,:,:])

            derivWhidden[i,:,:] = np.dot(dh,np.transpose(z1[i-1]))
            derivBiasHidden[i,:] = sum(dh)

        
        dh = np.dot(np.transpose(rete.W[0,:,:]),dh)
        dh = dh * derivFunHidden(a1[0,:,:])

        dhInput = np.dot(dh,np.transpose(x))
        derivBiasHidden[0,:] = sum(dh)

    
    return [derivWhidden,derivWOut,derivBiasHidden,derivBiasOut,dhInput]

def calcoloDerivate(y,a1,z1,a2,derivFunHidden, derivFunOutupt, derivFunErr,x,t,rete):

    deltaOut = derivFunOutupt(a2) #(c,N)
    deltaOut = deltaOut * derivFunErr(y,t) #(c,N)x(N,) = (c,N)

    deltaHidden = np.dot(np.transpose(rete.WOutput),deltaOut) #(m,c)x(c,N) = (m,N)
    deltaHidden = deltaHidden * derivFunHidden(a1) #(m,N)x(m,N) = (m,N)

    derivWOut = np.dot(deltaOut,np.transpose(z1)) #(c,N)x(N,m) = (c,m)
    derivWhidden = np.dot(deltaHidden,np.transpose(x)) #(m,N)x(N,d) = (m,d)

    derivBiasOut = sum(deltaOut)
    derivBiasHidden = sum(deltaHidden)

    return (derivWhidden,derivWOut,derivBiasHidden,derivBiasOut,deltaHidden)

def forwardStep(rete,x):

    #ATTENZIONE IL PASSO FORWARD COSì SCRITTO ANDREBBE BENE SE LA RETE FOSSE SHALLOW
    #MA NEL CASO DI PIù STRATI INTERNI DEVO VEDERE COME FARE LA BACKPROPAGATION
    #PER TUTTI GLI ALTRI Y,A1,A2,Z1, DATO CHE ORA OTTERREI I DELTA UTILI SOLO PER
    #GLI ULTIMI DUE STRATI DELLA RETE

    #Ricontrollare la forward step a più strati

    if rete.nStrati == 1:
        a1 = np.dot(rete.W1,x) + rete.b1 #(m,d)x(d,) + (m,1) = (m,) + (m,1) = (m,)
        z1 = rete.f[0](a1) #(m,)
        a2 = np.dot(rete.WOutput,z1) + rete.bOutput #(c,m)x(m,) + (c,1) = (c,) + (c,1) = (c,)
        y = rete.g(a2) #(c,)
    else:
        a1 = np.zeros((rete.nStrati,rete.m,len(x[0])))
        z1 = np.zeros((rete.nStrati,rete.m,len(x[0])))

        a1[0,:,:] = np.dot(rete.W1,x) + rete.b1
        z1[0,:,:] = rete.f[0](a1[0,:,:])
        k = 1
        for i in range(1,rete.nStrati):
            a1[i,:,:] = np.dot(rete.W[i-1,:,:],z1[i-1,:,:]) + rete.b[i-1,:,:]
            z1[i,:,:] = rete.f[i-1](a1[i,:,:])
            k = k+1
        a2 = np.dot(rete.WOutput,z1[k-1,:,:]) + rete.bOutput
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
