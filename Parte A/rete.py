import numpy as np
class Rete:
    pass

def nuovaRete(numeroInput, numeroStratiHidden, arrNumNeuroni, arrFunAttivazioneHidden, funAttivazioneOutput):
    rete = Rete()
    rete.nome = "Rete Neurale MultiStrato"

    #Generazione casuale degli iper parametri della rete. Per adesso dividiamo primo strato hidden dagli altri
    #a causa del diverso numero di dati in input
    SIGMA = 0.1
    rete.W1 = SIGMA * np.random.randn(arrNumNeuroni[0],numeroInput) 
    rete.b1 = SIGMA * np.random.randn(arrNumNeuroni[0],1) 

    #Inizializzo array di pesi e bias dei nodi interni per qualunque numero di nodi
    rete.W = np.empty((1,numeroStratiHidden-1), dtype = object)
    rete.b = np.empty((1,numeroStratiHidden-1), dtype = object)

    k = 1
    for i in range(0,numeroStratiHidden-1):
        rete.W[0][i] = SIGMA * np.random.randn(arrNumNeuroni[k],arrNumNeuroni[k-1]) 
        rete.b[0][i] = SIGMA * np.random.randn(arrNumNeuroni[k],1)
        k=k+1
    
    rete.WOutput = SIGMA * np.random.randn(arrNumNeuroni[k],arrNumNeuroni[k-1]) 
    rete.bOutput = SIGMA * np.random.randn(arrNumNeuroni[k],1) 

    #Dettagli della rete
    rete.nStrati = numeroStratiHidden
    rete.d = numeroInput
    rete.m = arrNumNeuroni[0:len(arrNumNeuroni)-2]
    rete.c = arrNumNeuroni[len(arrNumNeuroni)-1]
    
    #Funzioni di attivazione
    rete.f = arrFunAttivazioneHidden #array di funzioni di attivazione per i strati hidden
    rete.g = funAttivazioneOutput
    
    return rete

def infoRete(rete):
    ##Stampa le informazioni della rete come il  numero di strati, di elementi input, numero nodi interni, ecc...
    print("Informazioni sulla rete")

    print("Numero strati interni: ",rete.nStrati)
    print("W1:", rete.W1.shape)
    print("b1: ", rete.b1.shape,"\n")

    if rete.nStrati > 1:
        for i in range(0,rete.nStrati-1):
            print("Layer[",i+1,"]:")
            print("W: ", rete.W[0][i].shape)
            print("b: ", rete.b[0][i].shape,"\n")

    print("WOutput: ", rete.WOutput.shape)
    print("bOutput: ", rete.bOutput.shape)

def stampaIperParametri(rete):
    ##Stampa tutti i pesi e bias della rete data in input

    print("Peso primo strato interno:\n",rete.W1,"\n")
    print("Bias primo strato interno:\n",rete.b1,"\n\nPesi strati hidden")
    for i in range(0,rete.nStrati-1):
        print(rete.W[0][i],"\n\nBias Strati interni")
        print(rete.b[0][i],"\n\nPesi Strato Output")
    print(rete.WOutput,"\n\nBias Strat Outuput")
    print(rete.bOutput,"\n")
