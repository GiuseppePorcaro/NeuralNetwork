import numpy as np
class Rete:
    pass

def nuovaRete(numeroInput, numeroNodiHidden, numeroNodiOutput,funAttivazioneHidden, funAttivazioneOutput):
    rete = Rete()
    rete.nome = "Rete Neurale MultiStrato"

    #Generazione casuale degli iper parametri della rete.
    SIGMA = 0.1
    rete.W1 = SIGMA * np.random.randn(numeroNodiHidden,numeroInput) #(m,d)
    rete.b1 = SIGMA * np.random.randn(numeroNodiHidden,1) #(m,1)
    rete.WOutput = SIGMA * np.random.randn(numeroNodiOutput,numeroNodiHidden) #(c,m)
    rete.bOutput = SIGMA * np.random.randn(numeroNodiOutput,1) #(c,1)

    #Ulteriore step di aggiornamento per i pesi
    rete.DELTAW1 = SIGMA * np.full((numeroNodiHidden,numeroInput),0.1)
    rete.DELTAB1 = SIGMA * np.full((numeroNodiHidden,1), 0.1)
    rete.DELTAWOutput = SIGMA * np.full((numeroNodiOutput,numeroNodiHidden),0.1)
    rete.DELTABOutput = SIGMA * np.full((numeroNodiOutput,1),0.1)       

    #Dettagli della rete
    rete.d = numeroInput
    rete.m = numeroNodiHidden
    rete.c = numeroNodiOutput
    
    #Funzioni di attivazione
    rete.f = funAttivazioneHidden #array di funzioni di attivazione per i strati hidden
    rete.g = funAttivazioneOutput
    
    return rete

def infoRete(rete):
    ##Stampa le informazioni della rete come il  numero di strati, di elementi input, numero nodi interni, ecc...
    print("Informazioni sulla rete")

    print("W1: ", rete.W1.shape)
    print("b1: ", rete.b1.shape)
    print("WOutput: ", rete.WOutput.shape)
    print("bOutput: ", rete.bOutput.shape)
    print("DELTAHidden: ",rete.DELTAW1.shape)
    print("DELTABHidden: ",rete.DELTAB1.shape)
    print("DELTAOutput: ",rete.DELTAWOutput.shape)
    print("DELTABOutput: ",rete.DELTABOutput.shape)

def stampaIperParametri(rete):
    ##Stampa tutti i pesi e bias della rete data in input

    print("Peso primo strato interno:\n",rete.W1,"\n")
    print("Bias primo strato interno:\n",rete.b1,"\n\nPesi strati hidden")
    print(rete.WOutput,"\n\nBias Strat Outuput")
    print(rete.bOutput,"\n")
