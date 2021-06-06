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
    rete.bOutput = SIGMA * np.random.rand(numeroNodiOutput,1) #(c,1)

    #Ulteriore step di aggiornamento per i pesi
    rete.deltaHidden = SIGMA * np.random.randn(numeroNodiHidden,numeroInput) #(m,d)
    rete.deltaOutput = SIGMA * np.random.randn(numeroNodiHidden,numeroInput) #(m,d)        

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

    print("W1: [",rete.m,"x",rete.d,"]")
    print("b1: [",rete.m,"x",1,"]")
    print("WOutput: [",rete.c,"x",rete.m,"]")
    print("bOutput: [",rete.c,"x",1,"]")

def stampaIperParametri(rete):
    ##Stampa tutti i pesi e bias della rete data in input

    print("Peso primo strato interno:\n",rete.W1,"\n")
    print("Bias primo strato interno:\n",rete.b1,"\n\nPesi strati hidden")
    print(rete.WOutput,"\n\nBias Strat Outuput")
    print(rete.bOutput,"\n")
