import rete as r
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import time

import funzioniAttivazione as fa
import learningPhase as l
import backPropagation as bck
import funzioniMnist as fm

#TO DO: f.crossEntropy(y,t) tira una eccezione sulla funzione sigmoide e non so perchè.
#Bisogna risolvere questo problema.
#Forse il problema sta davvero nella funzione sigmoide dato che tira un "RuntimeWarning: overflow encountered in exp",
#ma lo fa solo alla epoca 0


def main():

    M = 50
    eta = 0.0005
    epocheMax = 200

    #Recupero e stampa dimensioni del dataset mnist
    print(">Caricamento dataset...\n",end='',flush=True)
    time.sleep(0.2)
    X = fm.loadMnistImages("mnist/train-images-idx3-ubyte.gz")
    labels = fm.loadMnistLabels("mnist/train-labels-idx1-ubyte.gz")
    T = fm.getTargetsFromLabels(labels)
    #Divisione del dataset in train-val-test (manca il test)    
    trainX = X[1:600] #(599,784)
    trainX = np.transpose(trainX) #(784,599)
    trainX = np.array(trainX)
    trainT = T[:,1:600] #(10, 599)
    trainT = np.array(trainT)

    valX = X[601:1200] #(199,784)
    valX = np.transpose(valX) #(784,199)
    valX = np.array(valX)
    valT = T[:,601:1200] #(10,199)
    valT = np.array(valT)

    print("X:\t",X.shape)
    print("labels:\t",labels.shape)
    print("T:\t",T.shape)
    print("TrainX:\t",trainX.shape)
    print("TrainT:\t",trainT.shape)
    print("ValX:\t",valX.shape)
    print("ValT:\t",valT.shape)

    print("Caricamento completo!\n")
    #creazione rete e avvio learning
    arrayFa = [fa.sigmoide,fa.sigmoide,fa.sigmoide,fa.sigmoide,fa.sigmoide]
    rete = r.nuovaRete(len(trainX),1,M,len(trainT),arrayFa,fa.sigmoide) 
    r.infoRete(rete)

    #Fase di learning
    print("\n\n>Inizio fase di learning:\n-Numero epoche:\t",epocheMax,"\n-Eta:\t",eta,end='\n',flush=True)
    time.sleep(0.1)
    [rete,err,errVal] = l.learningPhase(rete,epocheMax,trainX,trainT,valX,valT,0,eta,fa.derivSigmoide,fa.derivSigmoide,fa.derivCrossEntropy,fa.discesaDelGradiente)

    #Stampa errore e errore valutazione della rete
    print("Errore:\n",err)
    print("\n\nErrore valutazione:\n",errVal)

    #plt.plot(err)plt.show()plt.plot(errVal)plt.show()



main()