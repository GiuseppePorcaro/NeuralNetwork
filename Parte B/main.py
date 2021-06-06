import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import time
import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle

import rete as r
import utility as u
import learningPhase as l
import funzioniMnist as fm
import backPropagation as b
import regolaAggiornamento as ra
import funzioniAttivazioneErrore as f

def main():

    #Iper-parametri fissati del modello
    M = 50
    plot = 0
    test = 0
    etaPos = 1.2
    etaNeg = 0.5
    epocheMax = 200
    numClasses = 10
    numFeatures = 28*28

    #Recupero e stampa dimensioni del dataset mnist
    print(">Caricamento dataset...\n",end='',flush=True)
    time.sleep(0.2)

    (X, labels), (testX,testT) = mnist.load_data()
    tX = testX
    tT = testT
    #Plot delle immagini
    if plot == 1:
        u.plotImmagini(X, labels)

    #Pre-processing------------------------------------------------------------------------------
    # Converto in np array
    X, testX = np.array(X, np.float32), np.array(testX, np.float32)

    # Vettorizzo le immagini in un vettore di 784 feature
    X, testX = X.reshape([-1, numFeatures]), testX.reshape([-1, numFeatures])

    # Normalizzo i valori immagine da [0, 255] a [0, 1]
    X, testX = X / 255., testX / 255.

    #Shuffle
    X , labels = shuffle(X, labels)
    testX, testT = shuffle(testX,testT)

    #Ottengo le classi dalle etichette
    T = fm.getTargetsFromLabels(labels)


    #Divisione del dataset in train-val-test (manca il test)    
    trainX = X[0:600] #(599,784)
    trainX = np.transpose(trainX) #(784,599)
    trainT = T[:,0:600] #(10, 599)
    trainT = np.array(trainT)

    valX = X[601:700] #(199,784)
    valX = np.transpose(valX) #(784,199)
    valT = T[:,601:700] #(10,199)
    valT = np.array(valT)
    #-------------------------------------------------------------------------------------------

    u.infoShapes(X, labels, T, trainX, trainT, valX, valT)

    print("Caricamento completo!\n")

    #creazione rete e avvio learning
    rete = r.nuovaRete(len(trainX),M,len(trainT),f.sigmoide,f.sigmoide) 
    r.infoRete(rete)

    #Fase di learning - DA IMPLEMENTARE TRAMITE TECNICA K-FOLD CROSS-VALIDATION
    [rete,err,errVal] = l.learningPhase(rete, epocheMax, etaPos, etaNeg, trainX, trainT, valX, valT, f.derivSigmoide, f.derivSigmoide, f.derivCrossEntropy,ra.RPROP)

    

main()