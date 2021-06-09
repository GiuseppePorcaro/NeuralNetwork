import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import rete as r
import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle
import time

import funzioniAttivazioneErrore as f
import regoleAggiornamento as ra
import learningPhase as l
import backPropagation as bck
import funzioniMnist as fm
import utility

def main():

    M = 50
    eta = 0.00001
    plot = 0
    test = 0
    batch = 1
    numLayers = 1
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
        utility.plotImmagini(X, labels)

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

    valX = X[601:1201] #(199,784)
    valX = np.transpose(valX) #(784,199)
    valT = T[:,601:1201] #(10,199)
    valT = np.array(valT)
    #-------------------------------------------------------------------------------------------

    utility.infoShapes(X, labels, T, trainX, trainT, valX, valT)

    print("Caricamento completo!\n")

    #creazione rete e avvio learning
    arrayFa = [f.sigmoide,f.sigmoide,f.sigmoide,f.sigmoide,f.sigmoide]
    rete = r.nuovaRete(len(trainX),numLayers,M,len(trainT),arrayFa,f.sigmoide) 
    r.infoRete(rete)

    #Fase di learning
    print("\n\n>Inizio fase di learning:\n-Numero epoche:\t",epocheMax,"\n-Eta:\t",eta,end='\n',flush=True)
    time.sleep(0.1)
    [rete,err,errVal] = l.learningPhase(rete,epocheMax,trainX,trainT,valX,valT,batch,eta,f.derivSigmoide,f.derivSigmoide,f.crossEntropySoftmax,ra.discesaDelGradiente)

    #Fare plot degli errori
    utility.plotErrori(err,errVal)

    #Validazione modello
    '''
    print("\n\nValidazione del modello su test di 100 coppie:")
    testX = np.transpose(testX[1:100])
    testT = testT[1:100]
    sommaErroreTest = 0
    for k in range(1,10):
        yTest = bck.simulaRete(rete,testX)
        erroreTest = fa.crossEntropy(yTest,testT)
        sommaErroreTest = sommaErroreTest + erroreTest
    erroreTest = sommaErroreTest / k
    print(">Errore test: ",erroreTest)
    '''

main()