print("Caricamento librerie...", end='', flush=True)
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import rete as r
import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt


import funzioniAttivazioneErrore as f
import regoleAggiornamento as ra
import learningPhase as l
import backPropagation as bck
import funzioniMnist as fm
import utility
print("Fatto!\n")

def main():

    eta = 0.1
    plot = 0
    test = 0
    batch = 1
    numLayers = 1
    epocheMax = 500
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
    testT = fm.getTargetsFromLabels(testT)


    #Divisione del dataset in train-val-test (manca il test)    
    trainX = X[0:5000] #(599,784)
    trainX = np.transpose(trainX) #(784,599)
    trainT = T[:,0:5000] #(10, 599)
    trainT = np.array(trainT)

    valX = X[5001:5600] #(199,784)
    valX = np.transpose(valX) #(784,199)
    valT = T[:,5001:5600] #(10,199)
    valT = np.array(valT)


    #Controllo che almeno le prime 20 immagini del train set siano corrette con le rispettive label 
    #utility.checkImmaginiTrain(trainX, trainT,len(trainX[1]))
    #-------------------------------------------------------------------------------------------

    utility.infoShapes(X, labels, T, trainX, trainT, valX, valT)

    print("Caricamento completo!\n")

    #creazione rete e avvio learning------------------------------------------------------------
    arrayFa = [f.sigmoide] #Array di funzioni attivazione
    arrayNumNeuroni = [70,len(trainT)] #Array contenente per ciascun layer il proprio numero di neuroni

    #Controllo di poter creare la rete
    if not(utility.checkCreazioneRete(numLayers, arrayFa)) or not(utility.checkCreazioneRete(numLayers, arrayNumNeuroni[0:len(arrayNumNeuroni)-1])):
        print("Non è possibile costruire tale modello di rete:")
        print(">La dimensione di uno degli array (Funzione attivazioni o numNeuroni) non è corretta!")
        return 0

    rete = r.nuovaRete(len(trainX),numLayers,arrayNumNeuroni,arrayFa,f.softmax) 
    r.infoRete(rete)

    #Fase di learning
    print("\n\n>Inizio fase di learning:\n-Numero epoche:\t",epocheMax,flush=True)
    time.sleep(0.1)
    [rete,err,errVal] = l.learningPhase(rete,epocheMax,trainX,trainT,valX,valT,batch,eta,f.derivSigmoide,f.derivSigmoide,f.crossEntropySoftmax,ra.discesaDelGradiente,1)

    #Fare plot degli errori
    utility.plotErrori(err,errVal)

    
    #Validazione modello-----------------------------------------------------------------------------
    #Valutazione errore su test-set
    testX = np.transpose(testX[0:1000])
    testT = testT[:,0:1000]
    print("\n\nValidazione del modello su test-set di ",len(testX)," coppie, eseguito 10 volte:")
    sommaErroreTest = 0
    for k in range(1,11):
        yTest = bck.simulaRete(rete,testX)
        erroreTest = f.crossEntropy(yTest,testT)
        sommaErroreTest = sommaErroreTest + erroreTest
    erroreTest = sommaErroreTest / k
    print(">Errore test: ",erroreTest)

    #Valutazione precisione del test-set. Molto alta con uso di softmax
    yTest = bck.simulaRete(rete,testX)
    numCorrette = 0
    for i in range(0,len(yTest[1])):
        yTemp = utility.fromOutputToLabel(yTest[:,i])
        labelCheck = utility.fromOutputToLabel(testT[:,i])
        print("Coppia [",i,"]: y = ",yTemp," --- label = ",labelCheck)
        if yTemp == labelCheck:
            numCorrette = numCorrette + 1
    
    perc = (numCorrette/len(yTest[1])) * 100
    print("Percentuale di risposte corrette(",numCorrette,") della rete sul test set: ", perc,"%")
    #------------------------------------------------------------------------------------------------


main()