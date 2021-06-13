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
    testT = fm.getTargetsFromLabels(testT)


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

    #creazione rete e avvio learning------------------------------------------------------------
    arrayFa = [f.sigmoide] #Array di funzioni attivazione
    arrayNumNeuroni = [50,len(trainT)] #Array contenente per ciascun layer il proprio numero di neuroni

    #Controllo di poter creare la rete
    if not(utility.checkCreazioneRete(numLayers, arrayFa)) or not(utility.checkCreazioneRete(numLayers, arrayNumNeuroni[0:len(arrayNumNeuroni)-1])):
        print("Non è possibile costruire tale modello di rete:")
        print(">La dimensione di uno degli array (Funzione attivazioni o numNeuroni) non è corretta!")
        return 0

    rete = r.nuovaRete(len(trainX),numLayers,arrayNumNeuroni,arrayFa,f.sigmoide) 
    r.infoRete(rete)

    #Fase di learning
    print("\n\n>Inizio fase di learning:\n-Numero epoche:\t",epocheMax,flush=True)
    time.sleep(0.1)
    [rete,err,errVal] = l.learningPhase(rete,epocheMax,trainX,trainT,valX,valT,batch,eta,f.derivSigmoide,f.derivSigmoide,f.derivCrossEntropy,ra.discesaDelGradiente)

    #Fare plot degli errori
    utility.plotErrori(err,errVal)

    #Validazione modello
    print("\n\nValidazione del modello su test di 100 coppie:")
    testX = np.transpose(testX[0:100])
    testT = testT[:,0:100]
    sommaErroreTest = 0
    for k in range(1,11):
        yTest = bck.simulaRete(rete,testX)
        erroreTest = f.crossEntropy(yTest,testT)
        sommaErroreTest = sommaErroreTest + erroreTest
    erroreTest = sommaErroreTest / k
    print(">Errore test: ",erroreTest)

    ##########A QUANTO PARE I VALORI OUTPUT NON SONO QUELLI CORRETTI
    ######CONTROLLARE ANCHE DAL CODICE MATLAB DEL PROFESSORE SE LE Y CORRISPONDONO ALLE ETICHETTE
    yTest = bck.simulaRete(rete,testX)
    yTest = f.softmax(yTest)
    print("\n\n y[:,5]: ", yTest[:,5])
    print("t[5] ", testT[:,5])
    print("label output: ",fromOutputToLabel(yTest[:,5]))
    

def fromOutputToLabel(y):
    i = 0
    max = -1
    for k in range(0,10):
        if y[k] > max:
            max = y[k]
            i = k
    return i

main()