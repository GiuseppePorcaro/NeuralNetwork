print("Caricamento librerie...", end='', flush=True)
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import time
import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import rete as r
import utility as u
import learningPhase as l
import funzioniMnist as fm
import backPropagation as b
import regolaAggiornamento as ra
import funzioniAttivazioneErrore as f
print("Fatto!\n")

def main():

    #Iper-parametri fissati del modello
    M = 25
    k = 10
    plot = 0
    test = 0
    etaPos = 1.002
    etaNeg = 0.005
    epocheMax = 200
    numClasses = 10
    numFeatures = 28*28

    #Recupero e stampa dimensioni del dataset mnist
    print(">Caricamento dataset...\n",end='',flush=True)
    time.sleep(0.2)

    (X, labels), (testX,testT) = mnist.load_data()
    #Bisogna controllare che immagini ed etichette siano correttamente impostate
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

    #Divisione del dataset. QUI È DA METTERE SOLO IL DATASET X E DA QUESTO TIRARE FUORI TRAIN E TEST PER IL KFOLD   
    trainX = X[0:5000]
    trainX = np.transpose(trainX)
    trainT = T[:,0:5000]
    trainT = np.array(trainT)

    valX = X[601:1300]
    valX = np.transpose(valX)
    valT = T[:,601:1300]
    valT = np.array(valT)
    #-------------------------------------------------------------------------------------------

    u.infoShapes(X, labels, T, trainX, trainT, valX, valT)

    print("Caricamento completo!\n")
    
    print(">Algoritmo calcolo derivate:\t\tBackpropagation")
    print(">Algoritmo aggiornamento pesi e bias:\tRPROP\n")
    u.stampaInfoModelloRete("Shallow Network", M, len(trainT), "Sigmoide", "Sigmoide", "Cross Entropy")

    #Fase di scelta del modello della rete
    print("\nFase di valutazione del modello:",flush=True)
    kf = KFold(n_splits=k)
    print(">Numero di fold:\t\t\t",kf.get_n_splits(trainX),flush=True)
    
    valutazioneErrore = np.zeros((10,1))
    valutazionePrecisione = np.zeros((10,1))
    k = 0
    for train_index, test_index in kf.split(trainX):
        print(">Fold: ",k)
        sommaErrore = 0
        #Creazione rete per i k fold
        rete = r.nuovaRete(len(trainX),M,len(trainT),f.sigmoide,f.softmax) 

        #Recupero dei k-fold
        train_x, test_x = trainX[:,train_index] , trainX[:,test_index]
        train_t , test_t = trainT[:,train_index] , trainT[:,test_index]

        #train_x , test_x = np.transpose(train_x) , np.transpose(test_x)

        #Prova di learning del modello di rete
        [rete,err,errTest] = l.learningPhase(rete, epocheMax, etaPos, etaNeg, train_x, train_t, test_x, test_t, f.derivSigmoide, f.derivSigmoide, f.crossEntropySoftmax, ra.RPROP)

        #Calcolo media errore sul test-set
        sommaErrore = errTest.sum()
        valutazioneErrore[k][0] = sommaErrore / len(valutazioneErrore)

        #Calcolo precisione sul test-set
        yTest = b.simulaRete(rete,test_x)
        numCorrette = 0
        for i in range(0,len(yTest[1])):
            yTemp = u.fromOutputToLabel(yTest[:,i])
            labelCheck = u.fromOutputToLabel(test_t[:,i])
            #print("Coppia [",i,"]: y = ",yTemp," --- label = ",labelCheck)
            if yTemp == labelCheck:
                numCorrette = numCorrette + 1
                
        perc = (numCorrette/len(yTest[1])) * 100
        print(">Precisione test:\t", perc,"%")
        valutazionePrecisione[k][0] = perc
        k = k + 1
        


    mediaErrore = valutazioneErrore.sum() / len(valutazioneErrore)
    print("Valutazione terminata!\n\nRisultati del modello valutato:")
    print("Media delle valutazioni errore:\t\t",mediaErrore)
    #u.plotErrore(valutazioneErrore)

    mediaPrecisione = valutazionePrecisione.sum() / len(valutazionePrecisione)
    print("Media della precisione:\t\t",mediaPrecisione)
            

main()