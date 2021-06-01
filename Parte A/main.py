import rete as r
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import time

import funzioniAttivazione as fa
import learningPhase as l
import backPropagation as bck
import funzioniMnist as fm

def main():

    M = 50
    eta = 0.0005
    epocheMax = 200
    plot = 0
    numClasses = 10
    numFeatures = 28*28
    test = 0

    #Recupero e stampa dimensioni del dataset mnist
    print(">Caricamento dataset...\n",end='',flush=True)
    time.sleep(0.2)

    (X, labels), (testX,testT) = mnist.load_data()
    tX = testX
    tT = testT
    #Plot delle immagini
    if plot == 1:
        plotImmagini(X, labels)

    #Preprocessing------------------------------------------------------------------------------
    # Converto in np array
    X, testX = np.array(X, np.float32), np.array(testX, np.float32)

    # Vettorizzo le immagini in un vettore di 784 feature
    X, testX = X.reshape([-1, numFeatures]), testX.reshape([-1, numFeatures])

    # Normalizzo i valori immaginde da [0, 255] a [0, 1]
    X, testX = X / 255., testX / 255. 
    T = fm.getTargetsFromLabels(labels)

    #Divisione del dataset in train-val-test (manca il test)    
    trainX = X[1:600] #(599,784)
    trainX = np.transpose(trainX) #(784,599)
    trainT = T[:,1:600] #(10, 599)
    trainT = np.array(trainT)

    valX = X[601:1200] #(199,784)
    valX = np.transpose(valX) #(784,199)
    valT = T[:,601:1200] #(10,199)
    valT = np.array(valT)
    #-------------------------------------------------------------------------------------------

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

    if test == 1:
#############################Controllare il valore di y dato su matlab per verificare che sia corretto dal prof
        kX = testX[0:10]
        kX = np.transpose(kX)
        tT = testT[0:10]
        yTest = bck.simulaRete(rete,kX)
        plotImmagini(tX,tT)
        print("\n\nyTest: ",yTest)



def plotImmagini(trainX, trainT):
    num = 10
    images = trainX[:num]
    labels = trainT[:num]

    num_row = 2
    num_col = 5
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()


main()