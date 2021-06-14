import numpy as np
import matplotlib.pyplot as plt

#####   Funzioni di utilità #####
def plotErrori(errore, erroreVal):
    plt.plot(np.transpose(errore), label = 'errore')
    plt.plot(np.transpose(erroreVal), label = 'errore Valutazione')
    plt.show()

def plotErrore(errore):
    plt.plot(np.transpose(errore), label = 'errore')
    plt.show()

def stampaInfoModelloRete(tipoRete, m, c, funAttHidden, funAttOutput,funErrore):
    print("Modello di rete: ")
    print(">Tipo di modello:\t\t\t",tipoRete)
    print(">Numero nodi hidden:\t\t\t",m)
    print(">Numero nodi output:\t\t\t",c)
    print(">Funzione attivazione layer hidden:\t",funAttHidden)
    print(">Funzione attivazione layer output:\t",funAttOutput)
    print(">Funzione errore:\t\t\t",funErrore)

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

def infoShapes(X, labels, T, trainX, trainT, valX, valT):
    print("X:\t",X.shape)
    print("labels:\t",labels.shape)
    print("T:\t",T.shape)
    print("TrainX:\t",trainX.shape)
    print("TrainT:\t",trainT.shape)
    print("ValX:\t",valX.shape)
    print("ValT:\t",valT.shape)

def checkCreazioneRete(numLayersHidden, arr):
    #Tale funzione controlla che la dimensione dell'array di funzioni attivazione
    #o numero di nodi hidden sia della dimensione giusta per creare la rete multistrato
    if numLayersHidden == len(arr):
        return True
    return False


