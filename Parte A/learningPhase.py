import numpy as np
import backPropagation as b
import funzioniAttivazioneErrore as f

def learningPhase(rete, N, x, t, xVal, tVal, batch, eta, derivFunActOutput, derivFunErr,regolaAggiornamento,softmax,funErr):
    #Inizializzo le derivate altrimenti sono indefinite
    [derivW1,derivW2,derivBiasHidden,derivBiasOutput] = [0,0,0,0]

    #Inizializzazione array degli errori ed errore minimo
    err = np.ones((1,N))
    errVal = np.ones((1,N))
    yVal = b.simulaRete(rete,xVal)
    minErr = funErr(yVal,tVal)
    reteScelta = rete

    if batch == 1:
        eta = 0.00005
        if softmax == 1:
            eta = 0.005
    
    print("-Eta:\t\t",eta)

    print("\nEpoca\t\t Errore\t\t\t\t Errore Valutazione")
    print("---------------------------------------------------------------------------")
    for epoca in range(0,N):
        if batch == 0:
            #Learning di tipo on-line
            for n in range(0,len(x[0])):
                #Reshape perchè python considera diversi le dimensioni (x,) e (x,1), i quali comunque sono vettori
                xb = x[:,n].reshape((x.shape[0],1))
                tb = t[:,n].reshape((t.shape[0],1))

                [derivW1,derivW2,derivBiasHidden,derivBiasOutput,dhInput,dbhInput] = b.backPropagation(rete, xb, tb, derivFunActOutput, derivFunErr,softmax)

                rete = regolaAggiornamento(rete, eta, derivW1, derivW2, derivBiasHidden, derivBiasOutput, dhInput,dbhInput)
        else:
            #Learning di tipo batch
            [derivW1,derivW2,derivBiasHidden,derivBiasOutput,dhInput,dbhInput] = b.backPropagation(rete, x, t, derivFunActOutput, derivFunErr,softmax)

            rete = regolaAggiornamento(rete, eta, derivW1, derivW2, derivBiasHidden, derivBiasOutput,dhInput, dbhInput)    
        
        #Vado a simulare gli output della rete dopo l'aggiormenento alla epochesima-epoca e calcolo l'errore
        y = b.simulaRete(rete,x)
        yVal = b.simulaRete(rete,xVal)

        #Per ottenere l'errore medio su tutte le coppie del dataset vado a dividere per il numero di coppie N usate nella simulazione
        err[0][epoca] = funErr(y,t) / len(x)
        errVal[0][epoca] = funErr(yVal,tVal) / len(xVal)

        print(epoca,"\t\t",err[0][epoca],"\t\t",errVal[0][epoca])

        #Verifico se l'errore di valutazione è minore dell'errore minimo
        if (errVal[0][epoca] < minErr).any():
            minErr = errVal[0][epoca]
            reteScelta = rete

    return [reteScelta, err, errVal]


    







    
