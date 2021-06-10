import numpy as np
import backPropagation as b
import funzioniAttivazioneErrore as f

def learningPhase(rete, N, x, t, xVal, tVal, batch, eta, derivFunActHidden, derivFunActOutput, derivFunErr,regolaAggiornamento):
    [dervivW1,derivW2,derivBiasHidden,derivBiasOutput] = [0,0,0,0]
    #Learning phase

    err = np.ones((1,N))
    errVal = np.ones((1,N))
    yVal = b.simulaRete(rete,xVal)
    minErr = f.sumOfSquares(yVal,tVal)
    reteScelta = rete

    if batch == 1:
        eta = 0.000005
    
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

                [derivW1,derivW2,derivBiasHidden,derivBiasOutput,dhInput,dbhInput] = b.backPropagation(rete, xb, tb, derivFunActHidden, derivFunActOutput, derivFunErr)

                rete = regolaAggiornamento(rete, eta, derivW1, derivW2, derivBiasHidden, derivBiasOutput, dhInput,dbhInput)
        else:
            #Learning di tipo batch
            [derivW1,derivW2,derivBiasHidden,derivBiasOutput,dhInput,dbhInput] = b.backPropagation(rete, x, t, derivFunActHidden, derivFunActOutput, derivFunErr)

            rete = regolaAggiornamento(rete, eta, derivW1, derivW2, derivBiasHidden, derivBiasOutput,dhInput, dbhInput)
        
        #Vado a simulare gli output della rete dopo l'aggiormenento alla epochesima-epoca e calcolo l'errore
        y = b.simulaRete(rete,x)
        yVal = b.simulaRete(rete,xVal)
        err[0][epoca] = f.crossEntropy(y,t)
        errVal[0][epoca] = f.crossEntropy(yVal,tVal)

        print(epoca,"\t\t",err[0][epoca],"\t\t",errVal[0][epoca])

        #Verifico se l'errore di valutazione è minore dell'errore minimo
        if (errVal[0][epoca] < minErr).any():
            minErr = errVal[0][epoca]
            reteScelta = rete

    return [reteScelta, err, errVal]


    







    
