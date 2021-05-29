import numpy as np
import backPropagation as b
import funzioniAttivazione as f

def learningPhase(rete, N, x, t, xVal, tVal, batch, eta, derivFunActHidden, derivFunActOutput, derivFunErr,regolaAggiornamento):
    [dervivW1,derivW2,derivBiasHidden,derivBiasOutput] = [0,0,0,0]
    #Learning phase

    #Per il fatto del "rete multistrato con qualsiasi funzione di attivazione per ciascuno strato"
    #Devo prendere in input un array di deriv funzione di attivazione hidden
    err = np.ones((1,N))
    errVal = np.ones((1,N))
    yVal = b.simulaRete(rete,xVal)
    minErr = f.sumOfSquares(yVal,tVal)
    reteScelta = rete

    if batch == 1:
        eta = 0.5
    
    for epoca in range(1,N):
        if batch == 0:
            #Learning di tipo on-line
            for n in range(1,len(x[0])):
                #Provo a fare con un solo strato interno
                [derivW1,derivW2,derivBiasHidden,derivBiasOutput] = b.backPropagation(rete, x, t, derivFunActHidden, derivFunActOutput, derivFunErr)

                rete = regolaAggiornamento(rete, eta, derivW1, derivW2, derivBiasHidden, derivBiasOutput)
        else:
            #Learning di tipo batch
            [dervivW1,derivW2,derivBiasHidden,derivBiasOutput] = b.backPropagation(rete, x, t, derivFunActHidden, derivBiasOutput, derivFunErr)

            rete = regolaAggiornamento(rete, eta, derivW1, derivW2, derivBiasHidden, derivBiasOut)
        
        #Vado a simulare gli output della rete dopo l'aggiormenento alla epochesima-epoca e calcolo l'errore
        y = f.simulaRete(rete,x)
        yVal = f.simulaRete(rete,xVal)
        err[epoca] = f.sumOfSquares(y,t)
        errVal[epoca] = f.sumOfSquares(yVal,tVal)

        #Verifico se l'errore di valutazione è minore dell'errore minimo
        if errVal[epoca] < minErr:
            minErr = errVal[epoca]
            reteScelta = rete

    return [reteScelta, err, errVal]

    







    
