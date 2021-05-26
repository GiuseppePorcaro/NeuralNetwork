import numpy as np
import backPropagation as b
import funzioniAttivazione as f

def learningPhase(rete, N, x, t, xVal, tVal, batch, eta, derivFunActHidden, derivFunActOutput, derivFunErr):
    #Learning phase

    #Per il fatto del "rete multistrato con qualsiasi funzione di attivazione per ciascuno strato"
    #Devo prendere in input un array di deriv funzione di attivazione hidden
    err = np.ones(1,N)
    errVal = np.ones(1,N)
    yVal = b.simulaRete(rete,xVal)
    minErr = f.sumOfSquares(yVal,tVal)
    reteScelta = rete

    if batch == 1:
        eta = 0.5
    
    for epoca in range(1,N):
        if batch == 0:
            #Learning di tipo on-line
            
            #Provo a fare con un solo strato interno
            [dervivW1,derivW2,derivBiasHidden,derivBiasOutput] = backPropagation()



    
