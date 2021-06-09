import numpy as np
import backPropagation as b
import regolaAggiornamento as r
import funzioniAttivazioneErrore as f

def learningPhase(rete, N, etaPos, etaNeg, x, t, valX, valT, derivFunActHidden, derivFunActOutput, derivFunErr,regolaAggiornamento):
    #Per questa parte del progetto la learing phase implementerà solo la modalità batch

    [dervivW1,derivW2,derivBiasHidden,derivBiasOutput] = [0,0,0,0]
    [derivW1Pre,derivW2pre,derivBiasHiddenPre,derivBiasOutputPre] = [0,0,0,0]
    
    err = np.ones((1,N))
    errVal = np.ones((1,N))
    yVal = b.simulaRete(rete,valX)
    minErr = f.sumOfSquares(yVal,valT)
    reteScelta = rete

    eta = 0.01
    for epoca in range(0,N):
        print("epoca: ",epoca)
        #Learning Batch
        [derivW1,derivW2,derivBiasHidden,derivBiasOutput] = b.backPropagation(rete, x, t, derivFunActHidden, derivFunActOutput, derivFunErr)

        if epoca == 0:
            #Alla prima epoca, non avendo le derivate precedenti, faccio la discesa del gradiente
            rete = r.discesaDelGradiente(rete,eta,derivW1,derivW2,derivBiasHidden,derivBiasOutput)
        else:
            #Dalla seconda epoca in poi posso effettuare la RPROP
            rete = regolaAggiornamento(rete, etaPos, etaNeg, derivW1,derivW2,derivBiasHidden, derivBiasOutput, derivW1Pre, derivW2Pre, derivBiasHiddenPre, derivBiasOutputPre)

        [derivW1Pre,derivW2pre,derivBiasHiddenPre,derivBiasOutputPre] = [derivW1,derivW2,derivBiasHidden,derivBiasOutput]
    
        #Vado a verificare l'errore sia sul train-set sia sul validation-set
        y = b.simulaRete(rete,x)
        yVal = b.simulaRete(rete,valX)
        err[0][epoca] = f.crossEntropy(y,t)
        errVal[0][epoca] = f.crossEntropy(yVal,valT)

        #Vado a salvarmi la rete che minimizza l'errore, usando il validation-set
        if errVal[0][epoca] < minErr:
            minErr = errVal[0][epoca]
            reteScelta = rete
    
    return [rete,err,errVal]


    


