import numpy as np
import backPropagation as b
import funzioniAttivazioneErrore as f

def learningPhase(rete, N, etaPos, etaNeg, x, t, valX, valT, derivFunActHidden, derivFunActOutput, derivFunErr,regolaAggiornamento):
    #Per questa parte del progetto la learing phase implementerà solo la modalità batch

    [dervivW1,derivW2,derivBiasHidden,derivBiasOutput] = [0,0,0,0]
    [derivW1Pre,derivW2pre,derivBiasHiddenPre,derivBiasOutputPre] = [0,0,0,0]
    
    err = np.ones((1,N))
    errVal = np.ones((1,N))
    yVal = b.simulaRete(rete,xVal)
    minErr = f.sumOfSquares(yVal,tVal)
    reteScelta = rete

    eta = 0.01

    for epoca in range(0,N):

        #Learning Batch
        [dervivW1,derivW2,derivBiasHidden,derivBiasOutput] = b.backPropagation(rete, x, t, derivFunActHidden, derivFunActOutput, derivFunErr)

        if epoca == 0:
            #Alla prima epoca, non avendo le derivate precedenti, faccio la discesa del gradiente
            rete = discesaDelGradiente(rete,eta,derivHidden,derivOut,derivBiasHidden,derivBiasOut)
        else:
            #Dalla seconda epoca in poi posso effettuare la RPROP
            rete = regolaAggiornamento(rete, etaPos, etaNeg, dervivW1,derivW2,derivBiasHidden, derivBiasOutput, derivW1Pre, derivW2Pre, derivBiasHiddenPre, derivBiasOutputPre):

        [derivW1Pre,derivW2pre,derivBiasHiddenPre,derivBiasOutputPre] = [dervivW1,derivW2,derivBiasHidden,derivBiasOutput]

    


