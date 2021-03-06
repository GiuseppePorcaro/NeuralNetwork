import numpy as np
import rete as re
import backPropagation as b
import regolaAggiornamento as r
import funzioniAttivazioneErrore as f

def learningPhase(rete, N, etaPos, etaNeg, x, t, valX, valT, derivFunActHidden, derivFunActOutput, derivFunErr,regolaAggiornamento,softmax,funErrore):
    #Per questa parte del progetto la learing phase implementer√† solo la modalit√† batch

    #Inizializzazione derivate altrimenti sono indefinite
    [derivW1,derivW2,derivBiasHidden,derivBiasOutput] = [0,0,0,0]
    [derivW1Pre,derivW2Pre,derivBiasHiddenPre,derivBiasOutputPre] = [0,0,0,0]
    
    #Inizializzazione array errore e errore minimo
    err = np.ones((1,N))
    errVal = np.ones((1,N))
    yVal = b.simulaRete(rete,valX)
    minErr = funErrore(yVal,valT)
    reteScelta = rete

    #Eta per la singola discesa del gradiente al passo t = 1
    eta = 0.005
    print("-Eta:\t\t",eta)

    print("\nEpoca\t\t Errore\t\t\t\t Errore Valutazione")
    print("---------------------------------------------------------------------------")
    for epoca in range(0,N):
        #Learning Batch
        [derivW1,derivW2,derivBiasHidden,derivBiasOutput] = b.backPropagation(rete, x, t, derivFunActHidden, derivFunActOutput, derivFunErr,softmax)

        if epoca == 0:
            #Alla prima epoca, non avendo le derivate precedenti, faccio la discesa del gradiente
            rete = r.discesaDelGradiente(rete,eta,derivW1,derivW2,derivBiasHidden,derivBiasOutput)
            [derivW1Pre,derivW2Pre,derivBiasHiddenPre,derivBiasOutputPre] = [derivW1,derivW2,derivBiasHidden,derivBiasOutput]
        else:
            #Dalla seconda epoca in poi posso effettuare la RPROP
            [rete, derivW1Pre,derivW2Pre,derivBiasHiddenPre,derivBiasOutputPre] = regolaAggiornamento(rete, etaPos, etaNeg, derivW1,derivW2,derivBiasHidden, derivBiasOutput, derivW1Pre, derivW2Pre, derivBiasHiddenPre, derivBiasOutputPre)

        #Vado a verificare l'errore sia sul train-set sia sul validation-set. (con k-fold √® il test-set)
        y = b.simulaRete(rete,x)
        yVal = b.simulaRete(rete,valX)
        err[0][epoca] = funErrore(y,t) / len(x)
        errVal[0][epoca] = funErrore(yVal,valT) / len(valX)

        print(epoca,"\t\t",err[0][epoca],"\t\t",errVal[0][epoca])

        #Vado a salvarmi la rete che minimizza l'errore, usando il validation-set
        if errVal[0][epoca] < minErr:
            minErr = errVal[0][epoca]
            reteScelta = rete
    
    return [rete,err,errVal]


    


