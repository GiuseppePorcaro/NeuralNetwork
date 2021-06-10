import numpy as np

def discesaDelGradiente(rete,eta,derivHidden,derivOut,derivBiasHidden,derivBiasOut,dhInput,dbhInput):

    if rete.nStrati == 1:
        rete.W1 = rete.W1 - eta * derivHidden 
        rete.WOutput = rete.WOutput - eta * derivOut
        rete.b1 = rete.b1 - eta * derivBiasHidden
        rete.bOut = rete.bOutput - eta * derivBiasOut
    else:
        rete.W1 = rete.W1 - eta * dhInput
        rete.b1 = rete.b1 - eta * dbhInput

        for i in range(0,rete.nStrati-1):
            rete.W[0][i] = rete.W[0][i] - eta * derivHidden[0][i] 
            rete.b[0][i] = rete.b[0][i] - eta * derivBiasHidden[0][i]

        rete.WOutput = rete.WOutput - eta * derivOut
        rete.bOut = rete.bOutput - eta * derivBiasOut

    return rete