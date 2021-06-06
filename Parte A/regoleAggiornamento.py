import numpy as np

def discesaDelGradiente(rete,eta,derivHidden,derivOut,derivBiasHidden,derivBiasOut,dhInput):
    if rete.nStrati == 1:
        rete.W1 = rete.W1 - eta * derivHidden #(m,d) - val * (m,d)
        rete.WOutput = rete.WOutput - eta * derivOut
        rete.b1 = rete.b1 - eta * derivBiasHidden
        rete.bOut = rete.bOutput - eta * derivBiasOut
    else:
        rete.W1 = rete.W1 - eta * dhInput
        rete.b1 = rete.b1 - eta * derivBiasHidden[0,:]

        for i in range(1,rete.nStrati-1):
            rete.W[i,:,:] = rete.W[i,:,:] - eta * derivHidden[i,:,:] #(m,d) - val * (m,d)
            rete.b[i,:] = rete.b[i,:] - eta * derivBiasHidden[i,:]

        rete.WOutput = rete.WOutput - eta * derivOut
        rete.bOut = rete.bOutput - eta * derivBiasOut

    return rete