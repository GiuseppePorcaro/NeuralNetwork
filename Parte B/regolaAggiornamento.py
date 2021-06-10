import numpy as np

def RPROP(rete, etaPos, etaNeg, derivHidden,derivOutput,derivBHidden, derivBOutput, derivHiddenPre, derivOutputPre, derivBHiddenPre, derivBOutputPre):
    etaPosMax = 2.0
    etaNegMin = 0.001
    
    #regola di aggiornamento resilient backPropagation
    if derivHidden * derivHiddenPre < 0:
        rete.deltaHidden = max(etaNegMin,rete.deltaHidden * etaNeg)
    if derivHidden * derivHiddenPre > 0:
        rete.deltaHidden = min(etaPosMax,rete.deltaHidden * etaPos)

    if derivOutput * derivOutputPre < 0:
        rete.deltaOutput = max(etaNegMin,rete.deltaOutput * etaNeg)
    if derivOutput * derivOutputPre > 0:
        rete.deltaOutput = min(etaPosMax,rete.deltaOutput * etaPos)

    if derivBHidden * derivBHiddenPre < 0:
        rete.deltaBHidden = max(etaNegMin,rete.deltaBHidden * etaNeg)
    if derivBHidden * derivBHiddenPre > 0:
        rete.deltaBHidden = min(etaPosMax,rete.deltaBHidden * etaPos)

    if derivBOutput * derivBOutputPre < 0:
        rete.deltaBOutput = max(etaNegMin,rete.deltaBOutput * etaNeg)
    if derivBOutput * derivBOutputPre > 0:
        rete.deltaBOutput = min(etaPosMax,rete.deltaBOutput * etaPos)

    rete.W1 = rete.W1 - np.sign(derivHidden) * rete.deltaHidden
    rete.b1 = rete.b1 - np.sign(derivBHidden) * rete.deltaBHidden
    rete.WOutput = rete.WOutput - np.sign(derivOutput) * rete.deltaOutput
    rete.bOutput = rete.bOutput - np.sign(derivBOutput) * rete.deltaBOutput
    
    return rete

def discesaDelGradiente(rete,eta,derivHidden,derivOut,derivBiasHidden,derivBiasOut):

    rete.W1 = rete.W1 - eta * derivHidden 
    rete.WOutput = rete.WOutput - eta * derivOut
    rete.b1 = rete.b1 - eta * derivBiasHidden
    rete.bOut = rete.bOutput - eta * derivBiasOut
    
    return rete