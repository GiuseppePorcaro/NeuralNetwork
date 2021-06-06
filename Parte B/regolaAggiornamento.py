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

    #Capire di quale funzione sing() si tratta e implementare l'aggiornamento dei pesi.
    #Forse da fare anche con i bias
    

    return rete

def discesaDelGradiente(rete,eta,derivHidden,derivOut,derivBiasHidden,derivBiasOut,dhInput):

    rete.W1 = rete.W1 - eta * derivHidden #(m,d) - val * (m,d)
    rete.WOutput = rete.WOutput - eta * derivOut
    rete.b1 = rete.b1 - eta * derivBiasHidden
    rete.bOut = rete.bOutput - eta * derivBiasOut
    
    return rete