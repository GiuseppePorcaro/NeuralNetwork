import numpy as np

def RPROP(rete, etaPos, etaNeg, derivHidden,derivOutput,derivBHidden, derivBOutput, derivHiddenPre, derivOutputPre, derivBHiddenPre, derivBOutputPre):
    
    #Aggiornamento dei delta dei parametri della rete
    rete.DELTAW1 = aggiornaDELTA(rete.DELTAW1, derivHidden, derivHiddenPre, etaPos, etaNeg)
    rete.DELTAWOutput = aggiornaDELTA(rete.DELTAWOutput, derivOutput, derivOutputPre,etaPos,etaNeg)
    rete.DELTAB1 = aggiornaDELTA(rete.DELTAB1, derivBHidden, derivBHiddenPre,etaPos, etaNeg)
    rete.DELTABOutput = aggiornaDELTA(rete.DELTABOutput, derivBOutput, derivBOutputPre,etaPos,etaNeg)

    #Aggiornamento dei parametri come stabilito dalla regola di aggiornamento RPROP
    rete.W1 = rete.W1 - np.sign(derivHidden)*rete.DELTAW1
    rete.b1 = rete.b1 - np.sign(derivBHidden)*rete.DELTAB1
    rete.WOutput = rete.WOutput - np.sign(derivOutput)*rete.DELTAWOutput
    rete.bOutput = rete.bOutput - np.sign(derivBOutput)*rete.DELTABOutput
    
    #Restituisco le derivate correnti che diventeranno quelle precedenti nella prossima epoca
    return [rete, derivHidden,derivOutput,derivBHidden,derivBOutput]

def aggiornaDELTA(DELTA, deriv, derivPre,etaPos, etaNeg):
    #Vado a verificare il prodotto dei gradienti e aggiorno di conseguenza i DELTA dei parametri
    deltaMax = 5
    deltaMin = 0.005
    for i in range(0,len(DELTA)):
        for j in range(0, len(DELTA[0])):
            if deriv[i][j] * derivPre[i][j] > 0:
                DELTA[i][j] = min(DELTA[i][j]*etaPos, deltaMax)
            elif deriv[i][j] * derivPre[i][j] < 0:
                DELTA[i][j] = max(DELTA[i][j]*etaNeg,deltaMin)
    
    return DELTA

def aggiornaDELTA_Paper(DELTA, pesi, deriv, derivPre,etaPos, etaNeg):
    deltaMax = 5
    deltaMin = 0.005
    for i in range(0,len(DELTA)):
        for j in range(0,len(DELTA[0])):
            if deriv[i][j] * derivPre[i][j] > 0:
                DELTA[i][j] = min(DELTA[i][j]*etaPos, deltaMax)
                pesi[i][j] = pesi[i][j] - np.sign(deriv[i][j])*DELTA[i][j]
            elif deriv[i][j] * derivPre[i][j] < 0:
                DELTA[i][j] = max(DELTA[i][j]*etaNeg,deltaMin)
                deriv[i][j] = 0
            elif deriv[i][j] * derivPre[i][j] == 0:
                pesi[i][j] = pesi[i][j] - np.sign(deriv[i][j])*DELTA[i][j]
    return [DELTA, pesi, deriv]

def discesaDelGradiente(rete,eta,derivHidden,derivOut,derivBiasHidden,derivBiasOut):

    rete.W1 = rete.W1 - eta * derivHidden 
    rete.WOutput = rete.WOutput - eta * derivOut
    rete.b1 = rete.b1 - eta * derivBiasHidden
    rete.bOut = rete.bOutput - eta * derivBiasOut
    
    return rete