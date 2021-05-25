import rete as r
import numpy as np
import funzioniAttivazione as fa
import matplotlib.pyplot as plt
import backPropagation as bck

def main():
    
    arrayFa = [fa.sigmoide,fa.sigmoide]

    rete = r.nuovaRete(5,3,7,3,arrayFa,fa.identity)

    x = np.random.randn(5,1)

    r.infoRete(rete)
    
    [y,a1,z1,a2] = bck.forwardStep(rete,x)

    print("y:\n",y)










main()