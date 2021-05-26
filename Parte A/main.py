import rete as r
import numpy as np
import funzioniAttivazione as fa
import matplotlib.pyplot as plt
import backPropagation as bck

def main():
    
    arrayFa = [fa.sigmoide,fa.sigmoide]

    rete = r.nuovaRete(5,3,7,3,arrayFa,fa.identity)

    x = np.array([1,2,3,4,5])
    t = np.array([6,7,8,9,10])

    r.infoRete(rete)
    
    y = fa.derivSumOfSquares(x,t)

    print(y)










main()