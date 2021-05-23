import numpy as np

def sigmoide(x):
    y=(1/(1+np.exp(-x)))
    return y

def main():
    x = np.array([1,2,3,4,5])

    print(sigmoide(x))

main()

