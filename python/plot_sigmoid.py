import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz;
    
if __name__ == '__main__':
    sigmoid = lambda x: 1./(1 + np.exp(-x))
    x = np.linspace(-10., 10., 1000)
    plt.plot(x, sigmoid(x))
    plt.show()
    #tikz.save('sigmoid.tex', standalone=True)

    relu = lambda x: np.where(x>0, x, 0)
    plt.plot(x, relu(x))
    plt.show()
    # tikz.save('relu.tex', standalone=True)
