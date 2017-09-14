from numpy import *
from linear_regression import *

import matplotlib.pyplot as plt

def run():
    points = genfromtxt('income.csv', delimiter = ',', names = True)
    x_name = "anos_de_escolaridade"
    y_name = "salario"
    #interactions = 13000
    #learning_rate = 0.0037

    interactions = -1
    learning_rate = 0.001

    x = points[x_name]
    y = points[y_name]
    
    polyfit(x, y, 1)

    #y = mx + b
    initial_m = 0
    initial_b = 0
    
    b, m = gradient_descendent_runner(points, initial_b, initial_m, 
    learning_rate, interactions)

    #b = w0 e m = w1
    plt.plot(x, y, '.')
    plt.plot(x, m * x + b, '-')
    plt.show()

if __name__ == '__main__':
    run()