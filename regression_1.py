# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:13:13 2020

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv", delimiter = ',')
data_x = data[:, 0]
data_y = data[:, 1]
plt.scatter(data_x, data_y)
plt.show()

lr = 0.0005
epochs = 50
b = 0
k = 0
def cost_fun(b, k, data_x, data_y):
    totalerr = 0
    for i in range(0, len(data_x)):
        totalerr += (data_y[i] - (k * data_x[i] + b))**2
    return totalerr / len(data_x) / 2

def grad_descent(data_x, datay, lr, b, k, epochs):
    m = len(data_x)
    for i in range(epochs):
        grad_b = 0
        grad_k = 0
        for j in range(0, len(data_x)):
            grad_b += 1/m * ((k * data_x[j]+b) - data_y[j])
            grad_k += 1/m * data_x[j] * ((data_x[j] * k + b) - data_y[j])
        b -= lr * grad_b
        k -= lr * grad_k
    return b, k

print("starting: b : {0}, k : {1}, costerr : {2}".format(b, k, cost_fun(b, k, data_x, data_y)))
b, k = grad_descent(data_x, data_y, lr, b ,k, epochs)
print("after: b : {0}, k : {1}, costerr : {2}".format(b, k, cost_fun(b, k, data_x, data_y)))

plt.plot(data_x, data_y, 'g.')
plt.plot(data_x, data_x * k + b, 'r')
plt.show()
