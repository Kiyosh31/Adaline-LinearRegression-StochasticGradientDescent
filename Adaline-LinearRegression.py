# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:27:14 2020

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, dim, eta):
        self.n = dim
        self.eta = eta
        self.w = -1 + 2 * np.random.random((dim, 1))
        self.b = -1 + 2 * np.random.rand()
        
    def predict(self, x):
        y = np.dot(self.w.transpose(), x) + self.b
        return y
        
    def train(self, X, y, epochs=50):
        # X -> entradas
        # y -> salidas deseadas
        n, p = X.shape
        for i in range(epochs):
            dw = np.zeros((self.n, 1))
            db = 0
            for j in range(p):
                y_pred = self.predict(X[:,j])
                dw += (y[:, j] - y_pred) * X[:, j].reshape(-1,1)
                db += (y[:, j] - y_pred)
                
            self.w += (self.eta/p)*dw
            self.b += (self.eta/p)*db


data = np.genfromtxt('DataSet1.csv', delimiter=',', dtype=np.float)
#print(data)

X = np.array([data[:, 0]])
y = np.array([data[:, 1]])
#print(X)
#print(y)

# Instancia de Adaline y valores iniciales
net = Adaline(1, 0.5)
w = net.w
b = net.b

plt.title('Datos y coordenadas iniciales')
plt.grid()
plt.plot(X, y, 'ob')
plt.plot([0, 1], [w[0].conj().T * 0 + b, w[0].conj().T * 1 + b], '-m')
plt.show()

# Entrenamiento
net.train(X, y, 1000)
w = net.w
b = net.b

plt.title('Datos finales')
plt.grid()
plt.plot(X, y, 'ob')
plt.plot([0, 1], [w[0].conj().T * 0 + b, w[0].conj().T * 1 + b], '-m')
plt.show()