import numpy as np
import matplotlib.pyplot as plt


# Função sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1-sig)

#Função Rectified Linear (ReLU)
def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return np.where(x >= 0, 1, 0)

#Função Tangente Hiperbólica (tanh)
def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    return 1 - tanh(x)**2
