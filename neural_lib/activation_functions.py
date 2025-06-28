import numpy as np

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

#Função identidade (Codada apenas pra ficar mais fácil na rede)
def identity(x):
    return x

def identity_derivative(x):
    return 1

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x,)

def softmax_derivative(x):
    soft = softmax(x)
    return np.diag(soft) - np.outer(soft, soft)
 