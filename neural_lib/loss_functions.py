import numpy as np

# Função do erro quadrático
def mean_squared_error(true_y, predicted_y):
    return np.mean((true_y - predicted_y)**2)

# Função de entropia cruzada binária
def binary_crossentropy(true_y, predicted_y):
    return -np.mean(true_y * np.log(predicted_y) + (1 - true_y) * np.log(1 - predicted_y))

# Função de entropia cruzada multiclasse
def multiclass_crossentropy(true_y, predicted_y):
    return -np.mean(np.sum(true_y * np.log(predicted_y)))