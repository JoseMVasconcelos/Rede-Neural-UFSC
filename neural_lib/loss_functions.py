import numpy as np

# Função do erro quadrático
def mean_squared_error(true_y, predicted_y):
    return np.mean((true_y - predicted_y)**2)

def mean_squared_error_derivative(true_y, predicted_y):
    return 2*(predicted_y - true_y)

# Função de entropia cruzada binária
def binary_crossentropy(true_y, predicted_y):
    return -np.mean(true_y * np.log(predicted_y) + (1 - true_y) * np.log(1 - predicted_y))

def binary_crossentropy_derivative(true_y, predicted_y):
    return predicted_y - true_y


# Função de entropia cruzada multiclasse
def multiclass_crossentropy(true_y, predicted_y):

    epsilon = 1e-15
    predicted_y = np.clip(predicted_y, epsilon, 1 - epsilon)
    
    if len(true_y.shape) == 1:
        num_samples = len(true_y)
        num_classes = predicted_y.shape[1]
        y_true_one_hot = np.zeros((num_samples, num_classes))
        y_true_one_hot[np.arange(num_samples), true_y] = 1
    else:
        y_true_one_hot = true_y

    correct_class_prob = predicted_y[np.arange(len(true_y)), true_y]

    return -np.mean(np.sum(y_true_one_hot * np.log(correct_class_prob)))

def multiclass_crossentropy_derivative(true_y, predicted_y):
    if len(true_y.shape) == 1:
        num_samples = len(true_y)
        num_classes = predicted_y.shape[1]
        y_true_one_hot = np.zeros((num_samples, num_classes))
        y_true_one_hot[np.arange(num_samples), true_y] = 1
    else:
        y_true_one_hot = true_y
    return predicted_y - y_true_one_hot