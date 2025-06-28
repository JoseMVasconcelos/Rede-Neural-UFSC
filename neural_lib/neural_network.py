# neural_lib/network.py

import numpy as np
from . import activation_functions
from . import loss_functions

class NeuralNetwork:
    """
    Implementa uma rede neural artificial do tipo Perceptron Multicamadas (MLP).
    """
    def __init__(self, num_input_neurons, hidden_layer_sizes, num_output_neurons, 
                 activation_function='relu', output_activation_function='identity', loss_function='mse'):
        """Inicializa a arquitetura da rede, os pesos e os viéses.

        Args:
            num_input_neurons (int): O número de neurônios na camada de entrada, que deve
                corresponder ao número de features dos dados.
            hidden_layer_sizes (list[int]): Uma lista contendo o número de neurônios para cada
                camada oculta. Ex: [10, 5] para duas camadas ocultas.
            num_output_neurons (int): O número de neurônios na camada de saída.
            activation_function (str, optional): O nome da função de ativação a ser usada nas camadas intermediárias
                Default: 'relu'. Opções: 'relu', 'sigmoid', 'tanh', 'identity' e 'softmax'.
            output_activation_function (str, optional): O nome da função de ativação a ser usada na camada de saída
                Default: 'identity'. Opções: 'relu', 'sigmoid', 'tanh', 'identity' e 'softmax'.    
        """

        # Transforma a configuração das camadas da rede em uma só propriedade.
        self.layers_config = [num_input_neurons] + hidden_layer_sizes + [num_output_neurons]
        self.activation_function_name = activation_function
        self.output_activation_function_name = output_activation_function
        self.loss_function = loss_function
        
        self.weights = []
        self.biases = []
        self.dW = []
        self.db = []

        self.activations = {
            'sigmoid': activation_functions.sigmoid,
            'relu': activation_functions.relu,
            'tanh': activation_functions.tanh,
            'identity': activation_functions.identity,
            'softmax': activation_functions.softmax
        }

        self.act_derivatives = {
            'sigmoid': activation_functions.sigmoid_derivative,
            'relu': activation_functions.relu_derivative,
            'tanh': activation_functions.tanh_derivative,
            'identity': activation_functions.identity_derivative,
            'softmax': activation_functions.softmax_derivative
        }

        self.loss = {
            'mse': loss_functions.mean_squared_error,
            'binary': loss_functions.binary_crossentropy,
            'multiclass': loss_functions.multiclass_crossentropy
        }

        self.loss_derivative = {
            'mse': loss_functions.mean_squared_error_derivative,
            'binary': loss_functions.binary_crossentropy_derivative,
            'multiclass': loss_functions.multiclass_crossentropy_derivative
        }
        
        # Itera a partir da primeira camada oculta para criar os pesos e viéses que conectam cada camada à sua anterior.
        for i in range(1, len(self.layers_config)):
            w = np.random.randn(self.layers_config[i-1], self.layers_config[i]) * np.sqrt(1 / self.layers_config[i-1])
            b = np.zeros((self.layers_config[i], 1))
            self.weights.append(w)
            self.biases.append(b)

    def feedforward(self, input_data):
        """Executa a propagação para frente (forward pass) na rede.

        Args:
            input_data (np.array): Um array NumPy com os dados de entrada, onde cada linha
                é uma amostra e cada coluna é uma feature. 
                Dimensão esperada: (n_amostras, n_features_entrada).

        Returns:
            tuple[np.array, dict]: Uma tupla contendo:
                - y_hat (np.array): A matriz de predições da rede no formato (n_amostras, n_features_saida).
                - cache (dict): Um dicionário com os valores intermediários (v e y) de cada
                  camada, necessário para o backpropagation.
        """
        # Transpõe a entrada para a convenção interna da rede: (n_features, n_amostras).
        current_y = input_data.T
        
        activation_func = self.activations[self.activation_function_name]
        output_activation_func = self.activations[self.output_activation_function_name]

        for l in range(len(self.weights)):
            # Calcula a combinação linear, conforme a fórmula v = W.T * X + b.
            v = np.dot(self.weights[l].T, current_y) + self.biases[l]
            
            # Aplica a função de ativação não-linear.
            # Se for a última camada usa função de ativação da saída.
            # Senão utiliza função de ativação de camada oculta.
            if (l == len(self.weights) - 1):
                current_y = output_activation_func(v)
            else:
                current_y = activation_func(v)
        
        # Transpõe a saída final para o formato padrão (n_amostras, n_features_saida).
        return current_y.T
    
    def backpropagate(self, y_hat, input_data, expected_output):
        current_y = input_data.T

        intermediate_values = {'y0': input_data.T}
        loss_function_derivative = self.loss_derivative[self.loss_function]
        activation_func = self.activations[self.activation_function_name]
        output_activation_func = self.act_derivatives[self.output_activation_function_name]

        for l in range(len(self.weights)):
            v = np.dot(self.weights[l].T, current_y) + self.biases[l]
            if (l == len(self.weights) - 1):
                current_y = output_activation_func(v)
            else:
                current_y = activation_func(v)
            intermediate_values[f'v{l+1}'] = v
            intermediate_values[f'y{l+1}'] = current_y

        self.dW = [np.zeros_like(w) for w in self.weights]
        self.db = [np.zeros_like(b) for b in self.biases]

        v_output = intermediate_values[f'v{len(self.weights)}']
        delta = (y_hat.T - expected_output.T) * output_activation_func(v_output)

        for l in reversed(range(len(self.weights))):
            y_previous = intermediate_values[f'y{l}']

            self.dW[l] = np.dot(y_previous, delta.T)
            self.db[l] = np.sum(delta, axis=1, keepdims=True)
            
            if l > 0:
                W_current = self.weights[l]
                v_prev_layer = intermediate_values[f'v{l}']
                activation_derivative = self.act_derivatives[self.activation_function_name]
                delta = np.dot(W_current, delta) * activation_derivative(v_prev_layer)

    def update_weights(self):
        for l in range(len(self.weights)):
            self.weights[l] -= self.dW[l]
            self.biases[l] -= self.db[l]

    def train(self, input_data, input_label, epochs):
        for j in range(epochs):
            i = j % len(input_data)
            if input_data[i].ndim == 1:
                sample = input_data[i].reshape(1, -1)
            else:
                sample = input_data[i]

            label = input_label[i].reshape(1)
            predicted_y = self.feedforward(sample)
            self.backpropagate(predicted_y, sample, label)
            self.update_weights()
            

    def predict(self, input_data):
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        else:
            input_data = input_data
        y_hat = self.feedforward(input_data)
        return y_hat

    