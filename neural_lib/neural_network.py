# neural_lib/network.py

import numpy as np
from . import activation_functions

class NeuralNetwork:
    """
    Implementa uma rede neural artificial do tipo Perceptron Multicamadas (MLP).
    """
    def __init__(self, num_input_neurons, hidden_layer_sizes, num_output_neurons, 
                 activation_function='relu', output_activation_function='relu'):
        """Inicializa a arquitetura da rede, os pesos e os viéses.

        Args:
            num_input_neurons (int): O número de neurônios na camada de entrada, que deve
                corresponder ao número de features dos dados.
            hidden_layer_sizes (list[int]): Uma lista contendo o número de neurônios para cada
                camada oculta. Ex: [10, 5] para duas camadas ocultas.
            num_output_neurons (int): O número de neurônios na camada de saída.
            activation_function (str, optional): O nome da função de ativação a ser usada nas camadas intermediárias
                Default: 'relu'. Opções: 'relu', 'sigmoid' e 'tanh'.
            output_activation_function (str, optional): O nome da função de ativação a ser usada na camada de saída
                Defaults: 'relu'. Opções: 'relu', 'sigmoid' e 'tanh'.    
        """

        # Transforma a configuração das camadas da rede em uma só propriedade.
        self.layers_config = [num_input_neurons] + hidden_layer_sizes + [num_output_neurons]
        self.activation_function_name = activation_function
        self.output_activation_function_name = output_activation_function
        
        self.weights = []
        self.biases = []

        self.activations = {
            'sigmoid': activation_functions.sigmoid,
            'relu': activation_functions.relu,
            'tanh': activation_functions.tanh
        }
        
        # Itera a partir da primeira camada oculta para criar os pesos e viéses que conectam cada camada à sua anterior.
        for i in range(1, len(self.layers_config)):
            num_neurons_current = self.layers_config[i]
            num_neurons_previous = self.layers_config[i-1]
            
            # Os pesos são inicializados com valores aleatórios pequenos para quebrar a simetria.
            w = np.random.randn(num_neurons_previous, num_neurons_current) * 0.01
            
            # Os viéses são inicializados com zero.
            b = np.zeros((num_neurons_current, 1))
            
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
        
        cache = {'y0': current_y}
        
        activation_func = self.activations[self.activation_function_name]

        output_activation_func = self.activations[self.output_activation_function_name]

        weigth_length = len(self.weights);

        for l in range(weigth_length):
            W = self.weights[l]
            b = self.biases[l]
            y_anterior = current_y
            
            # Calcula a combinação linear, conforme a fórmula v = W.T * X + b.
            v = np.dot(W.T, y_anterior) + b
            
            # Aplica a função de ativação não-linear.
            # Se for a última camada usa função de ativação da saída.
            # Senão utiliza função de ativação de camada oculta.
            if (l == weigth_length - 1):
                current_y = output_activation_func(v)
            else:
                current_y = activation_func(v)
            
            cache[f'v{l+1}'] = v
            cache[f'y{l+1}'] = current_y

        y_hat = current_y
        
        # Transpõe a saída final para o formato padrão (n_amostras, n_features_saida).
        return y_hat.T, cache