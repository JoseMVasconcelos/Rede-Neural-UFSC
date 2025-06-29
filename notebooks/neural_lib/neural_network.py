import numpy as np
from . import activation_functions
from . import loss_functions

class NeuralNetwork():
    def __init__(self, input_size, hidden_sizes, output_size, activation, output_activation, loss):
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.loss_function_name = loss

        self.weights = []
        self.biases = []

        #Iniciando pesos e vieses da primeira camada
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2 / input_size))
        self.biases.append(np.zeros((hidden_sizes[0], 1)))

        #Iniciando pesos e vieses das camadas após a primeira, e antes da ultima
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * np.sqrt(2 / input_size))
            self.biases.append(np.zeros((hidden_sizes[i], 1)))

        #Iniciando pesos e vieses da camada de output
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2 / input_size))
        self.biases.append(np.zeros((output_size, 1)))


    def get_activation_function(self, function_name, derivative):
        if derivative:
            match function_name:
                case "relu":
                    return activation_functions.relu_derivative
                case "tanh":
                    return activation_functions.tanh_derivative
                case "sigmoid":
                    return activation_functions.sigmoid_derivative
                case "softmax":
                    return 0
                case "identity":
                    return activation_functions.identity_derivative
        else:
            match function_name:
                case "relu":
                    return activation_functions.relu
                case "tanh":
                    return activation_functions.tanh
                case "sigmoid":
                    return activation_functions.sigmoid
                case "softmax":
                    return activation_functions.softmax
                case "identity":
                    return activation_functions.identity
                
    def get_loss_function(self, loss_name, derivative):
        if derivative:
            match loss_name:
                case "mse":
                    return loss_functions.mse_derivative
                case "binary":
                    return loss_functions.binary_crossentropy_derivative
                case "multiclass":
                    return loss_functions.multiclass_crossentropy_derivative
        else:
            match loss_name:
                case "mse":
                    return loss_functions.mse
                case "binary":
                    return loss_functions.binary_crossentropy
                case "multiclass":
                    return loss_functions.multiclass_crossentropy



    def feedfoward(self, x_sample):
        y_predicted = x_sample.T
        cache_stack = []
        for l in range(len(self.weights)):
            v = np.dot(self.weights[l].T, y_predicted) + self.biases[l]
            if (l == len(self.weights) - 1):
                y_predicted = self.get_activation_function(self.output_activation_name, False)(v)
            else:
                y_predicted = self.get_activation_function(self.activation_name, False)(v)
            cache_stack.append((v, y_predicted))

        return y_predicted.T, cache_stack

    def backpropagate(self, y_sample, cache, x_sample, learning_rate):
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        v_last, act_last = cache.pop()

        if self.loss_function_name == "binary":
            output_error = self.get_loss_function("binary", True)(y_sample, act_last) * self.get_activation_function("sigmoid", True)(v_last)
        if self.loss_function_name == "multiclass":
            output_error = self.get_loss_function("multiclass", True)(y_sample, act_last)
        if self.loss_function_name == "mse":
            output_error = self.get_loss_function("mse", True)(y_sample, act_last) * self.get_activation_function("identity", True)(v_last)

        for l in reversed(range(len(self.weights))):
            if l == 0:
                a_previous = x_sample.T
            else:
                v_previous, a_previous = cache.pop()

            dw[l] = learning_rate * np.dot(a_previous, output_error.T)
            db[l] = learning_rate * np.sum(output_error, axis=0, keepdims=True)

            if l > 0:
                output_error = np.dot(self.weights[l], output_error)
            if l - 1 == len(self.weights) - 1:
                 output_error *= self.get_activation_function(self.output_activation_name, True)(v_previous)
            else:
                 output_error *= self.get_activation_function(self.activation_name, True)(v_previous)

        for l in range(len(self.weights)):
            self.weights[l] -= dw[l]
            self.biases[l] -= db[l]


        # for l in reversed(range(len(self.weights))):
        #     if l == len(self.weights)-1:
        #         #Output error is the E/X derived
        #         #the return of the loss function derived is E/Y derived

        #         ## Camada de ativação da ultima camada
        #         if self.loss_function_name == "binary":
        #             output_error = self.get_loss_function("binary", True)(y_sample, cache.pop()) * self.get_activation_function("sigmoid", True)(cache.pop())
        #         if self.loss_function_name == "multiclass":
        #             output_error = self.get_loss_function("multiclass", True)(y_sample, cache.pop())
        #             cache.pop()
        #         if self.loss_function_name == "mse":
        #             output_error = self.get_loss_function("mse", True)(y_sample, cache.pop()) * self.get_activation_function("identity", True)(cache.pop())
            
        #     else:
        #         if self.activation_name == 'relu':
        #             output_error = output_error * self.get_activation_function("relu", True)(cache.pop())
        #         if self.activation_name == 'tanh':
        #             output_error = output_error * self.get_activation_function("tanh", True)(cache.pop())

        #     weight_error_derived = np.dot(cache.pop(), output_error.T)

        #     dw[l] = learning_rate * weight_error_derived
        #     db[l] = learning_rate * np.sum(output_error, axis=1, keepdims=True)
            
            
        #     output_error = np.dot(self.weights[l], output_error)

        # for l in reversed(range(len(self.weights))):
        #     self.weights[l] -= dw[l]
        #     self.biases[l] -= db[l]


    def train(self, input_data, input_label, epochs, learning_rate=0.001):
        loss_history = []
        accuracy_history = []
        pred_history = []

        for epoch in range(epochs):
            i = epoch % len(input_data)
            if input_data[i].ndim == 1:
                x_sample = input_data[i].reshape(1, -1)
            else:
                x_sample = input_data[i]

            if input_label[i].ndim == 1 or np.array(input_label[i]).ndim == 0:
                y_sample = input_label[i].reshape(1, -1)
            else:
                y_sample = input_label[i]

            predicted_y, cache = self.feedfoward(x_sample)
            train_loss = self.get_loss_function(self.loss_function_name, False)(y_sample, predicted_y)
            loss_history.append(train_loss)
            pred_history.append(predicted_y)

            self.backpropagate(y_sample, cache, x_sample, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} -> Train Loss: {train_loss:.4f}")


        if self.loss_function_name == "mse":
            se = self.get_loss_function("mse", False)(input_label[:epoch], pred_history)
        else:
            for prediction in pred_history:
                if self.loss_function_name == "binary":
                    accuracy = (np.mean((prediction) > 0.5).astype(int) == y_sample) * 1
                    accuracy_history.append(accuracy)
                if self.loss_function_name == "multiclass":
                    accuracy = np.mean(np.argmax(prediction, axis=1) == y_sample) * 1
                    accuracy_history.append(accuracy)

                


        if self.loss_function_name == 'binary' or self.loss_function_name == 'multiclass':
            correct = accuracy_history.count(True)
            total = len(accuracy_history)
            acc = (correct/total)*100
            print(f"Accuracy on last epoch {acc}")
        
    def predict(self, input_data, input_label):
        accuracy_history = []

        for i in range(len(input_data)):
            if input_data[i].ndim == 1:
                x_sample = input_data[i].reshape(1, -1)
            else:
                x_sample = input_data[i]
            if input_label[i].ndim == 1 or np.array(input_label[i]).ndim == 0:
                y_sample = input_label[i].reshape(1, -1)
            else:
                y_sample = input_label[i]

            predicted_y, _ = self.feedfoward(x_sample)

            if self.loss_function_name == "binary":
                    accuracy = (np.mean((predicted_y) > 0.5).astype(int) == y_sample) * 1
                    # print()
                    # print(f"Predicted class: {np.mean((predicted_y) > 0.5).astype(int)} -> True Class: {y_sample[0][0]}")
                    accuracy_history.append(accuracy)
            if self.loss_function_name == "multiclass":
                print(predicted_y)
                accuracy = np.mean(np.argmax(predicted_y, axis=1) == y_sample) * 1
                accuracy_history.append(accuracy)

        
        if self.loss_function_name == "binary":
            correct = accuracy_history.count(True)
            total = len(accuracy_history)
            acc = (correct/total)*100
            return acc
        if self.loss_function_name == "multiclass":
            correct = accuracy_history.count(True)
            total = len(accuracy_history)
            acc = (correct/total)*100
            return acc



