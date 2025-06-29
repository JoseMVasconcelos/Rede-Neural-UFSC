import numpy as np
import activation_functions
import loss_functions

class NeuralNetwork():
    def __init__(self, input_size, hidden_sizes, output_size, activation, output_activation, loss):
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.loss_function_name = loss

        self.weights = []
        self.biases = []

        #Iniciando pesos e vieses da primeira camada
        self.weights.append(np.random.rand(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros((hidden_sizes[0], 1)))

        #Iniciando pesos e vieses das camadas após a primeira, e antes da ultima
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.rand(hidden_sizes[i-1], hidden_sizes[i]))
            self.biases.append(np.zeros((hidden_sizes[i], 1)))

        #Iniciando pesos e vieses da camada de output
        self.weights.append(np.random.rand(hidden_sizes[-1], output_size))
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
        cache_stack = [y_predicted]
        for l in range(len(self.weights)):
            v = np.dot(self.weights[l].T, y_predicted) + self.biases[l]
            if (l == len(self.weights) - 1):
                y_predicted = self.get_activation_function(self.output_activation_name, False)(v)
            else:
                y_predicted = self.get_activation_function(self.activation_name, False)(v)
            cache_stack.append(v)
            cache_stack.append(y_predicted)

        return y_predicted.T, cache_stack

    def backpropagate(self, predicted_y, x_sample, y_sample, cache):
        for data in cache:
            print(data.shape)
        for l in reversed(range(len(self.weights))):
            if l == len(self.weights)-1:
                #Output error is the E/X derived
                #the return of the loss function derived is E/Y derived
                if self.loss_function_name == "binary":
                    output_error = self.get_loss_function("binary", True)(y_sample, cache.pop()) * self.get_activation_function("sigmoid", True)(cache.pop())
                if self.loss_function_name == "multiclass":
                    output_error = self.get_loss_function("multiclass", True)(y_sample, cache.pop())
                    cache.pop()
                if self.loss_function_name == "mse":
                    output_error = self.get_loss_function("mse", True)(y_sample, cache.pop()) * self.get_activation_function("identity", True)(cache.pop())

                #Weight error derived is the E/W derived
                weight_error_derived = np.dot(cache.pop(), output_error.T)


                self.weights[l] -= weight_error_derived
                self.biases[l] -= np.sum(output_error, axis=0, keepdims=True)

                output_error = np.dot(self.weights[l], output_error)
            
            else:
                if self.activation_name == 'relu':
                    output_error = output_error * self.get_activation_function("relu", True)(cache.pop())
                if self.activation_name == 'tanh':
                    output_error = output_error * self.get_activation_function("tanh", True)(cache.pop())

                weight_error_derived = np.dot(cache.pop(), output_error.T)
                self.weights[l] -= weight_error_derived
                self.biases[l] -= np.sum(output_error, axis=0, keepdims=True)



    def train(self, input_data, input_label, epochs):
        loss_history = []
        accuracy_history = []

        for epoch in range(epochs):
            i = epoch % len(input_data)

            if input_data[i].ndim == 1:
                x_sample = input_data[i].reshape(1, -1)
            else:
                x_sample = input_data[i]

            if input_label[i].ndim == 1:
                y_sample = input_label[i].reshape(1, -1)
            else:
                y_sample = input_label[i]

            predicted_y, cache = self.feedfoward(x_sample)
            train_loss = self.get_loss_function(self.loss_function_name, False)(y_sample, predicted_y)
            loss_history.append(train_loss)

            if self.loss_function_name == "binary":
                accuracy = (np.mean((predicted_y) > 0.5).astype(int) == y_sample) * 1
                print(f"Predicted class: {np.mean((predicted_y) > 0.5).astype(int)}")
                print(f"True Class: {y_sample[0][0]}")
                accuracy_history.append(accuracy)

            self.backpropagate(predicted_y, x_sample, y_sample, cache)

        if self.loss_function_name == "binary":
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

            if input_label[i].ndim == 1:
                y_sample = input_label[i].reshape(1, -1)
            else:
                y_sample = input_label[i]

            predicted_y, _ = self.feedfoward(x_sample)

            if self.loss_function_name == "binary":
                    accuracy = (np.mean((predicted_y) > 0.5).astype(int) == y_sample) * 1
                    print()
                    print(f"Predicted class: {np.mean((predicted_y) > 0.5).astype(int)} -> True Class: {y_sample[0][0]}")
                    accuracy_history.append(accuracy)

        
        if self.loss_function_name == "binary":
            correct = accuracy_history.count(True)
            total = len(accuracy_history)
            acc = (correct/total)*100
            print(f"Accuracy on last epoch {acc}")



ann = NeuralNetwork(
    input_size=2,
    hidden_sizes=[4],
    output_size=1,
    activation="tanh",
    output_activation="sigmoid",
    loss="binary"
)

X_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_input = np.array([[0] ,[1] ,[1] ,[0]])



ann.train(X_input, y_input, 1000)

ann.predict(X_input, y_input)