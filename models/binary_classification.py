from neural_lib.neural_network import NeuralNetwork

import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess(csv_path):
    df = pl.read_csv(csv_path)

    continuos_features = [
        'Age', 'AnnualIncome', 'NumberOfPurchases',
        'TimeSpentOnWebsite', 'DiscountsAvailed'
    ]
    categorical_features = ['ProductCategory']
    passthrough_feats = ['Gender', 'LoyaltyProgram']

    features = continuos_features + categorical_features + passthrough_feats
    X_raw = df.select(features).to_numpy()
    y = df.select('PurchaseStatus').to_numpy().flatten()

    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), list(range(0, len(continuos_features)))),
        ('onehot', OneHotEncoder(sparse_output=False), [len(continuos_features)]),
    ], remainder='passthrough')

    X = preprocessor.fit_transform(X_raw)
    return X, y

def train_model(X_train, y_train, hidden_layers, epochs, activation_function):
    input_size = X_train.shape[1]
    neural_network = NeuralNetwork(
        num_input_neurons=input_size,
        hidden_layer_sizes=hidden_layers,
        activation_function=activation_function,
        num_output_neurons=1,
        output_activation_function='sigmoid',
        loss_function='binary'
    )

    neural_network.train(X_train, y_train, epochs)
    return neural_network

def evaluate_model(neural_network, X_test, y_test):
    predictions = neural_network.predict(X_test).flatten()
    results = (predictions >= 0.5).astype(int)
    accuracy = np.mean(results == y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

def get_params():
    return [
        {'hidden_layers': [32, 16], 'activation_function': 'relu',    'epochs': 100},
        {'hidden_layers': [32, 16], 'activation_function': 'tanh',    'epochs': 100},

        {'hidden_layers': [64, 32, 16], 'activation_function': 'relu',    'epochs': 200},
        {'hidden_layers': [64, 32, 16], 'activation_function': 'tanh',    'epochs': 200},

        {'hidden_layers': [128, 64, 32], 'activation_function': 'relu',    'epochs': 300},
        {'hidden_layers': [128, 64, 32], 'activation_function': 'tanh',    'epochs': 300},
    ]

def main():
    X, y = load_and_preprocess('./../data/customer-purchase-behavior.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for params in get_params():
        neural_network = train_model(
            X_train, y_train,
            params['hidden_layers'],
            params['epochs'],
            params['activation_function']
        )
        evaluate_model(neural_network, X_test, y_test)

if __name__ == '__main__':
    main()

