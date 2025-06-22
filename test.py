# test.py (na raiz do projeto)

import numpy as np
from neural_lib.neural_network import NeuralNetwork

def run_test():
    print("--- Testando a versão SIMPLIFICADA da NeuralNetwork ---")

    # 1. Dados de exemplo
    X = np.array([[0.1, 0.5, 0.2], [0.9, 0.3, 0.4]])
    print(f"\n[ENTRADA] Lote de dados X com dimensão: {X.shape}")

    # 2. Definir a arquitetura
    num_entradas = 3
    neuronios_camadas_ocultas = [4, 5]
    num_saidas = 1
    
    print(f"[ARQUITETURA] Neurônios de Entrada: {num_entradas}, Ocultas: {neuronios_camadas_ocultas}, Saída: {num_saidas}")

    # 3. MUDANÇA: Instanciando a rede com apenas UMA função de ativação
    try:
        # Note que agora passamos apenas 'activation_function'.
        # A função 'sigmoid' será usada em todas as camadas.
        net = NeuralNetwork(
            num_input_neurons=num_entradas,
            hidden_layer_sizes=neuronios_camadas_ocultas,
            num_output_neurons=num_saidas,
            activation_function='sigmoid' 
        )
        print("\n[INSTÂNCIA] Objeto NeuralNetwork simplificado criado com sucesso.")
        print(f"  - Função de ativação para todas as camadas: '{net.activation_function_name}'")

    except Exception as e:
        print(f"\n[ERRO] Falha ao instanciar a rede: {e}")
        return

    # 4. Executar o feedforward
    print("\n--- Executando feedforward ---")
    y_hat, cache = net.feedforward(X)
    print(f"[SAÍDA] Predição final (y_hat) com dimensão {y_hat.shape}:\n{y_hat}")
    
    # 5. Verificar a saída
    expected_shape = (X.shape[0], num_saidas)
    if y_hat.shape == expected_shape:
        print(f"\n[SUCESSO] A dimensão da saída está correta!")
    else:
        print(f"\n[FALHA] A dimensão da saída está INCORRETA.")

    print("\n--- Teste concluído ---")

if __name__ == "__main__":
    run_test()