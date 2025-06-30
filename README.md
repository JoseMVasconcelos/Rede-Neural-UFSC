# Projeto Final - Implementação de uma Rede Neural Artificial - INE5664

**Autores**:
- Bernardo De Marco Gonçalves (22102557);
- José Victor Machado de Vasconcelos (22100906); e,
- Marcos Roberto Fernandes Filho (22100915)

## Neural Network Library

Este projeto implementa uma biblioteca para o treinamento e utilização de Redes Neurais Artificiais (RNAs) do tipo Perceptron Multicamadas (MLP). A implementação da biblioteca encontra-se disponível no _folder_ `neural_lib`.

Para abstrair o conceito de RNAs, foi criada a classe `NeuralNetwork`. Para a inicialização e parametrização da rede, basta instanciar um objeto da classe. É possível parametrizar a quantidade de neurônios na camada de entrada; número de neurônios para cada camada oculta; o número de neurônios na camada de saída; a função de ativação a ser utilizada nas camadas intermediárias; e, a função de ativação a ser utilizada na camada de saída. As funções de ativação `relu`, `sigmoid`, `tanh`, `identity` e `softmax` são suportadas.

https://github.com/JoseMVasconcelos/Rede-Neural-UFSC/blob/953485dd165e47e5c8e36600e1f8af31b997a3d3/neural_lib/neural_network.py#L7-L25

Para treinamento da rede, o método `train` deve ser utilizado. Ele aceita como parâmetro os atributos (`input_data`) e as _labels_ (`input_label`) de treinamento, quantidade de épocas (`epochs`) e a taxa de aprendizado (`learning_rate`):

https://github.com/JoseMVasconcelos/Rede-Neural-UFSC/blob/f340d43baea546c9b4ac736d6da840340680c327/notebooks/neural_lib/neural_network.py#L149

Para predição de uma observação, o método `predict` deve ser executado:

https://github.com/JoseMVasconcelos/Rede-Neural-UFSC/blob/f340d43baea546c9b4ac736d6da840340680c327/notebooks/neural_lib/neural_network.py#L168

## Treinamento de Modelos com a Neural Network Library

Foram treinados três modelos de predição para regressão, classificação binária e classificação multiclasse com a biblioteca desenvolvida. Os seguintes _datasets_ foram utilizados para os treinamentos:

- [`regression_fish.csv`](./data/regression_fish.csv) - Regressão
- [`customer-purchase-behavior.csv`](./data/customer-purchase-behavior.csv) - Classificação Binária
- [`multiclass_drug.csv`](./data/multiclass_drug.csv) - Classificação Multiclasse

Para cada modelo treinado, foi desenvolvido um Jupyter _notebook_. Eles encontram-se disponíveis no _folder_ `notebooks`.
