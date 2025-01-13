import numpy as np

class Neuron:
    def __init__ (self, weights, bias, func):
        self.weights = weights
        self.bias = bias
        self.func = func
        
    def run(self, inputs):
        # Asegúrate de que las entradas sean también un array numpy
        inputs = np.array(inputs)
        
        # Cálculo de la suma ponderada
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        
        # Aplicar la función de activación
        if self.func == "ReLU":
            return max(0, weighted_sum)
        elif self.func == "Sigmoide":
            return 1 / (1 + np.exp(-weighted_sum))
        elif self.func == "Tangente Hiperbólica":
            return np.tanh(weighted_sum)
        elif self.func == "Binary Step":
            return 1 if weighted_sum > 0 else 0  # Binary Step activación
        else:
            return "Error: Función de activación no válida"

    def changeBias(self, bias):
        self.bias = bias