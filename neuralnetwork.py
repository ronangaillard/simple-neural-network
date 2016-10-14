import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, output_size, num_of_neurons):
        self.input_size = input_size
        self.output_size = output_size
        self.num_of_neurons = num_of_neurons

        # Left part weights
        self.W1 = np.random.rand(self.input_size, self.num_of_neurons)

        # Right part weights
        self.W2 = np.random.rand(self.num_of_neurons, self.output_size)

    # Computes the neural network output given the input 
    def process(self, input):
        Z2 = np.dot(input, self.W1)
        a2 = self.neural_function(Z2)
        Z3 = np.dot(a2, self.W2)
        return self.neural_function(Z3)

    # Sigmoid function
    def neural_function(self, x):
        return 1 / ( 1 + np.exp(-x) )