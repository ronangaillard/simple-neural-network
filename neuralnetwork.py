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
        self.Z2 = np.dot(input, self.W1)
        a2 = self.neural_function(self.Z2)
        self.Z3 = np.dot(a2, self.W2)
        return self.neural_function(self.Z3)

    # Sigmoid function
    def neural_function(self, x):
        return 1 / ( 1 + np.exp(-x) )

    # Derivative of the sigmoid function
    def neural_function_prime(self, x):
        return np.exp( -x ) / ( np.square( 1 + np.exp( -x ) ) )

    def compute_gradient(self, input, expected_output):
        output = self.process(input)
        a2 = self.neural_function(self.Z2)
        sigma3 = -(output - expected_output)
        sigma3 = np.multiply(sigma3 , self.neural_function_prime(self.Z3) )
        dJdW2 = np.dot(np.transpose(a2), sigma3)

        sigma2 = np.dot(sigma3, np.transpose(self.W2)) * self.neural_function_prime(self.Z2)
        dJdW1 = np.dot(np.transpose(input), sigma2)

        return dJdW1, dJdW2

    # Cost function
    def cost(self, input, expected_output):
        return 0.5 * np.sum( np.square( self.process(input) - expected_output ) )

