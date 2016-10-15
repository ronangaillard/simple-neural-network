import numpy as np
from neuralnetwork import NeuralNetwork

nn = NeuralNetwork(2, 1, 3)

custom_input = np.matrix([ [3, 10], [2, 3], [7, 8] ])
expected_output = np.matrix( [ [2], [1], [4]])

output = nn.process(custom_input)

print "Input :", custom_input
print "Output :", output

gradient = nn.compute_gradient(custom_input, expected_output)

print "Gradient :", gradient