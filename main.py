import numpy as np
from neuralnetwork import NeuralNetwork

nn = NeuralNetwork(2, 1, 3)

custom_input = np.matrix([ [3, 10], [2, 3], [7, 8] ])
output = nn.process(custom_input)

print "Input :", custom_input
print "Output :", output
