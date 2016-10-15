import numpy as np
from neuralnetwork import NeuralNetwork

scalar = 3

nn = NeuralNetwork(2, 1, 3)

custom_input = np.matrix([ [3, 10], [2, 3], [7, 8] ])
expected_output = np.matrix( [ [4], [1], [2]])

output = nn.process(custom_input)

for i in range(0,40):

    cost = nn.cost(custom_input, expected_output)
    gradientW1, gradientW2 = nn.compute_gradient(custom_input, expected_output)

    print "Cost : ", cost

    nn.W1 += gradientW1 * scalar
    nn.W2 += gradientW2 * scalar