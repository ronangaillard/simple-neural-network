import numpy as np
from neuralnetwork import NeuralNetwork
import matplotlib.pyplot as plt

scalar = 1

nn = NeuralNetwork(2, 1, 3)

custom_input = np.matrix([ [3, 5], [5, 1], [10, 2] ] , dtype=float)
expected_output = np.matrix( [ [75], [82], [93]], dtype=float)

# Normalise data
custom_input = custom_input / np.amax(custom_input, axis=0)
expected_output = expected_output / 100

cost_array = []

for i in range(0,1000):

    cost = nn.cost(custom_input, expected_output)
    gradientW1, gradientW2 = nn.compute_gradient(custom_input, expected_output)

    #print "Cost : ", cost
    cost_array.append(cost)

    nn.W1 += gradientW1 * scalar
    nn.W2 += gradientW2 * scalar

output = nn.process(custom_input)
print "Output :", output*100

plt.plot(cost_array)
plt.show()