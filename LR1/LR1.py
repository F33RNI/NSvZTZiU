"""
 Licensed under the Unlicense License;
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://unlicense.org

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np
import cv2


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define training array
training_inputs = []
training_outputs = [1, 7, 3, 2, 5, 4, 8, 6, 9, 0]
for i in range(0, 10):
    train_binary = cv2.bitwise_not(cv2.imread('train_values/' + str(i) + '.bmp', cv2.IMREAD_UNCHANGED)).flatten()
    train_binary[train_binary == 255] = 1
    training_inputs.append(train_binary)
training_inputs = np.array(training_inputs)
training_outputs = np.array([training_outputs])
training_outputs = training_outputs / 9
training_outputs = training_outputs.T

print('Training inputs: ')
print(training_inputs)  # 0 - 1
print()
print('Training outputs: ')
print(training_outputs)  # 0 - 1
print()

# Define random weights
np.random.seed(1)
synaptic_weights = 2 * np.random.random((10, 1)) - 1
print('Synaptic weights: ')
print(synaptic_weights)
print()
exit()


# Predict value
def think(inputs):
    global synaptic_weights
    inputs = inputs.astype(float)
    """
    print('Inputs: ')
    print(inputs[0])
    print('Weights: ')
    print(synaptic_weights)
    print('Weights sum : ')
    print(synaptic_weights[0] + synaptic_weights[1] + synaptic_weights[2] + synaptic_weights[3] + synaptic_weights[4]
          + synaptic_weights[5] + synaptic_weights[6] + synaptic_weights[7] + synaptic_weights[8] + synaptic_weights[9])
    print('NP.DOT: ')
    print(np.dot(inputs, synaptic_weights))
    print('Activation: ')
    print(sigmoid(np.dot(inputs, synaptic_weights)))
    #exit(0)
    """

    out_value = sigmoid(np.dot(inputs, synaptic_weights))
    return out_value


# Training function
for i in range(3000):
    output = think(training_inputs)
    error = training_outputs - output
    # Backpropagation
    adjustments = np.dot(training_inputs.T, error * (output * (1 - output)))
    synaptic_weights += adjustments

    if i == 2000:
        print('Output: ')
        print(output)
        print('Error: ')
        print(error)
        print('Adjustments : ')
        print(adjustments)
        print('Synaptic_weights: ')
        print(synaptic_weights)

    if i % 10 == 0:
        print('Iteration ' + str(i) + ' Predicted values: ' + str((np.around(think(training_inputs).flatten() * 9))
                                                                  .astype(int)))

print()
print('-------------')
print('Predicted values: ')
print((np.around(think(training_inputs).flatten() * 9)).astype(int))
print()
print('Synaptic weights: ')
print(synaptic_weights)
print('-------------')
