from neural_network import NeuralNetwork

# Main file that will interact with the neural network and the data.
# Does NOT contain any functionality that is inherently fundamental to the functionality of an NN 
# (ie. SGD, backprop, etc.). 
# Contains minimal functionality - mainly setting up the neural network and running it on training data.

def main():
    neuralNetwork = NeuralNetwork()
    # TODO: load MNIST dataset and split into training and testing datasets.

    data = [] # dummy training data for now. Will replace with training subset of MNIST dataset later.
    neuralNetwork.trainNeuralNetwork(data)

    # TODO: test the neural network using the testing dataset.
