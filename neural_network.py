import numpy as np

# The neural network class. Contains all the functions related to 
# a neural network instance, such as backpropagation.
class NeuralNetwork():
    # Sets up the anatomy of the neural network
    def __init__(self):
        # The neurons
        self.inputNeuronLayer = np.random.random(784)
        self.firstHiddenNeuronLayer = np.random.random(16)
        self.secondHiddenNeuronLayer = np.random.random(16)
        self.outputNeuronLayer = np.random.random(10)

        # The weights
        self.firstWeightMatrix = np.random.random((16,784))
        self.secondWeightMatrix = np.random.random((16,16))
        self.thirdWeightMatrix = np.random.random((10,16))

        # The biases
        self.firstBiasVector = np.random.random(16)
        self.secondBiasVector = np.random.random(16)
        self.thirdBiasVector = np.random.random(10)

    # Function: The "heart" of the backpropagation algorithm, performed on a single training example
    # Input: a single image - an array of size 784 containing doubles, and
    #        the target number that is drawn in the image
    # Output: the gradient vector for the single image - as an array of size 13,002 containing doubles
    # Usage: To be called on every single training example by the main backpropagation function.
    def backpropOneImage(self, image, target):
        pass

    # Function: Facilitates backpropagation by calling the backpropOneImage function on 
    #           each training example, then averaging out and scaling the resulting gradient vectors
    #           to find the next "step" in the gradient descent.
    # Input: a python dictionary where the keys are the training images - arrays of size 784 containing doubles, and
    #        the values are the target numbers that are drawn in the corresponding training image
    # Output: the gradient vector for the given set of images  - as an array of size 13,002 containing doubles
    # Usage: To be called on a "batch" of training images to find the next gradient descent step.
    def backpropagation(self, trainingData):
        pass

    # Function: Helper function to take in the output of backpropagation and convert it 
    #           to a form that is easier to work with when updating the parameters
    # Input: a gradient vector - as an array of size 13,002 containing doubles
    # Output: a tuple containing the components of the gradient vector organized into 
    #         the appropriate matrices and vectors
    # Usage: To be used after performing backprop on a set of training examples and having found the 
    #        average gradient vector. Would be called once for each gradient descent step.
    def paramVectorToTuple(self, gradientVector):
        pass 

    # Function: Helper function to apply the gradient descent step to the neural network by 
    #           adding the matrices and vectors in the tuple to the neural network 
    #           instance's matrices and vectors.
    # Input: a tuple containing the components of the gradient vector organized into the appropriate
    #        matrices and vectors
    # Output: none
    # Usage: To be used after calling the paramVectorToTuple function. Would be called once
    #        for each gradient descent step.
    def applyGradientDescentStep(self, tuple):
        pass

    # Function: Perform stochastic gradient descent (SGD) by running backpropagation on training data 
    #           batches repeatedly until the resulting gradient vector is sufficiently small 
    #           (or some other condition has been satisfied for us to stop gradient descent - TBD).
    # Input: all of the training data - format TBD
    # Output: none
    # Usage: First point of contact for training the neural network. Would only be called once for each
    #        time someone wants to train the NN. Manages training data batching, iterative backpropagation,
    #        and the stopping condition
    def trainNeuralNetwork(self):
        pass

    # TODO: Write function signatures (and eventually implementations) for testing the neural network.

    