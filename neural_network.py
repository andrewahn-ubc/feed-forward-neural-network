import numpy as np

# The neural network class. Contains all the functions related to 
# a neural network instance, such as backpropagation.
class NeuralNetwork():
    # Sets up the anatomy of the neural network
    def __init__(self):
        # The neurons
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

    # Function: "Squishes" the domain of a vector from (-inf, inf) to something smaller and
    #           more manageable, like [0,1] for the sigmoid function. The actual new domain will 
    #           depend on which function I end up choosing (TBD).
    # Input: A vector.
    # Output: A vector. 
    # Usage: To be used during forward-propagation.
    def squishification(bigVector):
        # TODO: implement a squishification function (maybe sigmoid)
        return bigVector

    # Function: Propagates the input values "forward" through the neural network and changes the 
    #           activations in the hidden layers and the output layer.
    # Input: A single image - an array of size 784 containing doubles, and
    #        the target number that is drawn in the image.
    # Output: None 
    # Usage: To be used before performing backpropagation on a single image. 
    #        Will be called once per image for each gradient descent step.
    def forward(self, image):
        np.matmul(self.firstWeightMatrix, image, self.firstHiddenNeuronLayer)
        self.firstHiddenNeuronLayer += self.firstBiasVector
        self.firstHiddenNeuronLayer = self.squishification(self.firstHiddenNeuronLayer)

        np.matmul(self.secondWeightMatrix, self.firstHiddenNeuronLayer, self.secondHiddenNeuronLayer)
        self.secondHiddenNeuronLayer += self.secondBiasVector
        self.secondHiddenNeuronLayer = self.squishification(self.secondHiddenNeuronLayer)

        np.matmul(self.thirdWeightMatrix, self.secondHiddenNeuronLayer, self.outputNeuronLayer)
        self.outputNeuronLayer += self.thirdBiasVector
        self.outputNeuronLayer = self.squishification(self.outputNeuronLayer)

    # Function: The "heart" of the backpropagation algorithm, performed on a single training example
    # Input: A single image - an array of size 784 containing doubles, and
    #        the target number that is drawn in the image.
    # Output: The gradient vector for the single image - as an array of size 13,002 containing doubles.
    # Usage: To be called on every single training example by the main backpropagation function.
    def backpropOneImage(self, image, target):
        pass

    # Function: Facilitates backpropagation by calling the backpropOneImage function on 
    #           each training example, then averaging out and scaling the resulting gradient vectors
    #           to find the next "step" in the gradient descent.
    # Input: A python dictionary where the keys are the training images - arrays of size 784 containing doubles, and
    #        the values are the target numbers that are drawn in the corresponding training image.
    # Output: The gradient vector for the given set of images  - as an array of size 13,002 containing doubles.
    # Usage: To be called on a "batch" of training images to find the next gradient descent step.
    def backpropagation(self, trainingData):
        # gradientVectors is a 2D array, where each row will be a single gradient vector 
        gradientVectors = np.array([])

        # finding the gradient vector for each training example
        for image in trainingData:
            target = trainingData[image]
            gradientVectorForThisImage = self.backpropOneImage(image, target)
            gradientVectors = np.vstack([gradientVectors, gradientVectorForThisImage])

        # find the average gradient vector then scale it to find out overall gradient 
        averageGradient = np.mean(gradientVectors, axis=0)
        scaledAverageGradient = averageGradient * 1.0   # EDIT THIS based on the learning rate strategy

        return scaledAverageGradient
        
    # Function: Helper function to take in the output of backpropagation and convert it 
    #           to a form that is easier to work with when updating the parameters
    # Input: A gradient vector - as an array of size 13,002 containing doubles.
    # Output: A tuple containing the components of the gradient vector organized into 
    #         the appropriate matrices and vectors.
    # Usage: To be used after performing backprop on a set of training examples and having found the 
    #        average gradient vector. Would be called once for each gradient descent step.
    def paramVectorToTuple(self, gradientVector):
        pass 

    # Function: Helper function to apply the gradient descent step to the neural network by 
    #           adding the matrices and vectors in the tuple to the neural network 
    #           instance's matrices and vectors.
    # Input: A tuple containing the components of the gradient vector organized into the appropriate
    #        matrices and vectors.
    # Output: None
    # Usage: To be used after calling the paramVectorToTuple function. Would be called once
    #        for each gradient descent step.
    def applyGradientDescentStep(self, tuple):
        pass

    # Function: Perform stochastic gradient descent (SGD) by running backpropagation on training data 
    #           batches repeatedly until the resulting gradient vector is sufficiently small.
    #           (or some other condition has been satisfied for us to stop gradient descent - TBD).
    # Input: All of the training data - format TBD.
    # Output: None
    # Usage: First point of contact for training the neural network. Would only be called once for each
    #        time someone wants to train the NN. Manages training data batching, iterative backpropagation,
    #        and the stopping condition.
    def trainNeuralNetwork(self):
        pass

    # TODO: Write function signatures (and eventually implementations) for testing the neural network.

    