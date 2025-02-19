import numpy as np
import math

# The neural network class. Contains all the functions related to 
# a neural network instance, such as backpropagation.
class NeuralNetwork():
    # Sets up the anatomy of the neural network
    def __init__(self):
        # The neurons
        self.firstHiddenNeuronLayer = np.random.random(16)
        self.secondHiddenNeuronLayer = np.random.random(16)
        self.outputNeuronLayer = np.random.random(10)

        # The neurons pre-squish (this gives us access to z = a1w1 + a2w2 + ... + b)
        self.firstHiddenNeuronLayerPS = np.random.random(16)
        self.secondHiddenNeuronLayerPS = np.random.random(16)
        self.outputNeuronLayerPS = np.random.random(10)

        # The weights
        self.firstWeightMatrix = np.random.random((16,784))
        self.secondWeightMatrix = np.random.random((16,16))
        self.thirdWeightMatrix = np.random.random((10,16))

        # The biases
        self.firstBiasVector = np.random.random(16)
        self.secondBiasVector = np.random.random(16)
        self.thirdBiasVector = np.random.random(10)

        # neural network specs
        self.numLayers = 3      # not including input layer
        self.alpha = 3        # TODO: update this alpha based on learning rate
        self.threshold = 1.0    # TODO: update this
        self.maxIterations = 200    # TODO: update this
        self.batchSize = 0.005    # a double representing batch sizes as a percentage of the original dataset size

        # stats
        self.iterationsSoFar = 0

    # Function: "Squishes" the domain of a vector from (-inf, inf) to [0,1] using the sigmoid function. 
    # Input: A vector.
    # Output: A vector. 
    # Usage: To be used during forward-propagation.
    def sigmoidVector(self, vector):
        sigmoid = lambda x: 1/(1 + math.exp(-x))
        vectorizedSigmoid = np.vectorize(sigmoid)
        return vectorizedSigmoid(vector)

    # Function: The derivate of the sigmoid function for a vector.
    def sigmoidDerivativeVector(self, vector):
        sigmoidDerivative = lambda x: math.exp(-x)/((1 + math.exp(-x)) ** 2)
        vectorized = np.vectorize(sigmoidDerivative)
        return vectorized(vector)

    # Function: Propagates the input values "forward" through the neural network and changes the 
    #           activations in the hidden layers and the output layer.
    # Input: A single image - an array of size 784 containing doubles, and
    #        the target number that is drawn in the image.
    # Output: None 
    # Usage: To be used before performing backpropagation on a single image. 
    #        Will be called once per image for each gradient descent step.
    def forward(self, image):
        self.firstHiddenNeuronLayer = self.firstWeightMatrix @ image
        self.firstHiddenNeuronLayer += self.firstBiasVector
        self.firstHiddenNeuronLayerPS = self.firstHiddenNeuronLayer
        self.firstHiddenNeuronLayer = self.sigmoidVector(self.firstHiddenNeuronLayer)

        self.secondHiddenNeuronLayer = self.secondWeightMatrix @ self.firstHiddenNeuronLayer
        self.secondHiddenNeuronLayer += self.secondBiasVector
        self.secondHiddenNeuronLayerPS = self.secondHiddenNeuronLayer
        self.secondHiddenNeuronLayer = self.sigmoidVector(self.secondHiddenNeuronLayer)

        self.outputNeuronLayer = self.thirdWeightMatrix @ self.secondHiddenNeuronLayer
        self.outputNeuronLayer += self.thirdBiasVector
        self.outputNeuronLayerPS = self.outputNeuronLayer
        self.outputNeuronLayer = self.sigmoidVector(self.outputNeuronLayer)

    # Function: Return the average value of the cost function (aka objective function) for the current image
    #           cost() = (1/2) Sum{10}_{i = 1} (aL_i - y_i)^2
    # Input: A 1D array containing the correct output for the neural network for the current image
    # Output: A double
    # Usage: To be used during backpropOneImage. Will eventually be used for the stopping condition in gradient descent.
    def cost(self, correct):
        return np.sum((self.outputNeuronLayer - correct) ** 2)/2

    # Function: The "heart" of the backpropagation algorithm, performed on a single training example
    # Input: A single image - an array of size 784 containing doubles, and
    #        the target integer in [0,9] that is drawn in the image.
    # Output: The gradient vector for the single image - as an array of size 13,002 containing doubles.
    # Usage: To be called on every single training example by the main backpropagation function.
    def backpropOneImage(self, image, target):
        # find the activations for the image
        self.forward(image)

        # correct output
        correct = np.zeros(10)
        target = math.floor(target)     # cast the target number to an integer
        correct[target] = 1

        # computer cost for this image
        cost = self.cost(correct)

        # backprop for last layer
        error_lastLayer = (self.outputNeuronLayer - correct)*self.sigmoidDerivativeVector(self.outputNeuronLayerPS)
        biasesGradient_lastLayer = error_lastLayer
        weightsGradient_lastLayer = (self.secondHiddenNeuronLayer.reshape(-1,1) @ error_lastLayer.reshape(1,-1)).T

        # backprop for 2nd hidden layer
        error_2ndHiddenLayer = (self.thirdWeightMatrix.T @ error_lastLayer)*self.sigmoidDerivativeVector(self.secondHiddenNeuronLayerPS)
        biasesGradient_2ndHiddenLayer = error_2ndHiddenLayer
        weightsGradient_2ndHiddenLayer = (self.firstHiddenNeuronLayer.reshape(-1,1) @ error_2ndHiddenLayer.reshape(1,-1)).T

        # backprop for 1st hidden layer 
        error_1stHiddenLayer = (self.secondWeightMatrix.T @ error_2ndHiddenLayer)*self.sigmoidDerivativeVector(self.firstHiddenNeuronLayerPS)
        biasesGradient_1stHiddenLayer = error_1stHiddenLayer
        weightsGradient_1stHiddenLayer = (image.reshape(-1,1) @ error_1stHiddenLayer.reshape(1,-1)).T

        return (biasesGradient_1stHiddenLayer, 
                weightsGradient_1stHiddenLayer, 
                biasesGradient_2ndHiddenLayer, 
                weightsGradient_2ndHiddenLayer, 
                biasesGradient_lastLayer,
                weightsGradient_lastLayer), cost


    # Function: Facilitates backpropagation by calling the backpropOneImage function on 
    #           each training example, then averaging out and scaling the resulting gradient vectors
    #           to find the next "step" in the gradient descent.
    # Input: a 2D array where each row is an image (an array of size 784 containing doubles), and
    #        a 1D array where each value is the target number of the corresponding training image.
    # Output: The gradient vector for the given set of images  - as an array of size 13,002 containing doubles, and 
    #         the average cost of this batch of images
    # Usage: To be called on a "batch" of training images to find the next gradient descent step.
    def backpropagation(self, X, y):
        # gradientVectors is a 2D array of gradient vectors and matrices
        # the even rows are bias gradients and the odd rows are weights gradients, with the first row associated with the first hidden layer
        # there will be n columns, where n is the number of images
        costs = np.array([])

        firstBiasGradient = np.empty((0,16))
        secondBiasGradient = np.empty((0,16))
        thirdBiasGradient = np.empty((0,10))
        firstWeightsGradient = np.empty((0,16,784))
        secondWeightsGradient = np.empty((0,16,16))
        thirdWeightsGradient = np.empty((0,10,16))

        if (len(X) != len(y)):
            print("Error: length of X and y are different. X is of length ", len(X), " and y is of length ", len(y), ".")

        # finding the gradient vector for each training example
        for i in range(len(X)):
            gradientTupleForThisImage, costForOneImage = self.backpropOneImage(X[i], y[i])
            costs = np.append(costs, costForOneImage)

            # just initialize an array for each of the gradient vectors and find the mean of them individually
            # then return a tuple of the mean vectors and matrices
            firstBiasGradient = np.vstack((firstBiasGradient, gradientTupleForThisImage[0]))
            firstWeightsGradient = np.insert(firstWeightsGradient, 0, gradientTupleForThisImage[1], axis=0) 
            secondBiasGradient = np.vstack((secondBiasGradient, gradientTupleForThisImage[2]))
            secondWeightsGradient = np.insert(secondWeightsGradient, 0, gradientTupleForThisImage[3], axis=0)
            thirdBiasGradient = np.vstack((thirdBiasGradient, gradientTupleForThisImage[4]))
            thirdWeightsGradient = np.insert(thirdWeightsGradient, 0, gradientTupleForThisImage[5], axis=0)
            
        # array containing the average gradient matrices and vectors
        avgFirstBiasGrad = np.mean(firstBiasGradient, axis=0)
        avgFirstWeightsGrad = np.mean(firstWeightsGradient, axis=0)
        avgSecondBiasGrad = np.mean(secondBiasGradient, axis=0)
        avgSecondWeightsGrad = np.mean(secondWeightsGradient, axis=0)
        avgThirdBiasGrad = np.mean(thirdBiasGradient, axis=0)
        avgThirdWeightsGrad = np.mean(thirdWeightsGradient, axis=0)

        # compute average cost 
        averageCost = np.mean(costs)

        # update progression
        self.iterationsSoFar += 1

        print(f"backprop on batch {self.iterationsSoFar} complete, cost: ", averageCost)

        # scale learning rate based on the value of the cost function 
        # if (averageCost > 3.5):
        #     self.alpha = 1.5
        # elif (averageCost > 2.5):
        #     self.alpha = 0.35 * averageCost
        # else:
        #     self.alpha = 0.3 * averageCost

        return (avgFirstBiasGrad,
                avgFirstWeightsGrad,
                avgSecondBiasGrad,
                avgSecondWeightsGrad,
                avgThirdBiasGrad,
                avgThirdWeightsGrad), averageCost
        
    # Function: Helper function to apply the gradient descent step to the neural network by 
    #           adding the matrices and vectors in the tuple to the neural network 
    #           instance's matrices and vectors.
    # Input: An array containing the components of the gradient vector organized into the appropriate
    #        matrices and vectors.
    # Output: None
    # Usage: To be used after calling the backpropagation function. Would be called once
    #        for each gradient descent step.
    def applyGradientDescentStep(self, gradientVectors):
        # The weights
        self.firstWeightMatrix -= gradientVectors[1] * self.alpha
        self.secondWeightMatrix -= gradientVectors[3] * self.alpha
        self.thirdWeightMatrix -= gradientVectors[5] * self.alpha

        # The biases
        self.firstBiasVector -= gradientVectors[0] * self.alpha
        self.secondBiasVector -= gradientVectors[2] * self.alpha
        self.thirdBiasVector -= gradientVectors[4] * self.alpha

    # Function: Perform stochastic gradient descent (SGD) by running backpropagation on training data 
    #           batches repeatedly until the resulting gradient vector is sufficiently small.
    #           (or some other condition has been satisfied for us to stop gradient descent - TBD).
    # Input: All of the training data - a 1D array of 2D arrays where each 2D array is a batch and each row is an image, and
    #        a 2D array where each row is the correct target numbers for the corresponding batch in X
    # Output: None
    # Usage: Iteratively applies backpropagation and updates parameters until cost is below a certain threshold or 
    #        max number of iterations has been reached. Manages iterative backpropagation and the stopping conditions.
    def gradientDescent(self, X, y):
        if (len(X) != len(y)):
            print("Error: the number of batches in X and y are different. X has ", len(X), " batches and y has ", len(y), ".")

        counter = 0
        numIterations = 0

        while (True):
            # apply backprop to a batch
            gradient, cost = self.backpropagation(X[counter], y[counter])

            # update parameters
            self.applyGradientDescentStep(gradient)

            # check break conditions (either cost is low enough or we've reached the max number of iterations)
            if (self.threshold >= cost or numIterations >= self.maxIterations):
                break

            # update counter (mod by number of batches) and number of iterations 
            counter = (counter + 1) % len(X)
            numIterations += 1
        
        print("Stochastic gradient descent complete!")


    # Function: Prepare data for SGD and initiate it
    # Input: a 2D array where each row is a training image of length 784, and
    #        a 1D array containing the corresponding target number for each image
    # Output: None
    # Usage: First point of contact for training the neural network. Would only be called once for each
    #        time someone wants to train the NN. Manages training data batching.
    def fit(self, X, y):
        # batch training data for SGD
        if (len(X) != len(y)):
            print("Error: the original size of X and y are different. X is length ", len(X), " and y is length ", len(y), ".")

        batchSize = math.floor(len(X) * self.batchSize)
        index = 0

        # Initialize the batches with the correct shape 
        X_batches = np.empty((0,batchSize,784))
        y_batches = np.empty((0,batchSize))
        # NOTE: if you want to initialize an empty array but still want to encode the correct shape, 
        #       you should use np.empty() with 0 in the correct spot. 
        #       np.array will assume that your shape dimensions are array elements.

        while (index + batchSize <= len(X) - 1):
            start = index
            end = index + batchSize

            X_batch = np.array(X[start:end, :])
            y_batch = np.array(y[start:end])
            X_batches = np.insert(X_batches, 0, X_batch, axis=0) # it doesn't matter where we insert here
            y_batches = np.insert(y_batches, 0, y_batch, axis=0)

            index += batchSize

        # Let's get it started!!!!
        self.gradientDescent(X_batches, y_batches)

    # Function: Make predictions for a set of images!
    # Input: An array of training images
    # Output: An array of integers - the predictions for the corresponding image
    # Usage: For testing and for actual usage if anyone decides to use my NN
    def predict(self, X_pred):
        # initialize
        y_pred = np.array([])

        for i in range(len(X_pred)):
            self.forward(X_pred[i])
            # output neuron with the highest activation is the prediction
            prediction = np.argmax(self.outputNeuronLayer)
            print("output layer: ", self.outputNeuronLayer)
            print("prediction: ", prediction)
            y_pred = np.append(y_pred, prediction)

        y_pred = np.round(y_pred)
        
        return y_pred

    