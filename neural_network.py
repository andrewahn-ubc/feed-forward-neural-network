import numpy as np
import math

# The neural network class. Contains all the functions related to 
# a neural network instance, such as backpropagation.
class NeuralNetwork():
    def he_normal(self, shape, size_in, size_out):
        limit = np.sqrt(2/(size_in))
        out = np.random.randn(*shape) * limit
        return out

    # Sets up the anatomy of the neural network
    def __init__(self):
        # The neurons
        self.firstHiddenNeuronLayer = np.random.random(256)
        self.secondHiddenNeuronLayer = np.random.random(128)
        self.thirdHiddenNeuronLayer = np.random.random(64)
        self.fourthHiddenNeuronLayer = np.random.random(32)
        self.outputNeuronLayer = np.random.random(10)

        # The neurons pre-squish (this gives us access to z = a1w1 + a2w2 + ... + b)
        self.firstHiddenNeuronLayerPS = np.random.random(256)
        self.secondHiddenNeuronLayerPS = np.random.random(128)
        self.thirdHiddenNeuronLayerPS = np.random.random(64)
        self.fourthHiddenNeuronLayerPS = np.random.random(32)
        self.outputNeuronLayerPS = np.random.random(10)

        # The weights
        self.firstWeightMatrix = self.he_normal((256,784), 784, 256)
        self.secondWeightMatrix = self.he_normal((128,256), 256, 128)
        self.thirdWeightMatrix = self.he_normal((64,128), 128, 64)
        self.fourthWeightMatrix = self.he_normal((32,64), 64, 32)
        self.fifthWeightMatrix = self.he_normal((10,32), 32, 10)

        # The biases
        self.firstBiasVector = np.zeros(256)
        self.secondBiasVector = np.zeros(128)
        self.thirdBiasVector = np.zeros(64)
        self.fourthBiasVector = np.zeros(32)
        self.fifthBiasVector = np.zeros(10)

        # neural network specs
        self.numLayers = 5      # not including input layer
        self.alpha = 0.5      # TODO: update this alpha based on learning rate
        self.threshold = 0.05    # TODO: update this
        self.maxIterations = 600    # TODO: update this
        self.batchSize = 0.01    # a double representing batch sizes as a percentage of the original dataset size
        
        # hyperparameters for plateau LR scheduling
        self.bestLoss = np.inf
        self.wait = 0
        self.patience = 15
        self.factor = 0.9
        self.minAlpha = 0.01

        # best parameters so far, for Early Stopping with Best Model Checkpointing
        self.bestModel = (self.firstBiasVector,
                          self.firstWeightMatrix,
                          self.secondBiasVector,
                          self.secondWeightMatrix,
                          self.thirdBiasVector,
                          self.thirdWeightMatrix,
                          self.fourthBiasVector,
                          self.fourthWeightMatrix,
                          self.fifthBiasVector,
                          self.fifthWeightMatrix)

        # stats
        self.iterationsSoFar = 0

    def sigmoidVector(self, vector):
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        vectorizedSigmoid = np.vectorize(sigmoid)
        return vectorizedSigmoid(vector)

    def sigmoidDerivativeVector(self, vector):
        sigmoidDerivative = lambda x: (1/(1+ np.exp(-x))) * (1 - np.exp(-x))
        vectorized = np.vectorize(sigmoidDerivative)
        return vectorized(vector)
    
    def reluVector(self, vector):
        return np.maximum(0, vector)
    
    def reluDerivativeVector(self, vector):
        return np.where(vector > 0, 1, 0)
    
    def softmax(self, vector):
        exp = np.exp((vector) - np.max(vector))
        return exp / np.sum(exp)
    
    def crossEntropyCost(self, correct, predicted):
        return - np.sum(correct * np.log(predicted)) / len(correct)
    
    def squaredErrorCost(self, correct):
        return np.sum((self.outputNeuronLayer - correct) ** 2)/2
    
    def gradientClipping(self, vector, max_norm=2.0):
        total_norm = np.sqrt(np.sum(vector ** 2))
        factor = max_norm / max(total_norm, max_norm)
        return vector * factor

    # Function: Propagates the input values "forward" through the neural network and changes the 
    #           activations in the hidden layers and the output layer.
    # Input: A single image - an array of size 784 containing doubles, and
    #        the target number that is drawn in the image.
    # Output: None 
    # Usage: To be used before performing backpropagation on a single image. 
    #        Will be called once per image for each gradient descent step.
    def forward(self, image):
        self.firstHiddenNeuronLayer = self.firstWeightMatrix @ image
        self.firstHiddenNeuronLayerPS = self.firstHiddenNeuronLayer + self.firstBiasVector
        self.firstHiddenNeuronLayer = self.reluVector(self.firstHiddenNeuronLayerPS)

        self.secondHiddenNeuronLayer = self.secondWeightMatrix @ self.firstHiddenNeuronLayer
        self.secondHiddenNeuronLayerPS = self.secondHiddenNeuronLayer + self.secondBiasVector
        self.secondHiddenNeuronLayer = self.reluVector(self.secondHiddenNeuronLayerPS)

        self.outputNeuronLayer = self.thirdWeightMatrix @ self.secondHiddenNeuronLayer
        self.outputNeuronLayerPS = self.outputNeuronLayer + self.thirdBiasVector
        self.outputNeuronLayer = self.softmax(self.outputNeuronLayerPS)

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
        cost = self.crossEntropyCost(correct, self.outputNeuronLayer)

        # backprop for last layer
        error_lastLayer = (self.outputNeuronLayer - correct) / len(correct)
        biasesGradient_lastLayer = error_lastLayer
        weightsGradient_lastLayer = (self.secondHiddenNeuronLayer.reshape(-1,1) @ error_lastLayer.reshape(1,-1)).T
        weightsGradient_lastLayer = self.gradientClipping(weightsGradient_lastLayer)

        # backprop for 2nd hidden layer
        error_2ndHiddenLayer = (self.thirdWeightMatrix.T @ error_lastLayer)*self.reluDerivativeVector(self.secondHiddenNeuronLayerPS)
        biasesGradient_2ndHiddenLayer = error_2ndHiddenLayer
        weightsGradient_2ndHiddenLayer = (self.firstHiddenNeuronLayer.reshape(-1,1) @ error_2ndHiddenLayer.reshape(1,-1)).T
        weightsGradient_2ndHiddenLayer = self.gradientClipping(weightsGradient_2ndHiddenLayer)

        # backprop for 1st hidden layer 
        error_1stHiddenLayer = (self.secondWeightMatrix.T @ error_2ndHiddenLayer)*self.reluDerivativeVector(self.firstHiddenNeuronLayerPS)
        biasesGradient_1stHiddenLayer = error_1stHiddenLayer
        weightsGradient_1stHiddenLayer = (image.reshape(-1,1) @ error_1stHiddenLayer.reshape(1,-1)).T
        weightsGradient_1stHiddenLayer = self.gradientClipping(weightsGradient_1stHiddenLayer)

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

        # Plateau learning rate scheduling
        if (self.bestLoss > averageCost):
            self.bestLoss = averageCost
            self.wait = 0
            self.bestModel = (self.firstBiasVector,
                          self.firstWeightMatrix,
                          self.secondBiasVector,
                          self.secondWeightMatrix,
                          self.thirdBiasVector,
                          self.thirdWeightMatrix)
        else:
            self.wait += 1
        
        if (self.wait >= self.patience):
            self.alpha = max(self.minAlpha, self.alpha * self.factor)
            self.wait = 0
            print("updated the learning rate just now, ts pmo... new value: ", self.alpha)

        # update progression
        self.iterationsSoFar += 1

        print(f"backprop on batch {self.iterationsSoFar} complete, cost: ", averageCost, "    best cost so far: ", self.bestLoss)

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
                # print(self.bestModel)
                self.firstBiasVector = self.bestModel[0]
                self.firstWeightMatrix = self.bestModel[1]
                self.secondBiasVector = self.bestModel[2]
                self.secondWeightMatrix = self.bestModel[3]
                self.thirdBiasVector = self.bestModel[4]
                self.thirdWeightMatrix = self.bestModel[5]
                print("The model chosen has the following cost: ", self.bestLoss)
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
            y_pred = np.append(y_pred, prediction)

        y_pred = np.round(y_pred)
        
        return y_pred

    