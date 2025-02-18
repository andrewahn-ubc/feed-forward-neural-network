from neural_network import NeuralNetwork
import numpy as np

# Main file that will interact with the neural network and the data.
# Does NOT contain any functionality that is inherently fundamental to the functionality of an NN 
# (ie. SGD, backprop, etc.). 
# Contains minimal functionality - mainly setting up the neural network and running it on training data.

def main():
    # TODO: load MNIST dataset and split into training and testing datasets.
    # TODO: test the neural network using the testing dataset.

    # Prep dummy data 
    X_train = np.random.random((105, 784))  # training data size will be a multiple of the batch size so some of these might not get used
    y_train = np.ones(105)
    X_test = np.random.random((12, 784))
    y_test = np.ones(12)

    model = NeuralNetwork()
    model.fit(X_train, y_train)
    print("done training!")

    y_pred = model.predict(X_test)
    print(y_pred)

    # TODO: return the accuracy


if __name__ == "__main__":
    main()

