from neural_network import NeuralNetwork
import numpy as np
from os.path import join
from data_loader import MnistDataloader

# Main file that will interact with the neural network and the data.
# Does NOT contain any functionality that is inherently fundamental to the functionality of an NN 
# (ie. SGD, backprop, etc.). 
# Contains minimal functionality - mainly setting up the neural network and running it on training data.

def load_data():
    #
    # Set file paths based on added MNIST Datasets
    #
    input_path = './data'
    # input_path = '/home/ubuntu/feed-forward-neural-network/data'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    #
    # Load MINST dataset
    #
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    return x_train, y_train, x_test, y_test

def main():
    # load MNIST dataset and split into training and testing datasets.
    X_train, y_train, X_test, y_test = load_data()

    X_train = np.array(X_train)     # shape (60000, 784)
    y_train = np.array(y_train)     # shape (60000,)
    X_test = np.array(X_test)       # shape (10000, 784)
    y_test = np.array(y_test)       # shape (10000,)

    model = NeuralNetwork()
    model.fit(X_train, y_train)
    print("done training!")

    y_pred = model.predict(X_test)
    print(y_pred)
    accuracy = 1 - np.mean(y_pred != y_test)

    print("Test accuracy: ", accuracy)


if __name__ == "__main__":
    main()