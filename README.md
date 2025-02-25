> "What I cannot create, I do not understand" -Richard Feynman

# Feed-Forward Neural Network

A feed-forward neural network built from scratch (ie. using only numpy, no tensorflow or pytorch) as an exercise to learn machine learning.

## Background

There are a lot of tutorials on how to build an NN from scratch, but I challenged myself to come up with my own implementation. I watched [3B1B's series on neural networks](https://youtu.be/aircAruvnKk?si=3YaX6TYLx1CXsgmj) and referenced [Michael Nielson's online textbook](http://neuralnetworksanddeeplearning.com/) to get a theoretical understanding of NNs.

## Implementation Details

At the moment, hyperparameters such as the number of layers, the number of neurons in each layer, and activation functions are hardcoded. I might parameterize it later, but my main goal for this project was to learn deep learning, so I'm content with where I am right now. I tested the model on the classic Optical Character Recognition problem using the MNIST hand-written digits dataset and was able to achieve 88.27% test accuracy using the following model architecture:

- Input layer: 784 neurons
- 1st hidden layer: 256 neurons, ReLu
- 2nd hidden layer: 128 neurons, ReLu
- 3rd hidden layer: 64 neurons, ReLu
- 4th hidden layer: 32 neurons, ReLu
- Output layer: 10 neurons, Softmax
- Loss function: Categorical Cross-Entropy
- Optimizer: Stochastic Gradient Descent
- Other hyper-parameters:
    - Batch size: 0.3% of training set
    - Epochs: 2
    - Learning rate scheduling: Reduce on Plateau

 ## Features
- Layers
    - Dense
- Optimizers
    - Stochastic Gradient Descent (mini-batch)
- Initializers
    - Random Uniform
    - Xavier Normal
    - He Normal
- Activation Functions
    - ReLU
    - Sigmoid
    - Softmax
- Loss Functions
    - Mean Squared Error
    - Categorical Cross Entropy
- Learning rate scheduling
    - Step decay
    - Exponential decay
    - Reduce on Plateau
- Other optimization techniques
    - gradient clipping
    - early stopping with best model checkpointing

## Goals

- Once the implementation is done, test it on the MNIST hand-written digits dataset and get a relatively high accuracy (anything >80%)
    - Update: DONE, achieved 88.27% test accuracy
    - ![ffnn_perf2](https://github.com/user-attachments/assets/57211abd-9dd5-439c-b024-6172f963cf0f)


- Explore some fun deep learning techniques (learning rate schedules, early stopping, different gradient descent variations, hyperparameter tuning, different initialization methods, etc.).
    - Update: DONE, tried various LR schedules, early stopping methods, initialization methods, activation functions, and model architectures. Did not implement any other optimizers though (like Adam)
 
## Next Steps
- Refactor model architecture to have flexible number of layers and neurons, and for the user to be able to select features using the CLI
- Implement Adam optimizer, SGD with momentum, and Dropout regularization
- (reach) build a web app to demo the model
