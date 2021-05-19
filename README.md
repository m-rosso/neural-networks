## Neural networks

This repository contains early studies concerning neural networks. Three main topics are present here: folder "Tests" has codes, datasets, and discussions over empirical applications of feedforward neural networks; folder "PyTorch Tutorial" contains codes for applying PyTorch library; finally, folder "Tutorial" presents a Jupyter notebook that makes use of a Python class, named *KerasNN*, developed to simplify even further the use of Tensorflow/Keras library to implement data modeling based on neural networks.
<br>
<br>
The class *KerasNN* provides an object consisting of a feedforward neural network constructed upon Tensorflow and Keras functions and classes. Its objective is to provide succinct codes for implementing neural networks by simply providing model's architecture and its main hyper-parameters, besides training, validation, and test data. Full documentation of this class can be found in Python module "keras_nn.py", but some of the components that can be declared when initializing a KerasNN object is number of layers, number of neurons, activation function of each layer, dropout and regularization parameters, optimization strategy (SGD, Adam) and its parameters (learning rate, decay, momentum), early stopping configuration.
<br>
<br>
In addition to its straightforward use, KerasNN class has some relevant attributes after fitting a model. Together with all attributes that follow from Keras model, costs and performance metrics by epoch of training are available, helping with the understanding of how the network evolves throughout its training process.
