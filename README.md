# Deep-Neural-MLP-Network
Multi-Layer Perceptron implemented to predict labels of the Cifar data set of handwritten digits. 
It is way more stable than the shallow network uploaded and is robust for higher learning rates and way deeper networks, even if this specific implementation shallow. 

Some applied attributes are
* ReLu activation function,
* batch normalization with moving averages,
* momentum,
* decaying learning rate,
* regularization,
* mini batch gradient decent,
* early stopping.

The Cifar 10 pyhon data set should be downloaded through http://www.cs.toronto.edu/~kriz/cifar.html and saved in folder cifar-10-python\cifar-10-batches-py for the program to work.
