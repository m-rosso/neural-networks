# Multilayer Perceptron (MLP) model for multi-class classification:

from numpy import argmax

import mlp_multi_classes_func
from mlp_multi_classes_func import MLP
from mlp_multi_classes_func import prepare_data
from mlp_multi_classes_func import train_model, evaluate_model, predict

# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
train_dl, test_dl = prepare_data(path)
print(f'Length of training data: {len(train_dl.dataset)}.')
print(f'Length of test data: {len(test_dl.dataset)}.')

# define the network
model = MLP(4)

# train the model
train_model(train_dl, model)

# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy on test data: %.3f' % acc)

# make a single prediction
row = [5.1, 3.5, 1.4, 0.2]
yhat = predict(row, model)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
