# Multilayer Perceptron (MLP) model for regression:

from numpy import sqrt

import mlp_regression_classes_func
from mlp_regression_classes_func import MLP
from mlp_regression_classes_func import prepare_data
from mlp_regression_classes_func import train_model, evaluate_model, predict

# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
train_dl, test_dl = prepare_data(path)
print(f'Length of training data: {len(train_dl.dataset)}.')
print(f'Length of test data: {len(test_dl.dataset)}.')

# define the network
model = MLP(13)

# train the model
train_model(train_dl, model)

# evaluate the model
mse = evaluate_model(test_dl, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))

# make a single prediction (expect class=1)
row = [0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]
yhat = predict(row, model)
print('Predicted: %.3f' % yhat)
