# Multilayer Perceptron (MLP) model for binary classification:

import mlp_binary_classes_func
from mlp_binary_classes_func import MLP
from mlp_binary_classes_func import prepare_data
from mlp_binary_classes_func import train_model, evaluate_model, predict


# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prepare_data(path)
print(f'Length of training data: {len(train_dl.dataset)}.')
print(f'Length of test data: {len(test_dl.dataset)}.')

# define the network
model = MLP(34)

# train the model
train_model(train_dl, model)

# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy on test data: %.3f' % acc)

# make a single prediction (expect class=1)
row = [1, 0, 0.99539, -0.05889, 0.85243, 0.02306, 0.83398, -0.37708, 1, 0.03760, 0.85243, -0.17755, 0.59755, -0.44945,
       0.60536, -0.38223, 0.84356, -0.38542, 0.58212, -0.32192, 0.56971, -0.29674, 0.36946, -0.47357, 0.56811, -0.51171,
       0.41078, -0.46168, 0.21266, -0.34090, 0.42267, -0.54487, 0.18641, -0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
