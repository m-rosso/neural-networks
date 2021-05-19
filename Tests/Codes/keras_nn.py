####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import time

from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, brier_score_loss

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU, PReLU
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.activations import swish

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################
  
####################################################################################################################################
# Neural network estimation with evaluation on validation data:

class KerasNN(object):
    """
	This class provides an object consisting on a feedforward neural network constructed upon Tensorflow and Keras functions and
	objects. Its objective is to provide succinct codes for implementing neural networks estimation just providing model's
	architecture and its main hyper-parameters, besides training and validation data.

    Arguments for initialization:
    	'model_architecture': dictionary whose keys are hidden layers (integers) and values are dictionaries containing number of
    	neurons, activation function and dropout rate.
        'num_inputs': integer indicating the number of neurons in the input layer.
        'output_activation': function name (string) to be used as activation for neurons in the output layer.
        'cost_function': function name (string) for the cost function.
        'num_epochs': number of training epochs.
        'batch_size': integer given the length of batches of data points for stochastic gradient descent.
        'default_adam': boolean indicating whether to apply default Adam optimizer.
        'optimizer': string indicating which optimizer to use. Choose between 'sgd' and 'adam'.
        'opt_params': dictionary whose keys are hyper-parameters of the chosen optimizer.
        'regularization': string indicating which regularization method to apply. Choose between 'l1' and 'l2'.
        'regul_param': float for the L2 regularization.
        'input_dropout': dropout rate for the input layer.
        'weights_init': string consisting of shortcuts for Keras distributions for weights initialization.
        'bias_init': string consisting of shortcuts for Keras distributions for biases initialization.

    Methods:
    	'run': fit the neural network using training data and predict response variable for validation data.
    		Arguments:
    			'train_inputs': numpy's nd-array given a matrix for inputs from training data.
    			'train_output': numpy's nd-array given a vector for outputs from training data.
    			'val_inputs': numpy's nd-array given a matrix for inputs from validation data.
    			'val_output': numpy's nd-array given a vector for outputs from validation data.
    			'verbose': integer for the level of information to be printed during execution of 'run' method.
                'performance_callback': boolean indicating whether to collect outcomes at each epoch of training.
                'early_stopping': boolean indicating whether to implement early stopping.
                'es_params': dictionary with early stopping parameters, 'min_delta', 'patience', and 'consecutive_patience'.

    		Outputs:
    			Returns an estimated model. Using attribute 'predictions' returns an array with predictions for validation data.
                Using attribute 'model_costs' returns a dataframe with cost functions at each training epoch.
    """
    
    def __init__(self, model_architecture, num_inputs,
                 output_activation = 'sigmoid', cost_function = 'binary_crossentropy',
                 num_epochs = 10, batch_size = None,
                 default_adam = True, optimizer = None, opt_params = None,
                 regularization = 'l2', regul_param = 0, input_dropout = 0,
                 weights_init = 'glorot_uniform', bias_init = 'zeros'):
        self.model_architecture = model_architecture
        self.num_inputs = num_inputs
        self.output_activation = output_activation
        self.cost_function = cost_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.default_adam = default_adam
        self.optimizer = optimizer
        self.opt_params = opt_params
        self.regularization = regularization
        self.regul_param = regul_param
        self.input_dropout = input_dropout
        self.weights_init = weights_init
        self.bias_init = bias_init
        
        # Declaring the model object:
        self.model = Sequential()

        # Dropout for the input layer:
        self.model.add(Dropout(input_dropout, input_shape=(self.num_inputs,)))

        # Hidden layers with dropout:
        for i in self.model_architecture.keys():
            if self.model_architecture[i]['activation'] == 'leaky_relu':
                self.model.add(Dense(units = self.model_architecture[i]['neurons'],
                                     kernel_regularizer = eval('{0}(l = self.regul_param)'.format(self.regularization)),
                                     kernel_initializer = self.weights_init,
                                     bias_initializer = self.bias_init))
                self.model.add(LeakyReLU(alpha = 0.01))

            elif self.model_architecture[i]['activation'] == 'prelu':
                self.model.add(Dense(units = self.model_architecture[i]['neurons'],
                                     kernel_regularizer = eval('{0}(l = self.regul_param)'.format(self.regularization)),
                                     kernel_initializer = self.weights_init,
                                     bias_initializer = self.bias_init))
                self.model.add(PReLU(alpha_initializer = "zeros",
                                     alpha_regularizer = None, alpha_constraint = None, shared_axes = None))

            elif self.model_architecture[i]['activation'] in ['swish']:
                self.model.add(Dense(units = self.model_architecture[i]['neurons'],
                                     activation = eval(self.model_architecture[i]['activation']),
                                     kernel_regularizer = eval('{0}(l = self.regul_param)'.format(self.regularization)),
                                     kernel_initializer = self.weights_init,
                                     bias_initializer = self.bias_init))

            else:
                self.model.add(Dense(units = self.model_architecture[i]['neurons'],
                                     activation = self.model_architecture[i]['activation'],
                                     kernel_regularizer = eval('{0}(l = self.regul_param)'.format(self.regularization)),
                                     kernel_initializer = self.weights_init,
                                     bias_initializer = self.bias_init))

            self.model.add(Dropout(rate = self.model_architecture[i]['dropout_param']))

        # Final layer with one neuron:
        self.model.add(Dense(units = 1, activation = self.output_activation))

        # Compiling the model to prepare it to be fitted:
        if self.default_adam:
            self.model.compile(loss = self.cost_function, optimizer = 'adam')

        else:
            if self.optimizer == 'sgd':              
                opt = SGD(learning_rate = self.opt_params['learning_rate'], momentum = self.opt_params['momentum'],
                          decay = self.opt_params['decay'])

            elif self.optimizer == 'adam':
                opt = Adam(learning_rate = self.opt_params['learning_rate'], beta_1 = self.opt_params['beta_1'],
                           beta_2 = self.opt_params['beta_2'], epsilon = self.opt_params['epsilon'])

            self.model.compile(loss = self.cost_function, optimizer = opt)
        
    def run(self, train_inputs, train_output, val_inputs, val_output, test_inputs = None, test_output = None,
            verbose = 1, performance_callback = True, early_stopping = False,
            es_params = None, save_best_model = False):
        # Callback for assessing running time and performance metrics on validation data by epoch of training:
        class PerformanceHistory(Callback):
            def __init__(self, save_best_model = save_best_model):
                super(Callback, self).__init__()
                self.running_time = []
                self.epoch_val_roc_auc = []
                self.epoch_val_avg_prec_score = []
                self.epoch_val_brier_score = []
                self.save_best_model = save_best_model

            def on_epoch_begin(self, batch, logs={}):
                self.epoch_time_start = time.time()

            def on_epoch_end(self, batch, logs={}):
                self.running_time.append(time.time() - self.epoch_time_start)
                self.epoch_val_roc_auc.append(roc_auc_score(val_output, [p[0] for p in self.model.predict(val_inputs)]))
                self.epoch_val_avg_prec_score.append(average_precision_score(val_output,
                                                                             [p[0] for p in self.model.predict(val_inputs)]))
                self.epoch_val_brier_score.append(brier_score_loss(val_output, [p[0] for p in self.model.predict(val_inputs)]))

                if self.save_best_model:
                    if roc_auc_score(val_output, [p[0] for p in self.model.predict(val_inputs)]) == max(self.epoch_val_roc_auc):
                        self.best_model = self.model

        # Callback for implementing early stopping based on ROC-AUC:
        if early_stopping:
            class EarlyStoppingROC(Callback):
                def __init__(self, min_delta = 0, patience = 1, consecutive_patience = es_params['consecutive_patience']):
                    super(Callback, self).__init__()
                    self.min_delta = min_delta
                    self.patience = patience
                    self.consecutive_patience = consecutive_patience
                    self.previous = 0
                    self.P_max = 0

                def on_epoch_end(self, epoch, logs={}):
                    score = roc_auc_score(val_output, [p[0] for p in self.model.predict(val_inputs)])
                    
                    # Checking whether an improvement in performance has been achieved:
                    if score - self.previous > self.min_delta:
                        if self.consecutive_patience:
                            self.P_max = 0

                        else:
                            pass
                        
                    else:
                        self.P_max += 1
                        
                    # Checking whether improvement has not occur for more than self.patience epochs of training:
                    if self.P_max > self.patience:
                        self.model.stop_training = True
                    
                    self.previous = score

        # List of callbacks:
        callbacks = []

        if performance_callback:
            performance_callback = PerformanceHistory()
            callbacks.append(performance_callback)

        if early_stopping:
            early_stop = EarlyStoppingROC(min_delta = es_params['min_delta'], patience = es_params['patience'])
            callbacks.append(early_stop)

        if (performance_callback == False) & (early_stopping == False):
            callbacks = None

        # Training the model:
        self.model.fit(x = train_inputs, 
                       y = train_output,
                       validation_data = (val_inputs, val_output),
                       epochs = self.num_epochs,
                       batch_size = self.batch_size,
                       shuffle = False,
                       callbacks = callbacks,
                       verbose = verbose
                       )

         # Cost function by training epoch:
        self.model_costs = pd.DataFrame(self.model.history.history)
        self.model_costs['epoch'] = [i + 1 for i in self.model_costs.index]
        
        # Predicting response for validation data:
        if test_inputs is not None:
            self.predictions = {'val': self.model.predict(val_inputs), 'test': self.model.predict(test_inputs)}

        else:
            self.predictions = self.model.predict(val_inputs)

        # Running time and performance metrics on validation data by epoch of training:
        self.epoch_performance = {
            'running_time': performance_callback.running_time,
            'epoch_val_roc_auc': performance_callback.epoch_val_roc_auc,
            'epoch_val_avg_prec_score': performance_callback.epoch_val_avg_prec_score,
            'epoch_val_brier_score': performance_callback.epoch_val_brier_score
        }

        if save_best_model:
            self.best_model = performance_callback.best_model
