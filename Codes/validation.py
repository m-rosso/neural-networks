####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################
import pandas as pd
import numpy as np
import os
import json

from datetime import datetime
import time

import progressbar
from time import sleep

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, brier_score_loss

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso

# pip install lightgbm
import lightgbm as lgb

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

####################################################################################################################################
# K-folds cross-validation for grid-search:

class KfoldsCV(object):
    """
    Arguments for initialization:
    Methods:
    Output objects:
    """
    def __init__(self, task = 'classification', method = 'logistic_regression',
                 metric = 'roc_auc', num_folds = 3,
                 pre_selecting = False, pre_selecting_param = None,
                 random_search = False, n_samples = None,
                 grid_param = None, default_param = None,
                 cost_function = None):
        self.task = task
        self.method = str(method)
        self.metric = metric
        self.num_folds = int(num_folds)
        self.default_param = default_param
        self.pre_selecting = pre_selecting
        self.pre_selecting_param = pre_selecting_param
        self.random_search = random_search
        self.n_samples = n_samples
        self.cost_function = cost_function
        
        # Constructing a grid of different combinations of hyper-parameters using alternatives declared during
        # initialization:
        if self.random_search is not True:
            list_param = [grid_param[k] for k in grid_param.keys()]
            list_param = [list(x) for x in np.array(np.meshgrid(*list_param)).T.reshape(-1,len(list_param))]
            self.grid_param = []
            for i in list_param:
                self.grid_param.append(dict(zip(grid_param.keys(), i)))
        
        # Constructing a grid of different combinations of hyper-parameters using probability distributions
        # declared during initialization:
        else:
            self.grid_param = []

            for i in range(1, self.n_samples+1):
                list_param = []

                for k in grid_param.keys():
                    try:
                        list_param.append(grid_param[k].rvs(1)[0])
                    except:
                        list_param.append(np.random.choice(grid_param[k]))
                self.grid_param.append(dict(zip(grid_param.keys(), list_param)))
    
    # Method for running Kfolds CV using declared inputs and output:
    def run(self, inputs, output, progress_bar = True):
        """
        
        """
        
        # Registering start time of bootstrap algorithm:
        start_time = datetime.now()
        
        # Dictionary with functions for computing performance metrics:
        metric = {
            'roc_auc': roc_auc_score,
            'avg_precision_score': average_precision_score,
            'brier_loss': brier_score_loss,
            'mse': mse
        }
        
        # Splitting training data into 'k' different folds of data:
        k = list(range(self.num_folds))
        k_folds_X = np.array_split(inputs, self.num_folds)
        k_folds_y = np.array_split(output, self.num_folds)
        
        # Defining objects for storing selected features (if self.pre_select = True), performance metrics for each
        # combination of hyper-parameters, and CV estimated scores:
        self.CV_selected_feat = {}
        self.CV_metric = pd.DataFrame()
        CV_scores = dict(zip([str(g) for g in self.grid_param],
                             [pd.DataFrame(data=[],
                                           columns=['cv_score']) for i in range(len(self.grid_param))]))

        # Progress bar measuring grid estimation progress:
        if progress_bar:
            bar_grid = progressbar.ProgressBar(maxval=len(self.grid_param), widgets=['\033[1mGrid estimation progress:\033[0m ',
                                                                                    progressbar.Bar('-', '[', ']'), ' ',
                                                                                    progressbar.Percentage()])
            bar_grid.start()
        
        # Loop over combinations of hyper-parameters:
        for j in range(len(self.grid_param)):
            # Object for storing performance metrics for each combination of hyper-parameters:
            CV_metric_list = []
            
            try:
                # Loop over folds of data:
                for i in k:
                    # Train and validation split:
                    X_train = pd.concat([x for l,x in enumerate(k_folds_X) if (l!=i)], axis=0, sort=False)
                    y_train = pd.concat([x for l,x in enumerate(k_folds_y) if (l!=i)], axis=0, sort=False)

                    X_val = k_folds_X[i]
                    y_val = k_folds_y[i]

                    # Prior selection of features:
                    if self.pre_selecting:
                        selected_features = pre_selection(input_train=X_train, output_train=y_train,
                                                          regul_param=self.pre_selecting_param,
                                                          task = self.task)
                        self.CV_selected_feat[str(i+1)] = selected_features

                        X_train = X_train[selected_features]
                        X_val = X_val[selected_features]

                    # Create the estimation object:
                    if self.method == 'logistic_regression':
                        model = LogisticRegression(solver='liblinear',
                                                   penalty = 'l1',
                                                   C = self.grid_param[j]['C'],
                                                   warm_start=True)
                        
                    elif self.method == 'GBM':
                        model = GradientBoostingClassifier(subsample = float(self.grid_param[j]['subsample']),
                                                           max_depth = int(self.grid_param[j]['max_depth']),
                                                           learning_rate = float(self.grid_param[j]['learning_rate']),
                                                           n_estimators = int(self.grid_param[j]['n_estimators']),
                                                           warm_start = True)

                    elif self.method == 'lasso':
                        model = Lasso(alpha=self.grid_param[j]['alpha'])

                    elif self.method == 'GBM_reg':
                        model = GradientBoostingRegressor(subsample = float(self.grid_param[j]['subsample']),
                                                          max_depth = int(self.grid_param[j]['max_depth']),
                                                          learning_rate = float(self.grid_param[j]['learning_rate']),
                                                          n_estimators = int(self.grid_param[j]['n_estimators']),
                                                          warm_start = True)
                        
                    elif (self.method == 'random_forest') & (self.task == 'classification'):
                        model = RandomForestClassifier(bootstrap = True, criterion = 'gini',
                                                       n_estimators = int(self.grid_param[j]['n_estimators']),
                                                       max_features = int(self.grid_param[j]['max_features']),
                                                       min_samples_split = int(self.grid_param[j]['min_samples_split']),
                                                       warm_start = True)
                        
                    elif (self.method == 'random_forest') & (self.task == 'regression'):
                        model = RandomForestRegressor(bootstrap = True, criterion = 'mse',
                                                      n_estimators = int(self.grid_param[j]['n_estimators']),
                                                      max_features = int(self.grid_param[j]['max_features']),
                                                      min_samples_split = int(self.grid_param[j]['min_samples_split']),
                                                      warm_start = True)
                    
                    elif (self.method == 'SVM') & (self.task == 'classification'):
                        model = SVC(C = float(self.grid_param[j]['C']),
                                    kernel = self.grid_param[j]['kernel'],
                                    degree = int(self.grid_param[j]['degree']),
                                    gamma = self.grid_param[j]['gamma'],
                                    probability = True, coef0 = 0.0, shrinking = True, tol = 0.001, max_iter = -1,
                                    cache_size = 200, class_weight = None, decision_function_shape = 'ovr',
                                    verbose = False, random_state = None)
                        
                    elif (self.method == 'SVM') & (self.task == 'regression'):
                        model = SVR(C = float(self.grid_param[j]['C']),
                                    kernel = self.grid_param[j]['kernel'],
                                    degree = int(self.grid_param[j]['degree']),
                                    gamma = self.grid_param[j]['gamma'],
                                    epsilon = 0.1, coef0 = 0.0, shrinking = True, cache_size = 200,
                                    tol = 0.001, max_iter = -1, verbose = False)

                    if self.method == 'light_gbm':
                        # Create dictionary with parameters:
                        param = {'metric': self.cost_function, 'objective': self.task,
                                 'bagging_fraction': float(self.grid_param[j]['bagging_fraction']),
                                 'learning_rate': float(self.grid_param[j]['learning_rate']),
                                 'max_depth': int(self.grid_param[j]['max_depth']),
                                 'num_iterations': int(self.grid_param[j]['num_iterations'])}
                        
                        # Defining dataset for light GBM estimation:
                        train_data = lgb.Dataset(X_train.values, label = y_train.values)

                        # Training the model:
                        model = lgb.train(param, train_data, 10, verbose_eval = False)

                        # Predicting scores:
                        score_pred = model.predict(X_val.values)
                        
                    else:
                        # Training the model:
                        model.fit(X_train, y_train)

                        # Predicting scores:
                        if self.task == 'classification':
                            score_pred = [p[1] for p in model.predict_proba(X_val)]

                        else:
                            score_pred = [p for p in model.predict(X_val)]

                    # Calculating performance metric:
                    CV_metric_list.append(metric[self.metric](y_val, score_pred))
                    
                    # Dataframes with CV scores:
                    ref = pd.DataFrame(data={'y_true': list(k_folds_y[i]),
                                             'cv_score': score_pred},
                                       index=list(k_folds_y[i].index))
                    CV_scores[str(self.grid_param[j])] = pd.concat([CV_scores[str(self.grid_param[j])],
                                                                    ref], axis=0, sort=False)
                
                # Dataframes with CV ROC-AUC statistics:
                self.CV_metric = pd.concat([self.CV_metric,
                                            pd.DataFrame(data={'tun_param': str(self.grid_param[j]),
                                                               'cv_' + self.metric: np.nanmean(CV_metric_list)},
                                                         index=[j])], axis=0, sort=False)

            except Exception as e:
                print('\033[1mNot able to perform CV estimation with parameters ' +
                      str(self.grid_param[j]) + '!\033[0m')
                
                self.CV_metric = pd.concat([self.CV_metric,
                                            pd.DataFrame(data={'tun_param': self.grid_param[j],
                                                               'cv_' + self.metric: np.NaN},
                                                         index=[j])], axis=0, sort=False)

                print(e)

            if progress_bar:
                bar_grid.update(j+1)
                sleep(0.1)
        
        # Best tuning hyper-parameters:
        try:
            if (self.metric == 'brier_loss') | (self.metric == 'mse'):
                self.best_param = self.CV_metric['cv_' + self.metric].idxmin()
            else:
                self.best_param = self.CV_metric['cv_' + self.metric].idxmax()
                
            self.best_param = self.grid_param[self.best_param]
            using_default = False
            
        except:
            self.best_param = self.default_param
            using_default = True
        
        # CV scores for best tuning parameter:
        try:
            self.CV_scores = CV_scores[str(self.best_param)]
        except:
            self.CV_scores = pd.DataFrame(data=[])
            
        # Registering end time of K-folds CV:
        end_time = datetime.now()
            
        # Printing outcomes:
        print('\n')
        print('\033[1mRunning time (K-folds CV estimation):\033[0m {} minutes.'.format(round(((end_time - start_time).seconds)/60, 2)))
        print('Start time: {0}, {1}'.format(start_time.strftime('%Y-%m-%d'), start_time.strftime('%H:%M:%S')))
        print('End time: {0}, {1}'.format(end_time.strftime('%Y-%m-%d'), end_time.strftime('%H:%M:%S')))
        print('\n')
        
        print('---------------------------------------------------------------------')
        print('\033[1mK-folds CV outcomes:\033[0m')
        print('Number of data folds: {0}.'.format(self.num_folds))
        
        if self.random_search:
            print('Number of samples for random search: {0}.'.format(self.n_samples))
        
        print('Estimation method: {0}.'.format(self.method.replace('_', ' ')))
        print('Metric for choosing best hyper-parameter: {0}.'.format(self.metric))
        print('Best hyper-parameters: {0}.'.format(self.best_param))
        
        if using_default:
            print('Warning: given problems during K-folds CV estimation, hyper-parameters selected were those provided as default.')
        
        try:
            best_metric = self.CV_metric[self.CV_metric['tun_param']==str(self.best_param)]['cv_' + self.metric].values[0]
            print('CV performance metric associated with best hyper-parameters: {0}.'.format(round(best_metric, 4)))
        
        except:
            pass
        
        print('---------------------------------------------------------------------')
        print('\n')

# Function that applies L1 regularized linear model (linear or logistic regression) to select features for
# each estimation of K-folds CV:
def pre_selection(input_train, output_train, regul_param, task = 'classification'):
    """

    """

    if (task == 'classification') | (task == 'binary'):
        model = LogisticRegression(solver='liblinear', penalty = 'l1',
                                   C = regul_param, max_iter = 100, tol = 1e-4)
        model.fit(input_train, output_train)
        betas = list(model.coef_[0])

    else:
        model = Lasso(alpha = regul_param, max_iter = 5000, tol = 1e-4)
        model.fit(input_train, output_train)
        betas = list(model.coef_)

    model_outcomes = pd.DataFrame(data={'feature': list(input_train.columns), 'beta': betas})

    return [f for f in list(model_outcomes[model_outcomes['beta']!=0].feature)]

# Function that calculates mean-squared error for regression tasks:
def mse(y_true, y_pred):
    return sum([abs(y-y_hat)**2 for y, y_hat in zip(y_true, y_pred)])/len(y_true)

####################################################################################################################################
# Function that applies L1 regularized linear model (linear or logistic regression) to select features for each
# estimation of K-folds CV:

def pre_selection(input_train, output_train, regul_param, task = 'classification'):
    """
    
    """

    if (task == 'classification') | (task == 'regression'):
        model = LogisticRegression(solver='liblinear', penalty = 'l1',
                                   C = regul_param, max_iter = 100, tol = 1e-4)
        model.fit(input_train, output_train)
        betas = list(model.coef_[0])
        
    else:
        model = Lasso(alpha = regul_param, max_iter = 5000, tol = 1e-4)
        model.fit(input_train, output_train)
        betas = list(model.coef_)

    model_outcomes = pd.DataFrame(data={'feature': list(input_train.columns), 'beta': betas})

    return [f for f in list(model_outcomes[model_outcomes['beta']!=0].feature)]

####################################################################################################################################
# Function that calculates mean-squared error for regression tasks:

def mse(y_true, y_pred):
    """
    
    """
    return sum([abs(y-y_hat)**2 for y, y_hat in zip(y_true, y_pred)])/len(y_true)

####################################################################################################################################
# Bootstrap estimation with train-test split and K-folds CV for hyper-parameters definition:

class bootstrap_estimation(object):
    """
    
    """
    def __init__(self, cv=False, task='classification', method='logistic_regression',
                 metric='roc_auc', num_folds=3,
                 pre_selecting=False, pre_selecting_param=None,
                 random_search=False, n_samples=None, grid_param=None, default_param=None,
                 replacement=True, n_iterations=100, bootstrap_scores=False,
                 cost_function=None):
        self.cv = cv
        self.task = task
        self.method = method
        self.replacement = replacement
        self.n_iterations = n_iterations
        self.bootstrap_scores = bootstrap_scores
        self.cost_function = cost_function
        
        # Declaring K-folds CV object:
        if self.cv:
            self.kfolds = KfoldsCV(task=task, method=method, metric=metric, num_folds=num_folds,
                                   pre_selecting=pre_selecting, pre_selecting_param=pre_selecting_param,
                                   random_search=random_search, n_samples=n_samples,
                                   grid_param=grid_param, default_param=default_param,
                                   cost_function=cost_function)
        
        # Declaring hyper-parameters for train-test estimation:
        else:
            self.best_param = default_param
    
    # 
    def run(self, train_inputs, train_output, test_inputs, test_output):
        if (self.task == 'classification') | (self.task == 'binary'):
            self.performance_metrics = {
                "test_roc_auc": [],
                "test_prec_avg": [],
                "test_brier": [],
                "best_param": []
            }
            
        else:
            self.performance_metrics = {
                "test_rmse": [],
                "best_param": []
            }
        
        # Registering start time of bootstrap algorithm:
        start_time = datetime.now()
        
        # Initializing progress bar for bootstrap n_iterations:
        if self.replacement:
            bar = progressbar.ProgressBar(maxval=len(range(self.n_iterations)),
                                          widgets=['\033[1mBoostrap estimation progress: \033[0m',
                                                   progressbar.Bar('=', '[', ']'), ' ',
                                                   progressbar.Percentage()])
        else:
            bar = progressbar.ProgressBar(maxval=len(range(self.n_iterations)),
                              widgets=['\033[1mAveraging estimation progress: \033[0m',
                                       progressbar.Bar('=', '[', ']'), ' ',
                                       progressbar.Percentage()])
        bar.start()
        
        # Bootstrap scores:
        if self.bootstrap_scores:
            self.boot_scores = np.repeat(np.NaN, len(test_output))
        
        for r in range(self.n_iterations):
            # Creating bootstrap samples:
            boot_sample = self.bootstrap_sampling(inputs=train_inputs, output=train_output,
                                                  replacement=self.replacement)
            train_inputs_boot = boot_sample['inputs']
            train_output_boot = boot_sample['output']

            # K-folds CV estimation:
            if self.cv:
                self.kfolds.run(inputs=train_inputs_boot, output=train_output_boot, progress_bar = False)
                self.best_param = self.kfolds.best_param

            # Train-test estimation:
            # Creating estimation object:
            if self.method == 'logistic_regression':
                model = LogisticRegression(solver='liblinear',
                                           penalty = 'l1',
                                           C = self.best_param['C'],
                                           warm_start=True)

            elif self.method == 'GBM':
                model = GradientBoostingClassifier(subsample = float(self.best_param['subsample']),
                                                   learning_rate = float(self.best_param['learning_rate']),
                                                   max_depth = int(self.best_param['max_depth']),
                                                   n_estimators = int(self.best_param['n_estimators']),
                                                   warm_start=True)

            elif self.method == 'lasso':
                model = Lasso(alpha=self.best_param['alpha'])

            elif self.method == 'GBM_reg':
                model = GradientBoostingRegressor(subsample = float(self.best_param['subsample']),
                                                  max_depth = int(self.best_param['max_depth']),
                                                  learning_rate = float(self.best_param['learning_rate']),
                                                  n_estimators = int(self.best_param['n_estimators']),
                                                  warm_start = True)
            
            elif (self.method == 'random_forest') & (self.task == 'classification'):
                model = RandomForestClassifier(bootstrap = True, criterion = 'gini',
                                               n_estimators = int(self.best_param['n_estimators']),
                                               max_features = int(self.best_param['max_features']),
                                               min_samples_split = int(self.best_param['min_samples_split']),
                                               warm_start = True)
                        
            elif (self.method == 'random_forest') & (self.task == 'regression'):
                model = RandomForestRegressor(bootstrap = True, criterion = 'mse',
                                              n_estimators = int(self.best_param['n_estimators']),
                                              max_features = int(self.best_param['max_features']),
                                              min_samples_split = int(self.best_param['min_samples_split']),
                                              warm_start = True)

            elif (self.method == 'SVM') & (self.task == 'classification'):
                model = SVC(C = float(self.best_param['C']),
                            kernel = self.best_param['kernel'],
                            degree = int(self.best_param['degree']),
                            gamma = self.best_param['gamma'],
                            probability = True, coef0 = 0.0, shrinking = True, tol = 0.001, max_iter = -1,
                            cache_size = 200, class_weight = None, decision_function_shape = 'ovr',
                            verbose = False, random_state = None)

            elif (self.method == 'SVM') & (self.task == 'regression'):
                model = SVR(C = float(self.best_param['C']),
                            kernel = self.best_param['kernel'],
                            degree = int(self.best_param['degree']),
                            gamma = self.best_param['gamma'],
                            epsilon = 0.1, coef0 = 0.0, shrinking = True, cache_size = 200,
                            tol = 0.001, max_iter = -1, verbose = False)

            if self.method == 'light_gbm':
                # Create dictionary with parameters:
                param = {'metric': self.cost_function, 'objective': self.task,
                         'bagging_fraction': float(self.best_param['bagging_fraction']),
                         'learning_rate': float(self.best_param['learning_rate']),
                         'max_depth': int(self.best_param['max_depth']),
                         'num_iterations': int(self.best_param['num_iterations'])}

                # Defining dataset for light GBM estimation:
                train_data = lgb.Dataset(train_inputs.values, label = train_output.values)

                # Training the model:
                model = lgb.train(param, train_data, 10, verbose_eval = False)

                # Predicting scores:
                score_pred = model.predict(test_inputs.values)
            
            else:
                # Running estimation:
                model.fit(train_inputs_boot, train_output_boot)

                # Predicting scores:
                if self.task == 'classification':
                    score_pred = [p[1] for p in model.predict_proba(test_inputs)]

                else:
                    score_pred = [p for p in model.predict(test_inputs)]
            
            # Calculating performance metrics:
            if (self.task == 'classification') | (self.task == 'binary'):
                self.performance_metrics["test_roc_auc"].append(roc_auc_score(test_output, score_pred))
                self.performance_metrics["test_prec_avg"].append(average_precision_score(test_output, score_pred))
                self.performance_metrics["test_brier"].append(brier_score_loss(test_output, score_pred))
            
            else:
                self.performance_metrics["test_rmse"].append(np.sqrt(mse(test_output, score_pred)))
            
            self.performance_metrics["best_param"].append(self.best_param)
            
            # Bootstrap statistics for performance metrics:
            self.boot_stats = {}
            
            for k in [k for k in self.performance_metrics.keys() if k != 'best_param']:
                self.boot_stats[k.replace('test_', '')] = {
                    "mean": np.nanmean(self.performance_metrics[k]),
                    "std": np.nanstd(self.performance_metrics[k])
                }
            
            # Bootstrap estimation scores:
            if self.bootstrap_scores:
                self.boot_scores = [np.nansum([x, y]) for x,y in zip(self.boot_scores, score_pred)]
                
                if (self.task == 'classification') | (self.task == 'binary'):
                    self.boot_metrics = {
                        "roc_auc": roc_auc_score(test_output, [s/(r+1) for s in self.boot_scores]),
                        "prec_avg": average_precision_score(test_output, [s/(r+1) for s in self.boot_scores]),
                        "brier": brier_score_loss(test_output, [s/(r+1) for s in self.boot_scores])
                    }
                
                else:
                    self.boot_metrics = {
                        "rmse": np.sqrt(mse(test_output, [s/(r+1) for s in self.boot_scores]))
                    }
            
            # Updating progress bar for bootstrap iterations:
            bar.update(list(range(self.n_iterations)).index(r)+1)
            sleep(0.1)
        
        # Bootstrap estimation scores:
        if self.bootstrap_scores:
            self.boot_scores = [s/(r+1) for s in self.boot_scores]
        
        # Registering end time of bootstrap algorithm:
        end_time = datetime.now()
        
        print('\n')
        print('\033[1mRunning time:\033[0m {} minutes.'.format(round(((end_time - start_time).seconds)/60, 2)))
        print('Start time: {0}, {1}'.format(start_time.strftime('%Y-%m-%d'), start_time.strftime('%H:%M:%S')))
        print('End time: {0}, {1}'.format(end_time.strftime('%Y-%m-%d'), end_time.strftime('%H:%M:%S')))
        print('\n')
        
        print('---------------------------------------------------------------------------------------------')
        if self.replacement:
            print('\033[1mBootstrap statistics:\033[0m')
        else:
            print('\033[1mAveraging statistics:\033[0m')
        
        print('Number of estimations: {}.'.format(self.n_iterations))
        for k in self.boot_stats.keys():
            print('avg({0}) = {1}'.format(k, round(self.boot_stats[k]['mean'], 4)))
            print('std({0}) = {1}'.format(k, round(self.boot_stats[k]['std'], 4)))
        
        if self.bootstrap_scores:
            print('\n')
            if self.replacement:
                print('\033[1mPerformance metrics based on bootstrap scores:\033[0m')
            else:
                print('\033[1mPerformance metrics based on averaging scores:\033[0m')
            
            for k in self.boot_metrics.keys():
                print('{0} = {1}'.format(k, round(self.boot_metrics[k], 4)))
        
        if self.cv:
            print('\n')
            counts = np.unique([str(p) for p in self.performance_metrics['best_param']], return_counts=True)
            freqs = list(counts[1])
            values = list(counts[0])
            self.most_freq_param = values[freqs.index(max(freqs))]
            print("Most frequent best hyper-parameters: {0} ({1} out of {2} times).".format(self.most_freq_param,
                                                                                            max(freqs),
                                                                                            self.n_iterations))
        else:
            print("Hyper-parameters used in estimations: {}.".format(self.best_param))
        
        print('---------------------------------------------------------------------------------------------')
        print('\n')

    # Method that creates a bootstrap sample:
    def bootstrap_sampling(self, inputs, output, replacement = True):
        n_sample = len(inputs)
        sample = sorted(np.random.choice(range(n_sample), size=n_sample, replace=replacement))

        return {'inputs': inputs.iloc[sample, :], 'output': output.iloc[sample]}

####################################################################################################################################
# K-folds CV estimation, refitting using all training data, and evaluating performance metrics on test data (when provided):

class Kfolds_fit(object):
    """
    
    """
    def __init__(self, task='classification', method='logistic_regression',
                 metric='roc_auc', num_folds=3,
                 pre_selecting=False, pre_selecting_param=None,
                 random_search=False, n_samples=None,
                 grid_param=None, default_param=None,
                 cost_function=None):
        self.task = task
        self.method = method
        self.cost_function = cost_function
        
        # Declaring K-folds CV object:
        self.kfolds = KfoldsCV(task=task, method=method, metric=metric, num_folds=num_folds,
                               pre_selecting=pre_selecting, pre_selecting_param=pre_selecting_param,
                               random_search=random_search, n_samples=n_samples,
                               grid_param=grid_param, default_param=default_param,
                               cost_function=cost_function)
    
    # Method that runs K-folds CV estimation, refits using all training data, and evaluate performance metrics on
    # test data (when provided):
    def run(self, train_inputs, train_output, test_inputs=None, test_output=None):
        # Registering start time of bootstrap algorithm:
        start_time = datetime.now()
        
        # Dictionary for storing performance metrics:
        if test_inputs is not None:
            self.performance_metrics = {}
        
        # K-folds CV estimation:
        self.kfolds.run(inputs=train_inputs, output=train_output, progress_bar = True)
        self.best_param = self.kfolds.best_param

        # Train-test estimation:
        # Creating estimation object:
        if self.method == 'logistic_regression':
            self.model = LogisticRegression(solver='liblinear',
                                            penalty = 'l1',
                                            C = self.best_param['C'],
                                            warm_start=True)

        elif self.method == 'GBM':
            self.model = GradientBoostingClassifier(subsample = float(self.best_param['subsample']),
                                                    learning_rate = float(self.best_param['learning_rate']),
                                                    max_depth = int(self.best_param['max_depth']),
                                                    n_estimators = int(self.best_param['n_estimators']),
                                                    warm_start=True)

        elif self.method == 'lasso':
            self.model = Lasso(alpha=self.best_param['alpha'])

        elif self.method == 'GBM_reg':
            self.model = GradientBoostingRegressor(subsample = float(self.best_param['subsample']),
                                                   max_depth = int(self.best_param['max_depth']),
                                                   learning_rate = float(self.best_param['learning_rate']),
                                                   n_estimators = int(self.best_param['n_estimators']),
                                                   warm_start = True)

        elif (self.method == 'random_forest') & (self.task == 'classification'):
            self.model = RandomForestClassifier(bootstrap = True, criterion = 'gini',
                                                n_estimators = int(self.best_param['n_estimators']),
                                                max_features = int(self.best_param['max_features']),
                                                min_samples_split = int(self.best_param['min_samples_split']),
                                                warm_start = True)

        elif (self.method == 'random_forest') & (self.task == 'regression'):
            self.model = RandomForestRegressor(bootstrap = True, criterion = 'mse',
                                               n_estimators = int(self.best_param['n_estimators']),
                                               max_features = int(self.best_param['max_features']),
                                               min_samples_split = int(self.best_param['min_samples_split']),
                                               warm_start = True)

        elif (self.method == 'SVM') & (self.task == 'classification'):
            self.model = SVC(C = float(self.best_param['C']),
                             kernel = self.best_param['kernel'],
                             degree = int(self.best_param['degree']),
                             gamma = self.best_param['gamma'],
                             probability = True, coef0 = 0.0, shrinking = True, tol = 0.001, max_iter = -1,
                             cache_size = 200, class_weight = None, decision_function_shape = 'ovr',
                             verbose = False, random_state = None)

        elif (self.method == 'SVM') & (self.task == 'regression'):
            self.model = SVR(C = float(self.best_param['C']),
                             kernel = self.best_param['kernel'],
                             degree = int(self.best_param['degree']),
                             gamma = self.best_param['gamma'],
                             epsilon = 0.1, coef0 = 0.0, shrinking = True, cache_size = 200,
                             tol = 0.001, max_iter = -1, verbose = False)

        if self.method == 'light_gbm':
            # Create dictionary with parameters:
            param = {'metric': self.cost_function, 'objective': self.task,
                     'bagging_fraction': float(self.best_param['bagging_fraction']),
                     'learning_rate': float(self.best_param['learning_rate']),
                     'max_depth': int(self.best_param['max_depth']),
                     'num_iterations': int(self.best_param['num_iterations'])}

            # Defining dataset for light GBM estimation:
            train_data = lgb.Dataset(train_inputs.values, label = train_output.values)

            # Training the model:
            self.model = lgb.train(param, train_data, 10, verbose_eval = False)

            # Predicting scores:
            if test_inputs is not None:
                score_pred = self.model.predict(test_inputs.values)

        else:
            # Running estimation:
            self.model.fit(train_inputs, train_output)

            if test_inputs is not None:
                # Predicting scores:
                if self.task == 'classification':
                    score_pred = [p[1] for p in self.model.predict_proba(test_inputs)]

                else:
                    score_pred = [p for p in self.model.predict(test_inputs)]

                # Calculating performance metrics:
                if (self.task == 'classification') | (self.task == 'binary'):
                    self.performance_metrics["test_roc_auc"] = roc_auc_score(test_output, score_pred)
                    self.performance_metrics["test_prec_avg"] = average_precision_score(test_output, score_pred)
                    self.performance_metrics["test_brier"] = brier_score_loss(test_output, score_pred)

                else:
                    self.performance_metrics["test_rmse"] = np.sqrt(mse(test_output, score_pred))
        
        # Registering end time of bootstrap algorithm:
        end_time = datetime.now()
        
        print('\n')
        if test_inputs is not None:
            print('--------------------------------------------')
            print('\033[1mPerformance metrics evaluated at test data:\033[0m')
            for k in self.performance_metrics.keys():
                print('{0} = {1}'.format(k, round(self.performance_metrics[k], 4)))
            print('--------------------------------------------')
            print('\n')
        
        print('\033[1mTotal rununning time:\033[0m {} minutes.'.format(round(((end_time - start_time).seconds)/60,
                                                                             2)))
        print('Start time: {0}, {1}'.format(start_time.strftime('%Y-%m-%d'), start_time.strftime('%H:%M:%S')))
        print('End time: {0}, {1}'.format(end_time.strftime('%Y-%m-%d'), end_time.strftime('%H:%M:%S')))
        print('\n')
