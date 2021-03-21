####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

####################################################################################################################################
# Logarithm of selected features:

class log_transformation(object):
    """Applies function to log-transform all variables in a dataframe except for those
    explicitly declared. Returns the dataframe with selected variables log-transformed
    and their respective names changed to 'L#PREVIOUS_NAME()'."""

    def __init__(self, not_log):
        self.not_log = not_log
        
    def transform(self, data):
        # Function that applies natural logarithm to numerical variables:
        def log_func(x):
            """Since numerical features are not expected to assume negative values here, and since, after a sample
            assessment, only a few negative values were identified for just a few variables, suggesting the occurrence of
            technical issues for such observations, any negative values will be truncated to zero when performing
            log-transformation."""
            if x < 0:
                new_value = 0
            else:
                new_value = x

            transf_value = np.log(new_value + 0.0001)

            return transf_value
        
        # Redefining names of columns:
        new_col = []
        log_vars = []
        
        self.log_transformed = data
        
        # Applying logarithmic transformation to selected variables:
        for f in list(data.columns):
            if f in self.not_log:
                new_col.append(f)
            else:
                new_col.append('L#' + f)
                log_vars.append('L#' + f)
                self.log_transformed[f] = data[f].apply(log_func)

        self.log_transformed.columns = new_col
        
        print('\033[1mNumber of numerical variables log-transformed:\033[0m ' + str(len(log_vars)) + '.')

####################################################################################################################################
# Standardizing selected features:

class standard_scale(object):
    """Fits and transforms all variables in a dataframe, except for those explicitly defined to not scale.
    Uses 'StandardScaler' from sklearn and returns not only scaled data, but also in its dataframe original
    format. If test data is provided, then their values will be standardized using means and variances from
    train data."""
    
    def __init__(self, not_stand):
        self.not_stand = not_stand
    
    def scale(self, train, test=None):
        # Creating standardizing object:
        scaler = StandardScaler()
        
        # Calculating means and variances:
        scaler.fit(train.drop(self.not_stand, axis=1))
        
        # Standardizing selected variables:
        self.train_scaled = scaler.transform(train.drop(self.not_stand, axis=1))
        
        # Transforming data into dataframe and concatenating selected and non-selected variables:
        self.train_scaled = pd.DataFrame(data=self.train_scaled,
                                         columns=train.drop(self.not_stand, axis=1).columns)
        self.train_scaled.index = train.index
        self.train_scaled = pd.concat([train[self.not_stand], self.train_scaled], axis=1)
        
        # Test data:
        if test is not None:
            # # Standardizing selected variables:
            self.test_scaled = scaler.transform(test.drop(self.not_stand, axis=1))
            
            # Transforming data into dataframe and concatenating selected and non-selected variables:
            self.test_scaled = pd.DataFrame(data=self.test_scaled,
                                            columns=test.drop(self.not_stand, axis=1).columns)
            self.test_scaled.index = test.index
            self.test_scaled = pd.concat([test[self.not_stand], self.test_scaled], axis=1)

####################################################################################################################################
# Method that creates dummies from categorical features following a variance criterium for selecting categories:
class one_hot_encoding(object):
    """
    Arguments for initialization:
        'features': list of categorical features whose categories should be selected.
        'variance_param': parameter for selection based on the variance of a given dummy variable.
    Methods:
        'create_dummies': for a given training data ('categorical_train'), performs selection of dummies based on variance criterium.
        Then, creates the same set of dummy variables for test data ('categorical_test').
    Output objects:
        'self.categorical_features': list of categorical features whose categories should be selected.
        'self.variance_param': parameter for selection based on the variance of a given dummy variable.
        'self.dummies_train': dataframe with selected dummies for training data.
        'self.dummies_test': dataframe for test data with dummies selected from training data.
        'self.categories_assessment': dictionary with number of overall categories, number of selected categories, and selected
        categories for each categorical feature.
    """
    def __init__(self, categorical_features,  variance_param = 0.01):  
        self.categorical_features = categorical_features
        self.variance_param = variance_param

    def create_dummies(self, categorical_train, categorical_test = None):
        self.dummies_train = pd.DataFrame(data=[])
        self.dummies_test = pd.DataFrame(data=[])
        self.categories_assessment = {}
        
        # Loop over categorical features:
        for f in self.categorical_features:
            # Training data:
            # Creating dummy variables:
            dummies_cat = pd.get_dummies(categorical_train[f]) 
            dummies_cat.columns = ['C#' + f + '#' + str.upper(str(c)) for c in dummies_cat.columns]

            # Selecting dummies_cat depending on their variance:
            selected_cat = [d for d in dummies_cat.columns if dummies_cat[d].var() > self.variance_param]

            # Dataframe with dummy variables for all categorical features (training data):
            self.dummies_train = pd.concat([self.dummies_train, dummies_cat[selected_cat]], axis=1)
            
            # Assessing categories:
            self.categories_assessment[f] = {
                "num_categories": len(dummies_cat.columns),
                "num_selected_categories": len(selected_cat),
                "selected_categories": selected_cat
            }

            if categorical_test is not None:
                # Test data:
                dummies_cat = pd.get_dummies(categorical_test[f])
                dummies_cat.columns = ['C#' + f + '#' + str.upper(str(c)) for c in dummies_cat.columns]

                # Checking if all categories selected from training data also exist for test data:
                for c in selected_cat:
                    if c not in dummies_cat.columns:
                        dummies_cat[c] = [0 for i in range(len(dummies_cat))]

                # Dataframe with dummy variables for all categorical features (test data):
                self.dummies_test = pd.concat([self.dummies_test, dummies_cat[selected_cat]], axis=1)

                # Preserving columns order as the same for training data:
                self.dummies_test = self.dummies_test[list(self.dummies_train.columns)]

####################################################################################################################################
# Function that recriates original missing values from dummy variable of missing value status:
def recreate_missings(var, missing_var):
    """
    Arguments:
        'var': variable (series, array, or list) to impute missing values.
        'missing_var': variable (series, array, or list) that indicates missing data.
        Attention: both arguments should have the same lenght and should have the same index.
    Outputs:
        A list with missing values recreated (if any exists in 'missing_var').
    """
    var_list = list(var)
    missing_var_list = list(missing_var)
    new_values = []
    
    # Loop over observations:
    for i in range(len(var_list)):
        if missing_var_list[i] == 1:
            new_values.append(np.NaN)
        else:
            new_values.append(var_list[i])
    
    return new_values

####################################################################################################################################
# Function that treats missing values by imputing 0 whenever they are found:
def impute_missing(var):
    """
    Arguments:
        'var': variable (series, array, or list) whose missing values should be replaced by 0.
    Outputs:
        A dictionary containing a list of values for the variable after missing values treatment, and a list of
        missing value status.
    """
    var_list = list(var)
    new_values = []
    missing_var = []
    
    # Loop over observations:
    for value in var_list:
        if np.isnan(value):
            new_values.append(0)
            missing_var.append(1)
        else:
            new_values.append(value)
            missing_var.append(0)
    
    return {'var': new_values, 'missing_var': missing_var}
