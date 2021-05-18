####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np

from datetime import datetime
import time

import re
# pip install unidecode
from unidecode import unidecode

####################################################################################################################################
####################################################################################################################################
#############################################################FUNCTIONS##############################################################

####################################################################################################################################
# Function that converts epoch into date:

def epoch_to_date(x):
    str_datetime = time.strftime('%d %b %Y', time.localtime(x/1000))
    dt = datetime.strptime(str_datetime, '%d %b %Y')
    return dt

####################################################################################################################################
# Function that defines categorical features from dummy variables:

def get_cat(df):
    c_cols = list(df.columns[df.columns.str.contains('C#')])
    c_feat = list(set([x.split('#')[1] for x in c_cols]))
    return c_feat

####################################################################################################################################
# Function that returns residual status for a given categorical feature:

def resid_cat(df, cat_feat):
    assess = df[df.columns[(df.columns.str.contains('C#' + cat_feat)) |
                           (df.columns.isin(['NA#' + cat_feat]))]].sum(axis=1)
    
    if sum([1 for x in assess.unique() if x > 1]) > 0:
        return 'Somethind has gone weird with one hot encoding for this feature!'
    else:
        return assess.apply(lambda x: 1 if x==0 else 0)

####################################################################################################################################
# Function for cleaning texts:

def text_clean(text, lower=True):
    if pd.isnull(text):
        return text
    
    else:
        # Removing accent:
        text_cleaned = unidecode(text)

        # Removing extra spaces:
        text_cleaned = re.sub(' +', ' ', text_cleaned)
        
        # Removing spaces before and after text:
        text_cleaned = str.strip(text_cleaned)
        
        # Replacing spaces:
        text_cleaned = text_cleaned.replace(' ', '_')
        
        # Replacing signs:
        for m in '+-!@#$%¨&*()[]{}\\|':
            if m in text_cleaned:
                text_cleaned = text_cleaned.replace(m, '_')

        # Setting text to lower case:
        if lower:
            text_cleaned = text_cleaned.lower()

        return text_cleaned

####################################################################################################################################
# Function that identifies if a given feature name corresponds to a velocity:

def is_velocity(string):
    if ('C#' in string) | ('NA#' in string):
        return False
    
    x1 = string.split('(')
    
    if len(x1) <= 1:
        return False

    x2 = x1[1]       

    if len(x2) <= 1:
        return False
    
    check = 0
    x3 = x2.split(')')[0].split(',')
    
    if len(x3) == 2:
        first_clause = len([1 for d in '0123456789' if d in x3[0]]) == 0
        second_clause = len([1 for d in '0123456789' if d in x3[1]]) > 0
        third_clause = len([1 for l in 'abcdefghijklmnopqrstuvxwyzç' if l in str.lower(x3[0])]) > 0
        fourth_clause = len([1 for l in 'abcdefghijklmnopqrstuvxwyzç' if l in str.lower(x3[1])]) == 0
        
        if first_clause & second_clause & third_clause & fourth_clause:
            check += 1
    
    return check > 0

####################################################################################################################################
# Function that produces a random sample preserving classes distribution of a categorical variable:

def balanced_sample(dataframe, categorical_var, classes, sample_share=0.5):
    """
    Arguments:
        'dataframe': dataframe containing indices to be drawn and a categorical variable whose distribution in
        the sample should be kept equal to that of whole data.
        'categorical_var': categorical variable of reference (string).
        'classes': dictionary whose keys are classes and values are their shares in the entire data.
        'sample_share': float indicating sample length as the proportion of entire data length.
    Output:
        Returns a list with randomly picked indices.
    """
    
    # Randomly picked indices:
    samples = [sorted(np.random.choice(dataframe[dataframe[categorical_var]==k].index,
                                       size=int(classes[k]*sample_share*len(dataframe)),
                                       replace=False)) for k in classes.keys()]
    
    sample = []

    # Loop over samples:
    for l in samples:
        # Loop over indices:
        for i in l:
            sample.append(i)
    
    return sample

####################################################################################################################################
# Function that produces permutations from elements and their possible values:

def permutation(dictionary):
    """
    Arguments:
        'dictionary': keys are elements of the permutation, while values are lists of alternatives for each
        element.
    
    Outputs:
        Return a list with all permutations given elements and possible values for each element.
    """

    list_param = [dictionary[k] for k in dictionary.keys()]
    list_param = [list(x) for x in np.array(np.meshgrid(*list_param)).T.reshape(-1,len(list_param))]
    permutations = []
    
    for i in list_param:
        permutations.append(dict(zip(dictionary.keys(), i)))
    
    return permutations
