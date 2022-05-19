# Notebook for creating baselines for classification
# Project: Adverse Druge Event (ADE) Classification from Tweets
# Group: pickle rick

# libraries
import re
import pandas as pd
import scipy as sp
import numpy as np
import scipy.sparse
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


# import data
tweets = pd.read_csv('DSCI691-GRP-PICKLE_RICK/Task_1/subtask_1a/data/training/tweets.tsv', sep='\t', header=None)
classes = pd.read_csv('DSCI691-GRP-PICKLE_RICK/Task_1/subtask_1a/data/training/class.tsv', sep='\t', header=1)

# WHY ARE THERE MORE CLASSES THAN TWEETS??

###############################################################################
# GOAL: Create a baseline model for classification
# CLASSIFY IF A DRUG EVENT IS PRESENT IN A TWEET - (ADE/NoADE)
# BASELINE: Logistic Regression Classifier (LR) with L2 Regularization (L2R) #

# 1. Preprocessing
# 1.1. Remove '@USER' and any proceeding '_' from tweets

tweets = tweets.replace(r'@\w+', '', regex=True)

# left join classes to tweets based on tweet id
data = pd.merge(tweets, classes, on=0, how='left')

# 2. split data into training and test sets
train = data.sample(frac=0.8, random_state=1)
test = data.drop(train.index)

# 3. EDA
# from pandas_profiling import ProfileReport
profile_train = ProfileReport(train, title="Pandas Profiling Report (Train)")
profile_train.to_file("DSCI691-GRP-PICKLE_RICK/Task_1/subtask_1a/profile_train.html")
# Notes:
# There is a class imbalance in the outcome variable "class"
# Only 7.2% of the tweets are labeled as "NoADE"
# see report for more details

# check for missing values
train.isnull().sum()
# no missing values!
test.isnull().sum()
# no missing values!

# check for class imbalance
train.groupby(['1_y']).size()
test.groupby(['1_y']).size()

# re-split data to make classes balanced
_, idx_holdout, _, y_holdout = train_test_split(tweets, classes, test_size=0.2, random_state=1)