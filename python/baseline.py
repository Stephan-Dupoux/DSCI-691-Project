# Notebook for creating baselines for classification
# Project: Adverse Druge Event (ADE) Classification from Tweets
# Group: pickle rick

# libraries
import re
import pandas as pd
import pandas_profiling
import scipy as sp
import numpy as np
import scipy.sparse
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score


# import data
tweets = pd.read_csv('DSCI691-GRP-PICKLE_RICK/Task_1/subtask_1a/data/training/tweets.tsv', sep='\t', header=None,
                     names=['tweet_id', 'tweet'])
classes = pd.read_csv('DSCI691-GRP-PICKLE_RICK/Task_1/subtask_1a/data/training/class.tsv', sep='\t', header=0)

# WHY ARE THERE MORE CLASSES THAN TWEETS??

###############################################################################
# GOAL: Create a baseline model for classification
# CLASSIFY IF A DRUG EVENT IS PRESENT IN A TWEET - (ADE/NoADE)
# BASELINE: Logistic Regression Classifier (LR) with L2 Regularization (L2R) #

# 1. Preprocessing
# there are more classses (n=17,385) than tweets(n=17,120)
# balance the classes by left joining the classes and tweets by var 0 and tweet_id

data = pd.merge(tweets, classes, how='left')

# 1.1. Remove '@USER' and any proceeding '_' from tweet variable in dataframe
data = data.replace(r'@\w+', '', regex=True)

# are there duplicates?
np.sum(data.duplicated()) 
# NO!
# are there missing values?
data.isnull().sum()
# no missing values!

# convert label to binary
data = data.replace(['NoADE', 'ADE'], [0, 1])


# 2. split data into training and test sets
# use stratified sampling to balance the classes
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=691)
X = data['tweet'].to_numpy()
y = data['label'].to_numpy()

for train_index, test_index in strat_split.split(X, y):
    print(f"Train index: {train_index}", f"Test index: {test_index}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# train, test = train_test_split(data, test_size=0.2, stratify= data.label,random_state=691)
# print(f"No. of training examples: {train.shape[0]}")
# print(f"No. of testing examples: {test.shape[0]}")

# 3. EDA
# from pandas_profiling import ProfileReport
# profile_train = pandas_profiling.ProfileReport(train, title="Pandas Profiling Report (Train)")
# profile_train.to_file("DSCI691-GRP-PICKLE_RICK/Task_1/subtask_1a/profile_train.html")
# check for class imbalance
data.groupby('label').size()

# visualize the distribution of y_train data
import matplotlib.pyplot as plt
ys = pd.Series(y_train)
ys.value_counts().plot(kind='bar')
plt.show()

# test data
ys2 = pd.Series(y_test)
ys2.value_counts().plot(kind='bar')
plt.show()

# Notes:
# There is a class imbalance in the outcome variable "class"
# Only 7.2% of the tweets are labeled as "NoADE"
# see report for more details
# suggestion from jake: keep data as is

# 4. text representation
# convert tweets to matrix of word counts and remove stop words
from sklearn.feature_extraction.text import CountVectorizer

countvec = CountVectorizer(stop_words='english')

# normalise count matrix to decrease the effect of word frequencies
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()

# vectorize and transform train and test data
train_transformed = tfidf.fit_transform(countvec.fit_transform(X_train))
test_transformed = tfidf.transform(countvec.transform(X_test))

# 5. evaluation
# classification using logistic regression
# course notes uses the 'liblinear' solver however sklearn uses the 'lbfgs' solver as default
log_reg = LogisticRegression(solver='lbfgs', random_state=691, class_weight='balanced')

# fit
log_reg.fit(train_transformed, y_train)
y_pred = log_reg.predict(test_transformed)

# print results
print(f"Logistic Regression:")
print(f"AUC: {roc_auc_score(y_test, y_pred)}")  #0.797
print(f"Precision: {precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)[0]:.2f}") # 0.37
print(f"Recall: {precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)[1]:.2f}") # 0.68
print(f"F1 Score: {precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)[2]:.2f}") # 0.48