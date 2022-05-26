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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


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

# 2. split data into training and test sets
train, test = train_test_split(data, test_size=0.2, stratify= data.label,random_state=691)
print(f"No. of training examples: {train.shape[0]}")
print(f"No. of testing examples: {test.shape[0]}")

# 3. EDA
# from pandas_profiling import ProfileReport
profile_train = pandas_profiling.ProfileReport(train, title="Pandas Profiling Report (Train)")
profile_train.to_file("DSCI691-GRP-PICKLE_RICK/Task_1/subtask_1a/profile_train.html")
# Notes:
# There is a class imbalance in the outcome variable "class"
# Only 7.2% of the tweets are labeled as "NoADE"
# see report for more details
# suggestion from jake: keep data as is

# check for missing values
train.isnull().sum()
# no missing values!
test.isnull().sum()
# no missing values!

# check for class imbalance
data.groupby('label').size()
train.groupby(['label']).size()
test.groupby(['label']).size()

# visualize the data
import seaborn as sns

p1 = sns.countplot(x='label', data=train)
p1.set_title('ADE Dist. in Training Data')
# show percent of each class
for p in p1.patches:
    p1.annotate('{:6.2f}%'.format(p.get_height()/len(train)*100), (p.get_x() + 0.3, p.get_height() + 0.3))

p1.set_xlabel('ADE')

p2 = sns.countplot(x='label', data=test)
p2.set_title('ADE Dist. in Test Data')
# show percent of each class
for p in p2.patches:
    p2.annotate('{:6.2f}%'.format(p.get_height()/len(test)*100), (p.get_x() + 0.3, p.get_height() + 0.3))

p2.set_xlabel('ADE')

train.info()

# 4. text representation
# convert tweets to matrix of word counts and remove stop words
from sklearn.feature_extraction.text import CountVectorizer

countvec = CountVectorizer(stop_words='english')

# normalise count matrix to decrease the effect of word frequencies
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()

# vectorize and transform train and test data
train_transformed = tfidf.fit_transform(countvec.fit_transform(train.tweet))
test_transformed = tfidf.transform(countvec.transform(test.tweet)) 