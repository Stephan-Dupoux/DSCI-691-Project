# Notebook for creating baselines for classification
# Project: Adverse Druge Event (ADE) Classification from Tweets
# Group: pickle rick
# This script contains two parts: a baseline model using logistic regression and
# a baseline model using SVM with TF-IDF features 


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
from sklearn.svm import SVC
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

# remove any emoji from the tweet
data = data.replace(r'[^\w\s]', '', regex=True)

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
###################################################################################################
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

###################################################################################################
# LOGISTIC REGRESSION
###################################################################################################
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

# confusion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)

###################################################################################################
# SVM WITH TF-IDF FEATURES
###################################################################################################
# standard SVM classifier with TF-IDF features
# linear kernel
sv_m = SVC(kernel='linear', class_weight='balanced', random_state=691)
# fit
sv_m.fit(train_transformed, y_train)
y_pred_sv = sv_m.predict(test_transformed)

# print results
print(f"SVM:")
print(f"AUC: {roc_auc_score(y_test, y_pred_sv)}")
print(f"Precision: {precision_recall_fscore_support(y_test, y_pred_sv, average='binary', pos_label=1)[0]:.2f}")
print(f"Recall: {precision_recall_fscore_support(y_test, y_pred_sv, average='binary', pos_label=1)[1]:.2f}")
print(f"F1 Score: {precision_recall_fscore_support(y_test, y_pred_sv, average='binary', pos_label=1)[2]:.2f}")

######################################################################
# create baseline word2vec model with tweet data

# input: list of tokenized tweets
tweets_ls = []
for tweet in data['tweet']:
    tweets_ls.append(tweet.split())
# build word2vec model
import gensim.models as gm
# `workers` is the number of cores to use and does not work without Cython
import Cython
base_model = gm.Word2Vec(tweets_ls, vector_size=200, min_count=1, workers=4)
# ran in 1.4 seconds
base_model.build_vocab(tweets_ls)
total = base_model.corpus_count

# retrain base_model with GloVe vocaublary and starting weights
base_model.build_vocab([glove_vec.index_to_key], update=True)
# train on tweets
base_model.train(tweets_ls, total_examples=total, epochs=base_model.epochs)
# set of word vectors with glove weights and trained on tweets
base_model_wv = base_model.wv # KeyedVectors instance

# function to transform tweets to word2vec vectors
# accounts for dimensionality of vectors - if word not in base_model_wv, use 0 vector
# uses the mean of all word vectors in tweet
def tweet_to_wv(tweets):
    tweet_wv = []
    for tweet in tweets:
        tweet_vec = np.zeros(200)
        for word in tweet:
            if word in base_model_wv.index_to_key:
                tweet_vec += base_model_wv[word]
            else:
                tweet_vec += np.zeros(200)
        tweet_vec /= len(tweet)
        tweet_wv.append(tweet_vec)
    return tweet_wv

# transform train and test data
train_wv = tweet_to_wv(X_train)
test_wv = tweet_to_wv(X_test)
##########################################
# SVM with word2vec features
# linear kernel
svm_wv = SVC(kernel='linear', class_weight='balanced', random_state=691)
# fit ~ 1hr
svm_wv.fit(train_wv, y_train)
y_pred_wv = svm_wv.predict(test_wv)
# save model
import pickle
filename = 'DSCI691-GRP-PICKLE_RICK/Project/svm_wv.sav'
pickle.dump(svm_wv, open(filename, 'wb'))

# print metrics
from sklearn import metrics
print(f"SVM with word2vec features:")
print(metrics.classification_report(y_test, y_pred_wv))

##########################################
# rbf kernel
svm_wv_rbf = SVC(kernel='rbf', class_weight='balanced', random_state=691)
# fit
svm_wv_rbf.fit(train_wv, y_train)
y_pred_wv_rbf = svm_wv_rbf.predict(test_wv)
# save model
filename = 'DSCI691-GRP-PICKLE_RICK/Project/svm_wv_rbf.sav'
pickle.dump(svm_wv_rbf, open(filename, 'wb'))
# print metrics
print(f"SVM with word2vec features and rbf kernel:")
print(metrics.classification_report(y_test, y_pred_wv_rbf))

##############################################
# SVM with rbf kernel and gridsearch
from sklearn.model_selection import GridSearchCV
parameters = {
    'C' : [0.1, 1, 10],
    'gamma' : [1, 'auto', 'scale']
}
svm_wv_rbf2 = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', random_state=691), parameters, cv=5)
# fit
svm_wv_rbf2.fit(train_wv, y_train) # ~20 mins
y_pred_wv_rbf2 = svm_wv_rbf2.predict(test_wv)
# save model
filename = 'DSCI691-GRP-PICKLE_RICK/Project/svm_wv_rbf2.sav'
pickle.dump(svm_wv_rbf2, open(filename, 'wb'))
# print metrics
print(f"SVM with word2vec features and rbf kernel and gridsearch:")
print(metrics.classification_report(y_test, y_pred_wv_rbf2))