# libraries
import re
import pandas as pd
#import pandas_profiling
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
from sklearn.model_selection import train_test_split


# import data
tweets = pd.read_csv('/Users/stephandupoux/Library/CloudStorage/OneDrive-DrexelUniversity/DSCI-691-Project-main/data/training/tweets.tsv', sep='\t', header=None,
                     names=['tweet_id', 'tweet'])
classes = pd.read_csv('/Users/stephandupoux/Library/CloudStorage/OneDrive-DrexelUniversity/DSCI-691-Project-main/data/training/class.tsv', sep='\t', header=0)


data = pd.merge(tweets, classes, how='left')

data = data.replace(r'@\w+', '', regex=True)

data = data.replace(r'[^\w\s]', '', regex=True)

# convert label to binary
data = data.replace(['NoADE', 'ADE'], [0, 1])


strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=691)
X = data['tweet'].to_numpy()
y = data['label'].to_numpy()

for train_index, test_index in strat_split.split(X, y):
    print(f"Train index: {train_index}", f"Test index: {test_index}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

train, test = train_test_split(data, test_size=0.2, stratify= data.label,random_state=691)
print(f"No. of training examples: {train.shape[0]}")
print(f"No. of testing examples: {test.shape[0]}")

n_epochs = 250
lr = 0.01
n_folds = 5
lstm_input_size = 1
hidden_state_size = 30
batch_size = 30
num_sequence_layers = 2
output_dim = 11
num_time_steps = 4000
rnn_type = 'LSTM'

import os
import torch
import torch.nn as nn
import time
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 2
batch_size = 100
num_epochs = 2
learning_rate = 0.003


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

models = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=X_Train_Array,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=X_Test_Array,
                                          batch_size=batch_size, 
                                          shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for batch, d in enumerate(train_loader):
        X = train_loader.dataset
    
        Y = torch.from_numpy(X[:, 50])
        X = torch.from_numpy(X[:, :50])
        
        Y = torch.flatten(Y)
        X = torch.flatten(X)
        
        # Forward pass
        outputs = models(X)
        loss = criterion(outputs, Y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

benis = np.hstack((X_Train_Array, y_train.reshape(-1,1)))

train_loader = torch.utils.data.DataLoader(dataset=benis,
                                           batch_size=batch_size, 
                                           shuffle=True)




X = train_loader.dataset
X = X[:, :50]



for batch, d in enumerate(train_loader):
    print(d)
    print(batch)



indicies = torch.tensor([0, 2])
benis = torch.index_select(d, 1, indicies)

benis = train_loader.dataset
benis[:, 50]

