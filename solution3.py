# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 16:50:14 2015

@author: VishnuC
@email: vrajs5@gmail.com
Beating the benchmark for Microsoft Malware Classification Challenge (BIG 2015)
"""
import os
import numpy as np
import gzip
from csv import reader, writer
import six
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def fromPredToPredProb(y_pred):
    y_prob = np.zeros((y_pred.shape[0],9))
    for i in range(0,y_pred.shape[0]):
        j = y_pred[i] - 1
        y_prob[i][j] = 1
    return y_prob


# Decide read/write mode based on python version
read_mode, write_mode = ('r','w') if six.PY2 else ('rt','wt')

# Decide zip based on python version
if six.PY2:
    from itertools import izip
    zp = izip
else:
    zp = zip

# Set path to your consolidated files
path = '/Users/Shared/kaggle/'
os.chdir(path)

# File names
ftrain = 'train_consolidation.gz'
ftest = 'test_consolidation.gz'
flabel = 'trainLabels.csv'
fsubmission = 'submission.gz'

print('loading started')
# Lets read labels first as things are not sorted in files
labels = {}
y_true = []
with open(flabel) as f:
    next(f)    # Ignoring header
    for row in reader(f):
        labels[row[0]] = int(row[1])
print('labels loaded')


# Dimensions for train set
ntrain = 10868
nfeature = 16**2 + 1 + 1 # For two_byte_codes, no_que_marks, label
train = np.zeros((ntrain, nfeature), dtype = int)
with gzip.open(ftrain, read_mode) as f:
    next(f)    # Ignoring header
    for t,row in enumerate(reader(f)):
        train[t,:-1] = map(int, row[1:]) if six.PY2 else list(map(int, row[1:]))
        train[t,-1] = labels[row[0]]
        if(t+1)%1000==0:
            print(t+1, 'records loaded')
print('training set loaded')

del labels

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train[:,:-1],train[:,-1], test_size = 0.2, random_state = 123)

del train

# Parameters for Randomforest
n_jobs = 4
verbose = 2

C_set = [0]
#
for c in C_set:
    clf = GaussianNB()

    # Start training
    print('training started', c)
    clf.fit(X_train, y_train)
    print('training completed', c)

    y_pred = clf.predict_proba(X_train)
    ll = log_loss(y_train, y_pred, eps=1e-15, normalize=True)
    print('log_loss for training set is ', ll)

    y_pred = clf.predict_proba(X_test)
    ll = log_loss(y_test, y_pred, eps=1e-15, normalize=True)
    print('log_loss for test set is ', ll)
'''

clf = PassiveAggressiveClassifier(n_jobs = n_jobs)

print('training started')
clf.fit(X_train, y_train)
print('training completed')

y_pred = clf.predict(X_train)
ll = log_loss(y_train, fromPredToPredProb(y_pred), eps=1e-15, normalize=True)
print('log_loss for training set is ', ll)

y_pred = clf.predict(X_test)
ll = log_loss(y_test, fromPredToPredProb(y_pred), eps=1e-15, normalize=True)
print('log_loss for test set is ', ll)

'''
