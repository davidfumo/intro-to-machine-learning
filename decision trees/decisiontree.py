# -*- coding: utf-8 -*-
# import numpy for array manipulations
import numpy as np

# Load sklearn datasets
from sklearn.datasets import load_iris

# Laod tree module
from sklearn import tree

# load iris dataset
iris = load_iris()

# features
features = iris.feature_names
print(features)

# Feature matrix
X = iris.data

# Display 3 instances/examples
print(X[[1, 50, 145]])

# remove comment below if you want to see the entire matrix X
# print(X)

# Target labels
# y is already in numeric form, no need for extra data transformation
y = iris.target



# Display 3 labels
print(y[[1,50,145]])


# Display the names of 1, 50 and 145 ith examples
names = iris.target_names
for i in y[[1,50,145]]:
    print(names[i])


# Display the entire vector y
print(y)


# split data
# leave 2 examples from each specie 
test_idx = [0,1, 50,51, 100,101]

# training data (without the idx 0,1,50,51,100,101)
train_target = np.delete(y, test_idx)
train_data = np.delete(X, test_idx, axis=0)

# testing data
test_target = y[test_idx]
test_data = X[test_idx]

# Decision Tree Classifier
# Here we are initializing using the default parameters
clf = tree.DecisionTreeClassifier()

# Model Training
# In sk-learn we train a algorithm using the fit method and passing
# feature matrix (X) and output labels (Y)
clf.fit(train_data, train_target)



# predict new instances (here we are using the test data)
y_pred = clf.predict(test_data)

# Display predicted values
print(y_pred)


# Display the true values 
print(test_target)

