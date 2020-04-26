# -*- coding: utf-8 -*-
"""
Linear regression

y = mx + b
    OR
y = b0 + b1*x1

where 
    m | b1 = slope
    x | x1 = independent variable
    b | b0 = constant
    y = dependent variable
"""


# Data Preprocessing

import numpy as np
import pandas as pd

# Importing the libraries
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Taking care of missing data
# Note: Step Not required because dataset does not contain any empty 
# cell/values.
 
# Encoding categorical data
# Note: Not required because in this dataset there's no such thing to encode. 
# No categorical values

# Preparing train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

# Feature scaling using standard scaler
# NOTE: this step is not required here because linear regressor contains 
# inbuilt feature scaler.

# Preparing linear regression model
from sklearn.linear_model import LinearRegression
#training the model
l_reg = LinearRegression().fit(X_train, y_train)

# predicting results using test data
y_pred = l_reg.predict(X_test)

# checking results with graphs
from matplotlib import pyplot as plt
plt.scatter(X_test, y_test, c="red", marker="x")
plt.plot(X_test, l_reg.predict(X_test), c="blue")
plt.title("Salary vs Experience (test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
