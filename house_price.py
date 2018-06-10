# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:21:57 2018

@author: cege-cts-viso1
"""
import pandas as pd
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
currentWd = 'C:/Users/cege-cts-viso1/Documents/Python Scripts/ML/Melb_House_Price/'
os.chdir(currentWd)

melb_data = pd.read_csv('melb_data.csv')
data = pd.read_csv('train.csv')

print(melb_data.describe())
print(data.describe())


print(melb_data.head())
# select two columns
columns_of_interest = ['Landsize', 'BuildingArea']
# or
columns_of_interest = melb_data.columns[[13, 14]]
# then extract two columns from the pandas data frame
two_columns = melb_data[columns_of_interest]
# and describe the two columns as
print(two_columns.describe())
# select target variable as y
y = melb_data.Price

# select predictors as X from a list of columns called melb_predictors
melb_predictors = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_data[melb_predictors]

# define the model
melb_model = DecisionTreeRegressor()
melb_model.fit(X, y)

#--------------------------------------------------
# Make predictions for the first rows
#--------------------------------------------------

print('Make predictions for the following 5 houses:')
print(X.head())
print('The predictions are:')
predicted_home_prices = melb_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melb_model = DecisionTreeRegressor()
melb_model.fit(train_X, train_y)
# save predicted values
val_predictions = melb_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))



new_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath','BedroomAbvGr', 'TotRmsAbvGrd']
Z = data[new_predictors]
y = data.SalePrice

my_model = DecisionTreeRegressor()
my_model.fit(Z, y)
my_model.predict(Z.head())
