#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 15:16:25 2018

@author: Gaurav Gahukar
module edited-at-night-train station
"""

#Phase - 1 : importing dependencies 
#importting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Phase - 1 :
#impotting traing datasets
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values


#feature sclling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)


#getting the inputs and output
x_train = training_set[0:1257]
y_train = training_set[1:1258]

#reshaping
x_train = np.reshape(x_train, (1257, 1,1))


# PART 2 - BUILDING OF RNN FOR DATA FEEDING

#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#initializing the RNN
model_rnn_regressor = Sequential()

#adding the input layer and Lstm Layer
model_rnn_regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

#adding the output layer
model_rnn_regressor.add(Dense(units = 1))

#Compiling the RNN and...
model_rnn_regressor.compile(optimizer= 'adam', loss= 'mean_squared_error')

#fitting the RNN to the Training set
model_rnn_regressor.fit(x_train, y_train, batch_size = 32, epochs = 200)



# part getting the real stolck ptice of
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values


#Phase - 3 : 
#getting the prodicted stock price of
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, [20,1,1])
predicted_stock_price = model_rnn_regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Phase - 4 : Visualisation of Resultss
#Visualisation the result
plt.plot(real_stock_price, color='red', label ='Real GoogleStock Price')
plt.plot(predicted_stock_price, color='blue', label= 'Predicted Google Stock price')
plt.title('Google Stock ptice Prediction ')
plt.xlabel('time')
plt.ylabel('Google Stock PRice ')
plt.legand()
plt.show()
