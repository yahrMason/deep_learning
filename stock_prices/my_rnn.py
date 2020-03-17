'''
RNN Model:
    Predicting the stock price of a chosen stock

'''
# Dependencies
import datetime as dt
import pandas as pd
import yfinance as yf
import numpy as np
import math
import matplotlib.pyplot as plt

# Stock Parameters
ticker = 'CGC'
start = '2018-11-01'
end = '2020-01-01'


'''
Step 1 - Data download and processing
'''

# Download pricing data
total_dat = yf.Ticker(ticker).history(period='max').iloc[:,1:2].values

# Train-Test split
train_part = 0.80
train_split = np.arange(math.floor(train_part*len(total_dat)))
test_split = np.arange(math.floor(train_part*len(total_dat)), len(total_dat))

train_dat = total_dat[train_split]
test_dat = total_dat[test_split]

# interoplate missing values
def interp_nan(x):
    array = x[:,0]
    indicies = np.where(np.isnan(array))[0]
    indicies = [int(x) for x in indicies]
    for ix in indicies:
        if (ix != 0) & (ix != len(array)):
            array[ix] = (array[ix-1]+array[ix+1])/2
        elif ix == len(array):
            array[ix] = array[ix-1]
        else:
            array[ix] = array[ix+1]
    return array.reshape((len(array),1))

train_dat = interp_nan(train_dat)
test_dat = interp_nan(test_dat)

# Feature Scaling
# We will use "Normalization" (value-min)/range instead of standardization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
train_dat_scaled = sc.fit_transform(train_dat)
# will feature scale the test data later


# ** DEFINING THE TIME STEPS FOR THE RNN **
# Creating a data structure with n timesteps and 1 output
ts = 30 # timesteps
X_train = []
y_train = []
for i in range(ts, len(train_dat_scaled)):
    X_train.append(train_dat_scaled[i-ts:i, 0])
    y_train.append(train_dat_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# this is where we scale the test data
inputs = total_dat[len(train_dat) - len(test_dat) - ts:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(ts, ts+len(test_dat)):
    X_test.append(inputs[i-ts:i, 0])
X_test = np.array(X_test)


# Reshaping
# ** we need to reshape to a 3D tensor as that is what is expected
# add that last one in order to signify how many indicators we have
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



'''
Part 2 - Building the RNN
'''
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# will only specify the last 2 diminensions because the observations dimension (0) is already assumed
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fifth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Summary
regressor.summary()

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 15, batch_size = 20)

# predict future price
pred_dat = regressor.predict(X_test)
pred_dat = sc.inverse_transform(pred_dat) 
pred_dat = np.concatenate((train_dat[:,0], pred_dat[:,0]))


# Visualising the results
plt.plot(pred_dat, color = 'blue', label = f'Predicted {ticker} Stock Price')
plt.plot(total_dat, color = 'red', label = f'Real {ticker} Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(f'{ticker} Stock Price')
plt.legend()
plt.show()

