"""Deep Neural Network
   Predicting the age of abalone from physical measurements.
"""
__author__ = "Bhavesh Kumar"
__email__  = "er.bhavesh@live.com"
__github__ = "https://github.com/bhaveshkumarraj"

import numpy as np
import urllib.request as urllib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  
from keras.models import Sequential
from keras.layers import Dense

# Preprocess data
labelEncoder = LabelEncoder()
oneHotEncoder = OneHotEncoder(categorical_features=[0])
ss = StandardScaler()

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
raw_data = urllib.urlopen(url)
DataMatrix = np.loadtxt(raw_data, dtype=str, delimiter=',')
X = DataMatrix[:,:8]
y = DataMatrix[:,8]
X[:,0] = labelEncoder.fit_transform(X[:,0])
X = oneHotEncoder.fit_transform(X).toarray()
X_train, X_test, y_train, y_test = train_test_split(X,y)
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

scaler = MinMaxScaler().fit(y_train.reshape(-1,1))
y_train = scaler.transform(y_train.reshape(-1,1))
y_test  = scaler.transform(y_test.reshape(-1,1))


# Neural Network
prediction_network = Sequential()
prediction_network.add(Dense(units=10, kernel_initializer='uniform', activation='relu', input_dim=10))
prediction_network.add(Dense(units=10, kernel_initializer='uniform', activation = 'relu'))
prediction_network.add(Dense(units=1, kernel_initializer='uniform'))
prediction_network.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Fitting model
prediction_network.fit(X_train, y_train, batch_size=10, epochs=50)

# Predict the test set
y_pred = prediction_network.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print("Root Mean Square Error : {:.4f}".format(rmse))











