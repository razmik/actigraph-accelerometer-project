"""
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
"""

import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
root_folder = "E:\Data\Accelerometer_Processed_Raw_Epoch_Data_Unsupervised\outputs/Wrist_LSM110_(2016-10-05)_row_0_to_4149.csv".replace('\\', '/')
cols = ['raw_Wrist_sdvm', 'raw_Wrist_mangle', 'raw_Wrist_XMax', 'raw_Wrist_YMax', 'raw_Wrist_ZMax',
        'raw_Wrist_XMean', 'raw_Wrist_YMean', 'raw_Wrist_ZMean', 'raw_Wrist_X90perc', 'raw_Wrist_Y90perc',
        'raw_Wrist_Z90perc', 'raw_Wrist_X75perc', 'raw_Wrist_Y75perc', 'raw_Wrist_Z75perc']
dataset = read_csv(root_folder, header=0, usecols=cols)
values = dataset.values

dataset_pred = read_csv(root_folder, header=0, usecols=['waist_ee'])
predicting_value = dataset_pred.values

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaled_x = scaler_features.fit_transform(values)

# normalize predicting values
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_y =scaler_y.fit_transform(predicting_value)

# specify the number of lag hours
n_15min_intervals = 4
n_features = 14

# frame as supervised learning
reframed = series_to_supervised(scaled_x, n_15min_intervals, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[list(range(56, 70))], axis=1, inplace=True)  # We predict only waist_ee

# drop the first values in predicting dataset
delete_indexes = list(range(0, n_15min_intervals))
scaled_y = np.delete(scaled_y, delete_indexes)

# add predicting (Y) to the supervised learning frame
reframed['vary(t)'] = scaled_y

# split into train and test sets
values = reframed.values
n_train_hours = int(len(reframed) * (6/10))
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
n_obs = n_15min_intervals * n_features
train_X, train_y = train[:, :n_obs], train[:, -1]  # Select the last column of the dataset as the predicting variable
test_X, test_y = test[:, :n_obs], test[:, -1]
print('Training data set X_shape, length, y_shape', train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_15min_intervals, n_features))
test_X = test_X.reshape((test_X.shape[0], n_15min_intervals, n_features))
print('Training data set X_shape, length, y_shape', train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=400, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_15min_intervals * n_features))

# invert scaling for forecast
# inv_yhat = concatenate((test_X[:, :n_obs], yhat), axis=1)
# inv_yhat = concatenate((yhat, test_X[:, -13:]), axis=1)
inv_yhat = scaler_y.inverse_transform(yhat)
inv_yhat = inv_yhat[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_X[:, :n_obs], test_y), axis=1)
# inv_y = concatenate((test_y, test_X[:, -13:]), axis=1)
inv_y = scaler_y.inverse_transform(test_y)
inv_y = inv_y[:, 0]

## print results
print('\n\nActual Y:-----------------------------------------------------------\n')
for i in inv_y:
    print(i)
print('\n\nPredicted Y:-------------------------------------------------------- \n')
for i in inv_yhat:
    print(i)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)