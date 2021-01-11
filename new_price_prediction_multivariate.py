import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio
import yfinance as yf
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')
# To show Plotly graphs in browser
pio.renderers.default = "browser"

# get stock data
# download data from Yahoo Finance API
stock = yf.Ticker("PFE")  # Company name
# get last 5 years of data
stock_data = stock.history(period="24Y")

# plotly to show data
fig = px.line(stock_data, x=stock_data.index, y="Close", title="Date Vs CLose Price")
fig.show()

# data = stock_data.filter(['Close'])
data = stock_data.copy()
data = data.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1)
# Converting the dataframe to a numpy array
dataset = data.values

# defining train and test sizes for splitting of data
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size

# scale data between 0 to 1, so as to make it on the same scale.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the scaled training data set
train_data = scaled_data[0:train_size, :]

time_step = 90  # these many days of data will be considered to predict the next day's data. The more the better

# Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(time_step, len(train_data)):
    x_train.append(train_data[i - time_step:i, :])
    y_train.append(train_data[i, :])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data into the shape accepted by the LSTM (3D data)
'''
The number of features is equal to the number of features in the 2D dataset. So, the word “ input_dim” in the 3D tensor
of the shape [batch_size, timesteps, input_dim] means the number of the features in the original dataset. 
In our example, the “input_dim”=2
https://towardsdatascience.com/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00
'''
x_train = np.reshape(x_train, (x_train.shape[0], time_step, 4))
y_train = np.reshape(y_train, (y_train.shape[0], 4))
# y_train = np.reshape(y_train, (y_train.shape[0],time_step,4))

# Test data set
test_data = scaled_data[train_size - time_step:, :]

x_test = []
y_test = []
for i in range(time_step, len(test_data)):
    x_test.append(test_data[i - time_step:i, :])
    y_test.append(test_data[i, :])

# Convert x_test to a numpy array
x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape the data into the shape accepted by the LSTM (3D data)
x_test = np.reshape(x_test, (x_test.shape[0], time_step, 4))
y_test = np.reshape(y_test, (y_test.shape[0], 4))

# Build the LSTM network model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 4)))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=100))
model.add(Dense(units=4))

# Compile the model
adam = optimizers.adam(lr=0.0001)
model.compile(optimizer=adam, loss='mean_squared_error')
print(model.summary())
# Train the model

epochs = 150
batch_size = 64
verbose = 1
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min', verbose=1)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose,
          callbacks=[early_stopping])

# Getting the models predicted price values on Test set
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling

# Calculate/Get the value of RMSE
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print("Test set RMSE: ", rmse)

# Getting the models predicted price values on Test set
predictions_train = model.predict(x_train)
predictions_train = scaler.inverse_transform(predictions_train)  # Undo scaling

# Calculate/Get the value of RMSE
rmse_train = np.sqrt(np.mean(((predictions_train - y_train) ** 2)))
print("Train set RMSE: ", rmse_train)

# Get price for next days:
# Get the last timestep days closing price
new_df = stock_data.filter(['Open', 'High', 'Low', 'Close'])
last_timestep_days = new_df[-time_step:].values
pred_price = np.array([])
for day in range(1, 6):
    if day != 1:
        last_day_predicted_data = np.array(
            [round(pred_price[0][0], 2), round(pred_price[0][1], 2), round(pred_price[0][2], 2),
             round(pred_price[0][3], 2)])
        last_timestep_days = np.concatenate((last_timestep_days, [last_day_predicted_data]))
        last_timestep_days = np.delete(last_timestep_days, 0,
                                       axis=0)  # to remove the first row after adding lastest day predicted data
    # Scale the data to be values between 0 and 1
    last_timestep_days_scaled = scaler.transform(last_timestep_days)
    # Create an empty list
    X_test = []
    # Append teh past 60 days
    X_test.append(last_timestep_days_scaled)
    # Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    # Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
    # Get the predicted scaled price
    pred_price = model.predict(X_test)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    print("\nDay {} Predictions: \n Open: {}, \n High: {}, \n Low: {}, \n Close: {}".format(day, pred_price[0][0],
                                                                                            pred_price[0][1],
                                                                                            pred_price[0][2],
                                                                                            pred_price[0][3]))
