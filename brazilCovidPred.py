import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy.lib.recfunctions as rfn

import streamlit as st
import streamlit as st1
import streamlit as st2

FILEPATH_STATE = os.path.join(os.getcwd(), "", "state.csv")
CASES = "newCases"
DEATHS = "newDeaths"
RECOVERIES = "recovered"
TOTAL = "TOTAL"
_index = 0



state_name = []
with open(FILEPATH_STATE, encoding='cp850') as csvfile:
  reader = csv.reader((line.replace('\0', '') for line in csvfile))
  for row in reader:
  	if(row[1] !='brazile'):
  		state_name.append(row[1])



st.write("# COVID Prediction")

selected_states = st1.selectbox("Select a state:", state_name)
selected_city = st2.selectbox("Select a city:", ['TOTAL'])
selected_series = st.selectbox("Select a data set:", (CASES, DEATHS))


if selected_series == CASES:

    title = "Daily Cases"
    x_label = "Cases"
    _index = 0
if selected_series == DEATHS:

    title = "Daily Deaths"
    x_label = "Deaths"
    _index = 1


features = 2
df = pd.read_csv("corona.txt", usecols = ['newCases','vaccinated_per_100_inhabitants','newDeaths','state', 'city', 'date'])

df = df[(df["state"] == selected_states)]
df = df[(df["city"] == selected_city)]
print(df)
_raw_data  = df.iloc[:,3:5].values
_date_value = df['date'].values


dataset = _raw_data
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, features))
scaler = MinMaxScaler(feature_range = (0, 1))

dataset = scaler.fit_transform(dataset)

last_size = 100
train, last = dataset, dataset[len(dataset) - last_size:len(dataset)]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        X.append(a)
        b = (dataset[(i + look_back)])
        Y.append(b)
    return np.array(X), np.array(Y)
    
look_back = 4

X_train, Y_train = create_dataset(train, look_back)
X_last, Y_last = create_dataset(last, look_back)
last_actual = scaler.inverse_transform(Y_last)
Y_train_act = scaler.inverse_transform(Y_train)
# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], look_back, features))
X_last = np.reshape(X_last, (X_last.shape[0], look_back, features))
# Initialising the RNN

lstm_model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
lstm_model.add(LSTM(units = 45, return_sequences = True, input_shape = (None, features)))
lstm_model.add(Dropout(0.2))

# Adding a second LSTM layer nd some Dropout regularisation
lstm_model.add(LSTM(units = 45, return_sequences = True))
lstm_model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
lstm_model.add(LSTM(units = 45, return_sequences = True))
lstm_model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
lstm_model.add(LSTM(units = 45))
lstm_model.add(Dropout(0.2))

# Adding the output layer
lstm_model.add(Dense(units = features))

# Compiling the RNN
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
lstm_model.fit(X_train, Y_train, epochs = 100 , batch_size = 10, steps_per_epoch = 99)

test_predict = lstm_model.predict(X_train)
p_test = test_predict
test_predict = scaler.inverse_transform(p_test)


title = title + "(" + selected_states + " state)"
plt.figure(figsize=(10,6))
plt.plot(Y_train_act[:, _index], color='blue', label='Actual')
plt.plot(test_predict[:,_index], color='red', label='Predicted data')
plt.title(title)
plt.xlabel('Date')
plt.ylabel(x_label)
plt.legend()
plt.show()

st.pyplot(plt)

_select_date = st.date_input('select date')

_date_index_array = np.where(_date_value==str(_select_date))


st.write("Actual")
_date_index = -1
print(len(_date_index_array[0]))
if len(_date_index_array[0]) !=0:
	_date_index = _date_index_array[0][0]
print(_date_index)

if _date_index > look_back :
	st.write(int(Y_train_act[:,_index][_date_index-look_back]))
else: 
	st.write('no value')
st.write("Prediction")
if _date_index > look_back:
	st.write(int(test_predict[:,_index][_date_index-look_back]))
else:
	st.write('no value')