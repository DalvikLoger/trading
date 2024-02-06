import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model

df = pd.read_csv('goldstock.csv')
df = df.drop('Unnamed: 0', axis=1)
df['Date'] = pd.to_datetime(df['Date'])
EPOCHS = 10
BATCH_SIZE = 32
TIMESTEP = 60

st.write("# Test of Prediction with a LSTM model")

def populate_missing_date_values(df):
    start_date = df['Date'][0].date()

    # store the dates as a Series
    dates = df['Date']

    dataset = []
    num_days_in_future = 1

    for index, day in enumerate(dates):
        # extract the date from the current row in the dataframe
        current_df_date = str(day.date())

        # skip the first date
        if (index == 0):
            # add the first date to the dataset
            open_price = df['Open'][index]
            day_step = [current_df_date, open_price]
            dataset.append(day_step)
            continue

        # get the open and close prices
        open_price = df['Open'][index]
        close_price = df['Close'][index - 1]

        # get the date of the next day
        current_date = str(start_date + datetime.timedelta(days=num_days_in_future))

        # check if the current date is the same as the current date in the dataframe
        if (current_date != current_df_date):
            found_next_date = False

            # loop until the next date is found
            while not found_next_date:
                if (current_date == current_df_date):
                    found_next_date = True

                    # add the open price to the dataset
                    day_step = [current_date, open_price]
                    dataset.append(day_step)
                else:
                    # add the close price to the dataset
                    day_step = [current_date, close_price]
                    dataset.append(day_step)

                    # increment the date
                    num_days_in_future += 1
                    current_date = str(start_date + datetime.timedelta(days=num_days_in_future))
            else:
                # add the open price to the dataset
                day_step = [current_date, open_price]
                dataset.append(day_step)

            num_days_in_future += 1

    return dataset

df = df.sort_values(by='Date', ascending=True)

# remove duplicate observations
df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)

# populate missing dates and open price values
df = populate_missing_date_values(df)

# create a dataframe with only Date and Open columns
df = pd.DataFrame(df, columns=['Date', 'Open'])

train_df = df.head(len(df) - 19)
test_df = df.tail(19)
test_df.reset_index(drop=True, inplace=True)
X_train = train_df.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range=(0, 1))

# fit the scaler only on X_train
sc.fit(X_train)

# scale X_train values to between 0 and 1
X_train_scaled = sc.transform(X_train)
train_data = []
train_labels = []

# create the training data using 60 timesteps
for i in range(TIMESTEP, len(X_train_scaled)):
    train_data.append(X_train_scaled[i-TIMESTEP:i, 0])
    train_labels.append(X_train_scaled[i, 0])

# convert train_data and train_labels back into numpy arrays
train_data, train_labels = np.array(train_data), np.array(train_labels)

# reshape train_data to be 3D so its compatible with the RNNs input requirements
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

regressor = load_model('regressor_trading.h5')

last_60_days_2023 = train_df.iloc[:, 1:2].tail(TIMESTEP)
last_60_days_2023 = pd.concat([last_60_days_2023, test_df.iloc[:, 1:2]])
last_60_days_2023 = sc.transform(last_60_days_2023)

X_test = []
for i in range(TIMESTEP, 79):
    X_test.append(last_60_days_2023[i-TIMESTEP:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_value = regressor.predict(X_test)
predicted_value = sc.inverse_transform(predicted_value)

fig_train = plt.figure(figsize=(15, 9))

plt.plot(train_df['Date'], train_df['Open'], color='red')
plt.title('Gold Stock Price Train')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.legend(['Train', 'Test', 'Prediction'], loc='lower right')
st.pyplot(fig_train)

fig_prediction = plt.figure(figsize=(15, 9))
plt.plot(test_df['Date'], test_df['Open'], color='blue')
plt.plot(test_df['Date'], predicted_value, color='green')
plt.title('Gold Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.legend(['Train', 'Test', 'Prediction'], loc='lower right')

# Display the plot in Streamlit
st.pyplot(fig_prediction)
