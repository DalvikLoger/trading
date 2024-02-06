import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import math
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdate
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional

df = pd.read_csv('goldstock.csv')
df = df.drop('Unnamed: 0', axis=1)
df['Date'] = pd.to_datetime(df['Date'])
EPOCHS = 10
BATCH_SIZE = 32
TIMESTEP = 60

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

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

airpass_ma = df.rolling(window = 12, center = True).mean()
dflog = np.log(df)
st.title('Moving average and Autocorrelation')

# Plot the original data and moving average using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(df, color='blue', label='Origine')
plt.plot(airpass_ma, color='red', label='Moving Average')
plt.legend()
plt.title('Moving Average')

# Display the plot using Streamlit
st.pyplot(plt)

ax = pd.plotting.autocorrelation_plot(dflog)
years_to_show = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
ax.set_xticklabels(years_to_show, ha='right')

st.pyplot()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7)) # Création de la figure et des axes

dflog_1 = dflog.diff().dropna() # Différenciation ordre 1

dflog_1.plot(ax = ax1) #Série temporelle différenciée

pd.plotting.autocorrelation_plot(dflog_1, ax = ax2); #Autocorrélogramme de la série différenciée
st.pyplot(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7)) #Création de la figure et des axes

dflog_2 = dflog_1.diff(periods = 12).dropna() #Différenciation d'ordre 12

dflog_2.plot(ax = ax1) #Série doublement différenciée

pd.plotting.autocorrelation_plot(dflog_2, ax = ax2); #Autocorrélogramme de la série
st.pyplot(fig)

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

plot_acf(dflog_2, lags = 36, ax=ax1)
plot_pacf(dflog_2, lags = 36, ax=ax2)
plt.show()
st.pyplot(fig)
