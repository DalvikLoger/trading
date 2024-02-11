import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

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

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)

dflog = np.log(df)
airpass_ma = df.rolling(window = 120, center = True).mean()
st.title('Moving average and Autocorrelation')
st.write("The moving average allows us to identify the trend which is increasing in our case.")
# Plot the original data and moving average using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(df, color='blue', label='Origine')
plt.plot(airpass_ma, color='red', label='Moving Average')
plt.legend()
plt.title('Moving Average')
st.pyplot(plt)

st.write("One of the limitations of the ARMA model is that it can only model stationary processes. To see if a series is stationary, we can look at its autocorrelation diagram. For a stationary process, the simple autocorrelation decreases rapidly towards 0.")
fig, ax = plt.subplots()
ax = pd.plotting.autocorrelation_plot(dflog)
lag = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
ax.set_xticklabels(lag, ha='right')
st.pyplot(fig)
st.write("We see that the decay of the autocorrelation function is relatively slow. We therefore apply a differentiation of order 1 to our time series in order to see if this allows us to stationarize it.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7)) # Création de la figure et des axes

dflog_1 = dflog.diff().dropna() # Différenciation ordre 1

dflog_1.plot(ax = ax1) #Série temporelle différenciée

pd.plotting.autocorrelation_plot(dflog_1, ax = ax2); #Autocorrélogramme de la série différenciée
st.pyplot(fig)
st.write("The simple autocorrelation seems to tend towards 0 but has significant seasonal peaks (We can also see its seasonal patterns directly in the series graph). We will therefore differentiate the time series in such a way as to eliminate seasonality preventing stationarity (no rapid decay towards zero yet)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7)) #Création de la figure et des axes

dflog_2 = dflog_1.diff(periods = 12).dropna() #Différenciation d'ordre 12

dflog_2.plot(ax = ax1) #Série doublement différenciée

pd.plotting.autocorrelation_plot(dflog_2, ax = ax2); #Autocorrélogramme de la série
st.pyplot(fig)
st.write("We arrive here at a fairly satisfactory result despite the few irregular peaks")
st.subheader("SARIMA model")
st.write("The blue zone represents the zone of non-significance of the autocorrelations, this means that for the autocorrelations in this zone: they are not significantly different from 0 from a statistical point of view.")
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

plot_acf(dflog_2, lags = 36, ax=ax1)
plot_pacf(dflog_2, lags = 36, ax=ax2)
plt.show()
st.pyplot(fig)

st.write("We notice that the first peak has a value of 1, in fact it corresponds to a zero shift and therefore serves as a scale for the rest of the autocorrelogram.")
st.write("We notice that both the simple and partial autocorrelation tend towards 0 (apart from seasonal peaks), there does not seem to be any particular cut. We can therefore assume an ARMA(𝑝,𝑞) process. We will therefore start by estimating the non-seasonal part of our time series via an ARMA(1,1).")

import statsmodels.api as sm
import plotly.graph_objs as go
import plotly.express as px

# Assuming df has a datetime index and 'Open' column
# If not, you can set the index using: df.set_index('Your_Date_Column', inplace=True)
# Example: df.set_index('Date', inplace=True)
# If 'Date' is already your index, no need to set it again.

# Ensure the datetime index is set and has a frequency
df.index = pd.to_datetime(df.index)

# Handle duplicate indices by aggregating them (you can change this based on your needs)
df = df.groupby(df.index).mean()  # Example: Take the mean for duplicates

# Set the desired frequency, replace 'D' with your actual frequency
df = df.asfreq('D')

# Fit SARIMA model
model = sm.tsa.SARIMAX(df['Open'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
sarima = model.fit()

# Generate forecast for the year 2024
forecast_start = datetime.datetime(2024, 1, 1)
forecast_end = datetime.datetime(2024, 12, 31)
forecast_steps = (forecast_end - forecast_start).days + 1
forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='D')

# Use integer indices for forecast predictions
forecast_results = sarima.get_forecast(steps=forecast_steps)
forecast_predictions = forecast_results.predicted_mean.values

df['Percentage Change'] = df['Open'].pct_change() * 100

last_open_value = df['Open'].iloc[-1]

plt.figure(figsize=(10, 6))

plt.plot(df.index, df['Open'], label='Original Data')
plt.plot(forecast_index, forecast_predictions, label='Forecast for 2024')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Open')
plt.title('Original Data, Forecast, and Percentage Change in Ratings for 2024')

# Display legend
plt.legend(loc='upper left')
plt.tight_layout()

st.pyplot(plt)