import requests
import streamlit as st
import os
access_key = os.getenv("MARKETSTACK")
symbols = 'AAPL'
interval = "15min"
fields = {'access_key': access_key, 'symbols': symbols, "interval": interval}
response = requests.get("http://api.marketstack.com/v1/eod", params=fields)
result = response.json()

import pandas as pd
import matplotlib.pyplot as plt
st.title("Daily Trading")
# Create a DataFrame
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame
df = pd.DataFrame(api_response['data'])

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df.date)
AAPL = df[df['symbol']=='AAPL']
# Plot the data
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Plot OHLC
AAPL[['open', 'high', 'low', 'close', 'date']].plot(ax=axes[0], x='date', kind='line', title='OHLC Prices')
axes[0].set_ylabel('Price')

# Plot Volume
AAPL['volume'].plot(ax=axes[1], x='date', kind='bar', title='Volume')
axes[1].set_ylabel('Volume')

plt.tight_layout()
plt.show()
st.pyplot(plt)

import matplotlib.pyplot as plt

# Assuming 'AAPL' is your DataFrame
mean_100 = AAPL['open'].rolling(window=10, center=True).mean()
mean_50 = AAPL['open'].rolling(window=5, center=True).mean()

# Customizing the figure
plt.figure(figsize=(20, 7))

# Plotting the original data
plt.plot(AAPL['open'], color='blue', label='Original')

# Plotting the moving averages
plt.plot(mean_100, color='red', label='MA (Window 10)')
plt.plot(mean_50, color='green', label='MA (Window 5)')

# Filling between the two moving averages
plt.fill_between(mean_100.index, mean_100, mean_50, where=(mean_100 > mean_50), interpolate=True, color='gray', alpha=0.3, label='Difference Area')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('AAPL Stock Prices with Moving Averages')

# Adding grid lines
plt.grid(True, linestyle='--', alpha=0.6)

# Adding legend
plt.legend()
st.pyplot(plt)
