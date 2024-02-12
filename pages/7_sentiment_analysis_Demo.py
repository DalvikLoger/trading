API_KEY = '97QT8RYDEG87X94W'
import streamlit as st
symbol = 'AAPL'
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries 
st.title('Sentiment Analysis for Apple trend')
function = 'CURRENCY_EXCHANGE_RATE'
symbol = 'IBM'
interval = '1min'

fields = {'key': API_KEY, 'function': function, 'symbol': symbol, 'interval': interval}
response = requests.get("https://www.alphavantage.co", params=fields)
ts = TimeSeries(key=API_KEY, output_format='pandas')

from_currency='USD'
symbol = 'AAPL'
interval = '15min'
data, meta_data = ts.get_intraday(symbol=symbol, interval=interval)
st.dataframe(data)

mean_100 = data['4. close'].rolling(window = 10, center = True).mean()
mean_50 = data['4. close'].rolling(window = 5, center = True).mean()
#Affichage de la série 
plt.figure(figsize=(20,7))
plt.plot(data['4. close'], color = 'blue', label = 'Origine')
plt.plot(mean_100, color = 'red', label = 'Moyenne mobile Base 10')
plt.plot(mean_50, color = 'green', label = 'Moyenne mobile Base 5')
plt.fill_between(mean_100.index, mean_100, mean_50, where=(mean_100 > mean_50), interpolate=True, color='yellow', alpha=0.3, label='Difference Area')
plt.legend()
plt.title('Moyenne mobiles')
    
import requests
tickers = 'AAPL'
# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey={API_KEY}'
r = requests.get(url)
data = r.json()

try:
    # Make the API request
    r = requests.get(url)

    # Check if the request was successful (status code 200)
    if r.status_code == 200:
        data = r.json()

        # Extract information from the first 10 feed items if available
        feed_items = data.get('feed', [])

        for i, feed_item in enumerate(feed_items[:10]):
            overall_sentiment_score = feed_item.get('overall_sentiment_score')
            category = feed_item.get('category_within_source')
            authors = feed_item.get('authors')
            summary = feed_item.get('summary')
            url = feed_item.get('url')

            st.subheader(f"\nEntry {i + 1}")
            st.write(f"Overall Sentiment Score for {tickers}: {overall_sentiment_score}")
            st.write(f"Overall Category for {tickers}: {category}")
            st.write(f"Overall authors for {tickers}: {authors}")
            st.write(f"Overall summary for {tickers}: {summary}")
            st.write(f"Overall url for {tickers}: {url}")

    else:
        st.write(f"Error: Unable to fetch data. Status code: {r.status_code}")

except Exception as e:
    st.write(f"Error: {e}")