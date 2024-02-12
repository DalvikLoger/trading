API_KEY = alpha_vantage
import streamlit as st
symbol = 'AAPL'
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries 
st.title('Sentiment Analysis for Apple trend')
try:
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='compact')
    
    # Display the data
    st.dataframe(data)

    # Plot the closing prices
    plt.figure(figsize=(10, 6))
    data['4. close'].plot(title=f'{symbol} Intraday Closing Prices (1 Minute Intervals)')
    plt.xlabel('Time')
    plt.ylabel('Close Price (USD)')
    plt.show()
    st.pyplot(plt)
except ValueError as e:
    st.write(f"Error: {e}")
    
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
