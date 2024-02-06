import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm

df = pd.read_csv('goldstock.csv')
df = df.drop('Unnamed: 0', axis=1)
df['Date'] = pd.to_datetime(df['Date'])

st.title("# Gold Data plots")
st.subheader("Few plots show the data distribution, these data are extracted from the stock market price of gold \n Date: A unique date for each trading day recorded.")
st.write("Close: The closing price of gold on the relevant date.")
st.write("Volume: Gold trading volume on the relevant date.")
st.write("Open: The opening price of gold on the relevant date.")
st.write("High: The highest recorded price of gold during the trading day.")
st.write("Low: The lowest price recorded for gold in the trading day.")
for col in df.columns:
    if df[col].dtype != 'datetime64[ns]':
        sns.histplot(df[col], bins=20, kde=True)
        plt.xlabel(col)
        plt.ylabel('count')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()  # Use Streamlit's st.pyplot() instead of plt.show()

st.write('Gold Trading Volume on the relevant Year')
plt.figure(figsize=(15, 9))
sns.lineplot(data=df, x='Date', y='High', label='High')
sns.lineplot(data=df, x='Date', y='Low', label='Low')
plt.legend()
plt.title('LinePlot of High & Low Price')
plt.ylabel('Prices')
plt.xlabel('Year')
st.set_option('deprecation.showPyplotGlobalUse', False)
# Display the plot using st.pyplot()
st.pyplot()