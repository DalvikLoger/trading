import streamlit as st
import os
import warnings
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Above is a special style template for matplotlib, highly useful for visualizing time series data
from pylab import rcParams
from plotly import tools
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error

df = pd.read_csv('goldstock.csv')
df = df.drop('Unnamed: 0', axis=1)
df['Date'] = pd.to_datetime(df['Date'])

st.write("# Graphique des données")

for col in df.columns:
    if df[col].dtype != 'datetime64[ns]':
        sns.histplot(df[col], bins=20, kde=True)
        plt.xlabel(col)
        plt.ylabel('count')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()  # Use Streamlit's st.pyplot() instead of plt.show()

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