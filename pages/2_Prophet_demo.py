import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# Load your data
df = pd.read_csv('goldstock.csv')
df = df.drop('Unnamed: 0', axis=1)
df['Date'] = pd.to_datetime(df['Date'])

st.title("# Prediction with Prophet")

# Prepare data for Prophet
df_close = df[['Date', 'Close']]
df_close.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Create and fit the Prophet model
model_close = Prophet()
model_close.fit(df_close)

# Make future dataframe
future_close = model_close.make_future_dataframe(periods=365)

# Generate forecast
forecast_close = model_close.predict(future_close)

# Plot the forecast
fig_forecast = model_close.plot(forecast_close, figsize=(15, 9), xlabel='Date', ylabel='Close Price')

# Display the chart in Streamlit
st.pyplot(fig_forecast)

fig_predicted = go.Figure()

predicted_forecast = forecast_close[forecast_close['ds'].dt.year == 2024]

# Predict seasonal components for one year
predicted_seasonal_component = model_close.predict_seasonal_components(predicted_forecast)

# Plot the seasonal component for one year
predicted_year_component = go.Figure()
predicted_year_component.add_trace(go.Scatter(x=predicted_forecast['ds'], y=predicted_seasonal_component['yearly'], mode='lines', name='Seasonal'))

predicted_year_component.update_layout(title=f'Prophet Seasonal Component for {2024}', xaxis_title='Date', yaxis_title='Seasonal Component Value')

# Display the components chart for one year in Streamlit
st.plotly_chart(predicted_year_component)