import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('goldstock.csv')
df = df.drop('Unnamed: 0', axis=1)
df['Date'] = pd.to_datetime(df['Date'])

df_hourly = df.set_index('Date').resample('H').mean().reset_index()
df_daily = df.set_index('Date').resample('D').mean().reset_index()
df_weekly = df.set_index('Date').resample('W').mean().reset_index()

df_daily['Weekday'] = pd.Categorical(df_daily['Date'].dt.strftime('%A'),
                                     categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
df_daily['Hour'] = df_daily['Date'].dt.hour
df_daily['Day'] = df_daily['Date'].dt.weekday
df_daily['Month'] = df_daily['Date'].dt.month
df_daily['Year'] = df_daily['Date'].dt.year
df_daily['Month_day'] = df_daily['Date'].dt.day
df_daily['Lag'] = df_daily['Open'].shift(1)
df_daily['Rolling_Mean'] = df_daily['Open'].rolling(7, min_periods=1).mean()
df_daily = df_daily.dropna()

# Joining with the mean for each Hour and Weekday
df_daily = df_daily.join(df_daily.groupby(['Hour', 'Weekday'])['Open'].mean(), on=['Hour', 'Weekday'], rsuffix='_Average')

df_daily.dropna(inplace=True)

# Daily
df_daily_model_data = df_daily[['Open', 'Day',  'Month','Month_day','Rolling_Mean']].dropna()

# Hourly
model_data = df_daily[['Open', 'Day', 'Month_day', 'Month','Rolling_Mean','Lag', 'Date']].set_index('Date').dropna()

from sklearn.ensemble import IsolationForest

def run_isolation_forest(model_data: pd.DataFrame, contamination=0.005, n_estimators=200, max_samples=0.7) -> pd.DataFrame:
    
    IF = (IsolationForest(random_state=0,
                          contamination=contamination,
                          n_estimators=n_estimators,
                          max_samples=max_samples)
         )
    
    IF.fit(model_data)
    
    output = pd.Series(IF.predict(model_data)).apply(lambda x: 1 if x == -1 else 0)
    
    score = IF.decision_function(model_data)
    
    return output, score

outliers, score = run_isolation_forest(model_data)

df_daily = (df_daily
             .assign(Outliers = outliers)
             .assign(Score = score)
            )
anomaly_df = df_daily[df_daily['Outliers'] == 1 & (df_daily['Score'] <= 0.2)]

st.title("Anomaly Detection")

# Plot the 'Open' values
plt.figure(figsize=(10, 6))
plt.plot(df_daily['Date'], df_daily['Open'], label='Open')

# Highlight anomaly points
plt.scatter(anomaly_df['Date'], anomaly_df['Open'], color='red', label='Anomaly Points')

plt.title('Gold Stock Open Prices with Anomaly Points')
plt.xlabel('Date')
plt.ylabel('Open Prices')
plt.legend()
plt.show()
st.pyplot(plt) 

st.dataframe(anomaly_df)