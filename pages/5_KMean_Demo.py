import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('goldstock.csv')
df = df.drop('Unnamed: 0', axis=1)
df['Date'] = pd.to_datetime(df['Date'])

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

col_selection = ['Date', 'Close', 'Volume', 'Open', 'High', 'Low']
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

(df[col_selection]
.assign(Date = pd.to_datetime(df.Date))
.sample(5))

def transform_data(data):
    col_selection = ['Date', 'Close', 'Volume', 'Open', 'High', 'Low']
    
    return (df[col_selection]
            .assign(Date = pd.to_datetime(df.Date)))

train_transf = transform_data(train_df)
num = train_transf.select_dtypes(['int', 'float'])
scaler = StandardScaler()
scaled_X = scaler.fit_transform(num)

model = KMeans(n_clusters=2, n_init='auto')
cluster_labels = model.fit_predict(scaled_X)
num['Cluster'] = cluster_labels

st.title("Cluster Identification")
st.write("We trying in this demo to identify correlation or cluster between each column.")
plt.figure(figsize=(10, 8))
sns.heatmap(num.corr(), annot=True)
plt.title('Correlation Heatmap')
st.pyplot(plt)  # Display the figure using st.pyplot()
st.write("We can identify in this demo that there is two mass of different cluster.")
# Scatter plot 1
plt.figure(figsize=(10, 8))
sns.scatterplot(data=num, x=num.Volume, y=num['Close'], hue='Cluster')
plt.title('Scatter Plot - Volume vs. Close with Clusters')
st.pyplot(plt)  # Display the figure using st.pyplot()

st.write("We can now search why there is this separation, probably Covid situation grow up the gold value.")
plt.figure(figsize=(10, 8))
sns.scatterplot(data=train_transf, x='Date', y='Open', hue=num['Cluster'].astype(str))
plt.title('Scatter Plot - Date vs. Open with Clusters')
st.pyplot(plt) 