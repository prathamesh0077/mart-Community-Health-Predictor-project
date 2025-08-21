




import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

import matplotlib.pyplot as plt

st.markdown("""
    <style>
    /* Gradient full-page background */
    .stApp {
        background: linear-gradient(to right top, #2193b0, #6dd5ed);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        padding: 2rem;
    }

    /* Style all titles */
    h1, h2, h3, .stHeader, .stSubheader {
        color: #ffffff;
        font-weight: 700;
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #000000;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #90caf9;
    }

    /* Button styling */
    .stButton > button {
        background-color: #0d47a1;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 16px;
    }

    .stButton > button:hover {
        background-color: #1565c0;
        color: #e3f2fd;
    }

    /* Dataframe and other container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Chart styling tweaks */
    .element-container svg {
        background-color: white !important;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)



model = load_model('Stock Predictions Model.keras')


# model = load_model('D:\Data Scince\stock prediction\Stock Predictions Model.keras')


st.header('Stock Market Predictor ')

# ✅ Suggested stock symbols for user convenience
stock_options = {
    "Apple Inc. (AAPL)": "AAPL",
    "Tesla Inc. (TSLA)": "TSLA",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Meta Platforms Inc. (META)": "META",
    "NVIDIA Corporation (NVDA)": "NVDA",
    "Netflix Inc. (NFLX)": "NFLX",
    "JPMorgan Chase & Co. (JPM)": "JPM",
    "Intel Corporation (INTC)": "INTC",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS"
}

st.subheader("🔍 Choose or Enter a Stock Symbol")

# ✅ Let user choose from dropdown or enter manually
use_dropdown = st.radio("Select input method:", ["Dropdown", "Manual"])

if use_dropdown == "Dropdown":
    stock_label = st.selectbox("📊 Select a Stock", list(stock_options.keys()))
    stock = stock_options[stock_label]  # Symbol like "TSLA"
else:
    stock = st.text_input("✍️ Enter Stock Symbol", "TSLA")

# Auto-refresh every 12 hours
st_autorefresh(interval=43200000, key="datarefresh")


start = '2015-01-01'
end = datetime.today().strftime('%Y-%m-%d')

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data.sort_index(ascending=False))
#st.write(data)

data_train = pd.DataFrame(data['Close'][0: int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler(feature_range=(0,1))

pass_100_days = data_train.tail(100)
data_test = pd.concat([pass_100_days, data_test],ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# moving average
st.subheader('Price vs MA50 ')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, label='Ma_50_days', color= 'red')
plt.plot(data.Close,   label = 'Closing value', color = 'green')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,  label='MA_50_days', color= 'red')
plt.plot(ma_100_days, label=' MA_100_days', color='blue')
plt.plot(data.Close,  label = 'Closing value', color='green')
plt.legend()
plt.show()
st.pyplot(fig2)


st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days,  label = 'MA_100_days', color='red')
plt.plot(ma_200_days,  label='MA_200_days',color='blue')
plt.plot(data.Close, label='Closing value', color='green')
plt.legend()
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict*scale
y = y*scale

st.subheader('Original Price vs Predicted Price')

fig4 = plt.figure(figsize=(8,6))
plt.plot(predict ,  label='Original Price', color='red')
plt.plot(y,  label = 'Predicted Price', color='green' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

