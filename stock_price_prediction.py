import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

st.set_page_config(layout="centered")
st.title("S&P 500 Stock Movement Predictor")
st.markdown("Predict whether the **S&P 500** will go **UP or DOWN** tomorrow using a machine learning model.")

@st.cache_data
def load_data():
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Close"] < sp500["Tomorrow"]).astype(int)
    sp500 = sp500.loc["1990-01-04":].copy()
    return sp500

def train_model(data):
    predictors = ["Close", "Open", "High", "Low", "Volume"]
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    precision = precision_score(test["Target"], preds)
    return model, predictors, precision

def predict_next_day(model, data, predictors):
    latest_data = data.iloc[-1:][predictors]
    prediction = model.predict(latest_data)[0]
    probability = model.predict_proba(latest_data)[0]
    return prediction, probability

sp500 = load_data()

st.subheader("S&P 500 Closing Price History")
fig, ax = plt.subplots()
sp500["Close"].plot(ax=ax, figsize=(10, 4))
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.grid(True)
st.pyplot(fig)

model, predictors, precision = train_model(sp500)

st.subheader("🔮 Model Prediction for Tomorrow")
prediction, prob = predict_next_day(model, sp500, predictors)

if prediction == 1:
    st.success("The model predicts the S&P 500 will go **UP** tomorrow.")
else:
    st.error("The model predicts the S&P 500 will go **DOWN or stay the same** tomorrow.")

st.metric("Confidence (UP)", f"{prob[1]*100:.2f}%")
st.metric("Confidence (DOWN)", f"{prob[0]*100:.2f}%")
st.markdown(f"🧠 Model Precision on Test Data: **{precision*100:.2f}%**")
