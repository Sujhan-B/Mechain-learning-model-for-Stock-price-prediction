import pandas as pd
import math
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Predictor", layout="centered")
st.title("🧠 Logistic Regression - Insurance Buying Predictor")
st.markdown("Predict the **probability** of someone buying insurance based on their age.")

# Load the data
df = pd.read_csv("path to the data")

# Show data
with st.expander("📋 View Raw Data"):
    st.dataframe(df)

# Scatter Plot
st.subheader("📈 Age vs Bought Insurance")
fig, ax = plt.subplots()
ax.scatter(df.age, df.bought_insurance, marker='*', color='red')
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
st.pyplot(fig)

# Split and train model
x_tr, x_te, y_tr, y_te = train_test_split(df[['age']], df.bought_insurance, train_size=0.8, random_state=42)
model = LogisticRegression()
model.fit(x_tr, y_tr)

# Prediction function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict_probability(age):
    z = model.coef_[0][0] * age + model.intercept_[0]
    return sigmoid(z)

# User input
st.subheader("🔮 Predict for Custom Age")
age_input = st.number_input("Enter Age", min_value=1, max_value=100, value=25)

if st.button("Predict"):
    prob = predict_probability(age_input)
    st.success(f"Probability of buying insurance at age {age_input}: **{prob*100:.2f}%**")

    if prob > 0.5:
        st.markdown("🟢 Likely to buy insurance.")
    else:
        st.markdown("🔴 Unlikely to buy insurance.")
