import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math

# Load data
df = pd.read_csv("C:/Users/sujan/Downloads/insurance_data.csv")
print(df.head())

# Plotting
plt.scatter(df.age, df.bought_insurance, marker="*", color="red")
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
plt.title("Age vs Insurance Buying")
plt.show()

# Train-test split
x_tr, x_te, y_tr, y_te = train_test_split(df[['age']], df.bought_insurance, train_size=0.8, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(x_tr, y_tr)

# Predict
y_pre = model.predict(x_te)
print("Predictions:", y_pre)

# 🔧 Fix your sigmoid and prediction manually
def sig(x):
    return 1 / (1 + math.exp(-x))  # 🛠️ added colon and minus sign

def predi(a):
    z = model.coef_[0][0] * a + model.intercept_[0]  # uses real model values
    y = sig(z)
    return y

# Test prediction for age 56
a = 56
print(f"Probability of buying insurance at age {a}: {predi(a):.2f}")
