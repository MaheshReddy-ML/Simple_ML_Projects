"""
🏠 Hyderabad House Rent Predictor
----------------------------------

Author: Mahesh Reddy Sanikommu
Date: June 2025

Description:
This project predicts monthly house rent based on  sqft using linear regression.
The model is built from scratch with NumPy and visualized using Matplotlib.

Features:
- Data cleaning: handles messy strings, ranges, commas
- Manual gradient descent implementation
- Prediction based on user input
- Plotting actual vs predicted rent

"""

#  Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Load the dataset
df = pd.read_csv("Hyderabad_House_Data.csv")
print(df.head())  # Preview first 5 rows

#  Select relevant columns & drop missing values
data = df[["Area", "Price"]].dropna()
print(data.head())

#  Visualize raw data (before cleaning)
plt.scatter(data['Area'], data['Price'], color='red')
plt.xlabel('Area (sqft)')
plt.ylabel('Rent (₹)')
plt.title('Area vs Rent')
plt.show()

#  Check unique Area entries to understand data formatting
print(data["Area"].unique()[:20])

#  Clean Area values
def clean_area(val):
    val = str(val).replace(",", "").lower().replace("sqft", "").strip()
    if '-' in val:
        parts = val.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

data["cleaned_area"] = data["Area"].apply(clean_area)

#  Clean Price values
def clean_price(val):
    val = str(val).replace(",", "").strip().lower()
    try:
        return float(val)
    except:
        return np.nan

data["cleaned_prices"] = data["Price"].apply(clean_price)

#  Remove rows with invalid (NaN) cleaned values
data = data.dropna(subset=["cleaned_area", "cleaned_prices"])

#  Extract feature (area) and label (price) for training
x = data["cleaned_area"].values
y = data["cleaned_prices"].values

#  Initialize model parameters
w, b = 0, 0
lr = 0.0000001
epochs = 1000
n = len(x)

#  Train using Gradient Descent
for i in range(epochs):
    y_pred = w * x + b
    error = y_pred - y
    dw = (2/n) * np.dot(error, x)
    db = (2/n) * np.sum(error)
    w -= lr * dw
    b -= lr * db

# Output final weight & bias
print(f"\nTrained model parameters:")
print(f"Weight (w): {w}")
print(f"Bias (b): {b}")

#  Predict rent based on user input
try:
    area = int(input("\nEnter area in sqft: "))
    predicted = w * area + b
    print(f"Estimated Monthly Rent: ₹{predicted.round()}")
except ValueError:
    print("Invalid input. Please enter a number.")

# Plot Actual vs Predicted
plt.scatter(x, y, color="green", label="Actual Rents")
plt.plot(x, w * x + b, color="red", label="Predicted Line")
plt.xlabel("Area (sqft)")
plt.ylabel("Rent (₹)")
plt.title("Hyderabad House Rent Prediction")
plt.legend()
plt.xlim(0, 5000)
plt.ylim(0, 100000)
plt.show()
