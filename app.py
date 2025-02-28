import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset (Boston Housing as an example)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target * 100000  # Convert to realistic price scale

# Train a simple model
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Load model
def load_model():
    with open("house_price_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Streamlit App
st.title("🏡 House Price Prediction Dashboard")
st.write("Enter house details below to predict the price.")

# User Input Fields
median_income = st.number_input("Median Income ($1000s)", min_value=1.0, max_value=15.0, value=5.0)
house_age = st.number_input("House Age", min_value=1.0, max_value=50.0, value=10.0)
avg_rooms = st.number_input("Average Rooms per House", min_value=1.0, max_value=10.0, value=6.0)
population = st.number_input("Population", min_value=100.0, max_value=50000.0, value=2000.0)
ocean_proximity = st.number_input("Near Ocean (1 for Yes, 0 for No)", min_value=0, max_value=1, value=0)

# Prediction Button
if st.button("Predict Price 💰"):
    input_data = np.array([[median_income, house_age, avg_rooms, population, ocean_proximity]])
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")

# Display Model Performance
st.sidebar.header("📊 Model Performance")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.sidebar.write(f"Mean Squared Error: {mse:,.2f}")
