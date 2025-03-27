import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# User input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 100, 30)
fare = st.number_input("Fare", min_value=0.0, step=0.1)
embarked = st.selectbox("Embarked Port", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

# Convert categorical inputs
sex = 0 if sex == "Male" else 1
embarked = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}[embarked]

# Normalize input
input_data = np.array([[pclass, sex, age, fare, embarked]])
input_data = scaler.transform(input_data)  # Apply same scaling as training

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    outcome = "Survived 🟢" if prediction == 1 else "Did not survive 🔴"
    st.write(f"**Prediction:** {outcome}")
