import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Bill_model.pkl")
st.title("Electricity Bill Prediction")

ac_unit = st.number_input(
    "Enter the AC unit consumption")


if st.button("Predict"):
  input_data = pd.DataFrame({
    "AC_Units": [ac_unit]
  })
  prediction = model.predict(input_data)
  pred = prediction[0]
  st.success(f"Predicted Price: ₹{pred:.2f}")
