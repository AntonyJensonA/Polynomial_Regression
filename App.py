import streamlit as st
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import joblib

model = joblib.load("Bill_model.pkl")
st.title("Electricity Bill Prediction")

ac_unit = st.number_input(
    "Enter the AC unit consumption")


if st.button("Predict"):
    input_data = pd.DataFrame({
        "AC_Units": [ac_unit]
    })
    poly = PolynomialFeatures(degree=2)
    new_poly_df = poly.fit_transform(input_data)
    prediction = model.predict(new_poly_df)
    pred = prediction[0]
    st.success(f"Predicted Price: ₹{pred:.2f}")
