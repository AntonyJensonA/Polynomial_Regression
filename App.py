import streamlit as st
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import joblib

model = joblib.load("Bill_model.pkl")
st.title("Electricity Bill Prediction")
ac_unit = st.number_input(
    "Enter the AC's unit consumption",
    min_value =1.0 ,
    value =1.0,
    step= 1.0
)
fan_unit = st.number_input(
    "Enter the Fan's unit consumption",
    min_value =1.0 ,
    value =1.0,
    step= 1.0
)


if st.button("Predict"):
    if ac_unit < 1 or ac_unit > 150:
        st.error("AC units should be between 0 and 150 units")
    elif fan_unit < 1 or fan_unit > 150:
        st.error("Fan units should be between 0 and 150 units")
        
    else:
        input_data = pd.DataFrame({
            "AC_Units": [ac_unit],
            "Fan_Units" :[fan_unit]
        })
        poly = PolynomialFeatures(degree=2)
        new_poly_df = poly.fit_transform(input_data)
        prediction = model.predict(new_poly_df)
        pred = prediction[0]
        st.success(f"Predicted Price: ₹{pred:.2f}")
