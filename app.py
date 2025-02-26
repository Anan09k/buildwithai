from joblib import load
from numpy import array
import numpy
import streamlit as st

model = load("model.pkl")

st.title("Placement Package Prediction")
st.write("Enter CGPA")

cgpa_input = st.number_input("CGPA",max_value=10.0,min_value=0.0,step=0.1)

if st.button("Predict"):
    inputf = numpy.array([[cgpa_input]])  # Ensure input is a NumPy array
    prediction = model.predict(inputf).item()  # Extract single value
    st.success(f"Predicted package: {prediction:.3f}")  




