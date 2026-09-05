import streamlit as st
import numpy as np
import joblib

# Set the title of the web app
st.title("🌊 Ocean Wave Wavelength Predictor")
st.write("Enter sensor values to predict the wavelength.")

# Load the saved model and scaler
scaler = joblib.load('scaler.joblib')
xgb_model = joblib.load('xgb_model.joblib')

# Create input boxes for the wave sensor data
st.header("Input Wave Sensor Data")

wave_height = st.number_input("Wave Height (meters)", value=1.5)
dominant_period = st.number_input("Dominant Period (seconds)", value=8.0)
average_period = st.number_input("Average Period (seconds)", value=6.5)
wave_direction = st.number_input("Mean Wave Direction (degrees)", value=180.0)
water_temp = st.number_input("Water Temperature (°C)", value=20.0)

# Predict button
if st.button("Predict Wavelength"):
    # Repeat inputs across 6 timesteps to match the model's expected shape
    single_step = [wave_height, dominant_period, average_period, wave_direction, water_temp]
    raw_inputs = np.array([single_step] * 6, dtype=np.float32)
    
    # Scale inputs and flatten for XGBoost
    scaled_inputs = scaler.transform(raw_inputs)
    ml_input = scaled_inputs.reshape(1, -1)
    
    # Generate prediction
    prediction = xgb_model.predict(ml_input)[0]
    
    # Display result
    st.success(f"Predicted Wavelength: **{prediction:.2f} meters**")
