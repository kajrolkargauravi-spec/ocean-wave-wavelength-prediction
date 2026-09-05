import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(
    page_title="Ocean Wave Wavelength Predictor",
    page_icon="🌊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lstm_model.keras")
    scaler = joblib.load("lstm_scaler.joblib")
    return model, scaler

model, scaler = load_model()

st.title("🌊 Ocean Wave Wavelength Predictor")

st.write(
    "This application uses a trained LSTM deep-learning model "
    "to predict ocean wave wavelength from previous wave conditions."
)

st.info(
    "The model uses the previous 6 observations to predict "
    "the wavelength of the next observation."
)

st.header("Enter the Previous 6 Observations")

st.write(
    "For each observation, enter the wave and environmental conditions."
)

observations = []

for i in range(6):

    st.subheader(f"Observation {i + 1}")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        wave_height = st.number_input(
            "Wave Height (m)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key=f"height_{i}"
        )

    with col2:
        dominant_period = st.number_input(
            "Dominant Period (s)",
            min_value=0.1,
            value=8.0,
            step=0.1,
            key=f"dominant_{i}"
        )

    with col3:
        average_period = st.number_input(
            "Average Period (s)",
            min_value=0.1,
            value=6.0,
            step=0.1,
            key=f"average_{i}"
        )

    with col4:
        wave_direction = st.number_input(
            "Wave Direction (°)",
            min_value=0.0,
            max_value=360.0,
            value=180.0,
            step=1.0,
            key=f"direction_{i}"
        )

    with col5:
        water_temperature = st.number_input(
            "Water Temperature (°C)",
            value=25.0,
            step=0.1,
            key=f"temperature_{i}"
        )

    observations.append([
        wave_height,
        dominant_period,
        average_period,
        wave_direction,
        water_temperature
    ])

st.divider()

if st.button(
    "🌊 Predict Wavelength",
    type="primary",
    use_container_width=True
):

    input_data = np.array(
        observations,
        dtype=float
    )

    input_scaled = scaler.transform(
        input_data
    )

    input_sequence = input_scaled.reshape(
        1, 6, 5
    )

    prediction = model.predict(
        input_sequence,
        verbose=0
    )

    wavelength = float(
        prediction[0][0]
    )

    wavelength = max(
        wavelength,
        0
    )

    st.success("Prediction completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Wavelength",
            f"{wavelength:.2f} metres"
        )

    with col2:
        st.metric(
            "Model",
            "LSTM"
        )

    st.subheader("What does this mean?")

    st.write(
        f"""
        Based on the six previous ocean-wave observations,
        the model predicts a wavelength of approximately
        **{wavelength:.2f} metres**.

        Wavelength is the distance between two consecutive
        wave crests.
        """
    )

st.divider()

st.header("About the Research")

st.write(
    """
    This application is part of a research project comparing
    machine-learning and deep-learning approaches for ocean
    wave wavelength prediction.

    Random Forest, XGBoost, LSTM and GRU models were evaluated.
    The LSTM model achieved the lowest RMSE in our experiment
    and is therefore used for this application.
    """
)

with st.expander("⚠️ Scientific Note"):

    st.write(
        """
        The dataset does not directly contain measured wavelength.

        Wavelength was estimated using the deep-water wave
        relationship based on the dominant wave period.

        Therefore, the prediction represents an estimated
        wavelength rather than a directly measured wavelength.
        """
    )
