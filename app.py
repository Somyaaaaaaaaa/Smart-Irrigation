import streamlit as st
import numpy as np
import joblib
import datetime
import random

# ---- Page Config ----
st.set_page_config(
    page_title="Smart Irrigation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 Smart Irrigation System Dashboard")
st.markdown("Enter **scaled sensor values** between 0 and 1 to predict the **sprinkler status** for each parcel.")
st.markdown("---")

# ---- Load Model ----
@st.cache_resource
def load_model(path="Farm_Irrigation_System.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---- Sensor Names ----
sensor_names = [
    "Soil Moisture 1", "Soil Moisture 2", "Soil Temperature", "Air Temperature",
    "Humidity", "Light Intensity", "Wind Speed", "Rain Detection", "pH Level",
    "EC Value", "Nitrogen Level", "Phosphorus Level", "Potassium Level",
    "Crop Type Index", "Leaf Wetness", "Canopy Temp", "Solar Radiation",
    "Water Pressure", "Valve Feedback"
]

# ---- Initialize Random Fill Values ----
if "sensor_inputs" not in st.session_state:
    st.session_state.sensor_inputs = {f"sensor_{i}": 0.5 for i in range(len(sensor_names))}

# ---- Random Fill Button ----
if st.button("🎲 Auto-Fill with Random Values"):
    for i in range(len(sensor_names)):
        st.session_state.sensor_inputs[f"sensor_{i}"] = round(random.uniform(0, 1), 2)

# ---- Input Collection ----
st.subheader("Sensor Readings")
sensor_values = []

with st.expander("Fill Sensor Values (Scaled: 0 to 1)", expanded=True):
    input_cols = st.columns(3)
    for i, name in enumerate(sensor_names):
        with input_cols[i % 3]:
            value = st.slider(
                label=name,
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.sensor_inputs[f"sensor_{i}"],
                step=0.01,
                key=f"slider_{i}"
            )
            # Save latest value in session state
            st.session_state.sensor_inputs[f"sensor_{i}"] = value
            sensor_values.append(value)

# ---- Prediction ----
st.markdown("---")
if st.button("Predict Sprinkler Status") and model:
    input_array = np.array(sensor_values).reshape(1, -1)
    try:
        prediction = model.predict(input_array)[0]
        st.success("Prediction Complete")
        st.markdown("### 💧 Sprinkler Status (Parcel-wise):")

        grid_cols = st.columns(6)
        for i, status in enumerate(prediction):
            icon = "🟢 ON" if status == 1 else "🔴 OFF"
            with grid_cols[i % 6]:
                st.metric(label=f"Parcel {i}", value=icon)

        st.caption(f"Last predicted at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---- Footer ----
st.markdown("---")
st.caption("Built with Streamlit • Internship Project • AICTE • Built by: Somya Das")


