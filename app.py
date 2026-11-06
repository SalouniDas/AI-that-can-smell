import streamlit as st
import json
import pickle
import pandas as pd
from sensor import simulate_sensor_reading
from smell_model import SmellModel

st.set_page_config(page_title="AI That Can Smell", page_icon="👃")
st.title("👃 AI That Can Smell")
st.write("Pick a scent. The simulated electronic nose will sniff it, guess what it is, and describe it.")

# Load our data, the trained ML model, and the Smell Model
with open('samples.json', 'r') as f:
    samples = json.load(f)

try:
    with open('nose_model.pkl', 'rb') as f:
        nose_model = pickle.load(f)
except FileNotFoundError:
    st.error("Nose model not found. Please run 'python train_nose.py' in your terminal first.")
    st.stop()

smell_model = SmellModel()

# UI - Scent Selection
selected_smell = st.selectbox("Choose a sample to sniff:", list(samples.keys()))

if st.button("Sniff", type="primary"):
    # 1. Take a "sniff" (Simulate the sensor reading)
    molecules = samples[selected_smell]
    reading = simulate_sensor_reading(molecules, noise_level=0.05)

    st.subheader("1. Sensor Array Output")
    st.caption("How the 8 simulated gas sensors reacted to the chemical fragments.")
    # Format data for Streamlit bar chart
    chart_data = pd.DataFrame(reading, columns=["Sensor Activation"])
    st.bar_chart(chart_data)

    # 2. Predict the smell (Machine Learning Classification)
    prediction = nose_model.predict([reading])[0]
    probabilities = nose_model.predict_proba([reading])[0]
    confidence = max(probabilities) * 100

    st.subheader("2. AI Nose Prediction")
    st.markdown(f"The model is **{confidence:.1f}%** confident this is **{prediction}**.")

    # 3. Describe the smell (Chemical Fragment Mapping)
    st.subheader("3. Aroma Description")
    predicted_molecules = samples[prediction]
    descriptors = smell_model.describe(predicted_molecules)
    st.markdown(f"*{', '.join(descriptors)}*")