import streamlit as st
import pickle
import numpy as np

# Page settings
st.set_page_config(
    page_title="Student Engagement Detection",
    page_icon="🎓",
    layout="wide"
)

# Load model
model = pickle.load(open("engagement_model.pkl", "rb"))

# Sidebar
st.sidebar.title("📌 Project Information")
st.sidebar.write("Student Engagement Detection System")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Features:")
st.sidebar.write("- Alpha Power")
st.sidebar.write("- Beta Power")
st.sidebar.write("- Theta Power")
st.sidebar.write("- Gaze X")
st.sidebar.write("- Gaze Y")

# Title
st.title("🎓 Student Engagement Detection System")

st.markdown("""
### About the Project

This project predicts student engagement levels using EEG signals and eye-tracking features.

The system analyzes:
- Alpha Power
- Beta Power
- Theta Power
- Eye Gaze Coordinates

and classifies engagement as:

✅ High  
⚡ Medium  
😴 Low
""")

st.markdown("---")

# Input section
col1, col2 = st.columns(2)

with col1:
    alpha = st.slider("Alpha Power", 0.0, 100.0, 50.0)
    beta = st.slider("Beta Power", 0.0, 100.0, 50.0)

with col2:
    theta = st.slider("Theta Power", 0.0, 100.0, 20.0)
    gaze_x = st.slider("Gaze X", 0.0, 100.0, 50.0)

gaze_y = st.slider("Gaze Y", 0.0, 100.0, 50.0)

# Predict button
if st.button("🔍 Predict Engagement"):

    sample = np.array([
        [alpha, beta, theta, gaze_x, gaze_y]
    ])

    result = model.predict(sample)[0]

    st.markdown("## Prediction Result")

    if str(result).lower() == "high":
        st.success(f"🔥 Engagement Level: {result}")

    elif str(result).lower() == "medium":
        st.warning(f"⚡ Engagement Level: {result}")

    else:
        st.error(f"😴 Engagement Level: {result}")

# Footer
st.markdown("---")
st.caption("Developed as an Internship Project using EEG and Eye Tracking Data")
