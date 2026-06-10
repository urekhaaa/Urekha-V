import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("engagement_model.pkl", "rb"))

st.set_page_config(page_title="Student Engagement AI", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🎓 Student Engagement Detection System</h1>",
    unsafe_allow_html=True
)

st.write("Predict student engagement using EEG + Eye Tracking features")

# Inputs in columns (better UI)
col1, col2 = st.columns(2)

with col1:
    alpha = st.slider("Alpha Power 🧠", 0.0, 100.0, 50.0)
    beta = st.slider("Beta Power 🧠", 0.0, 100.0, 50.0)
    theta = st.slider("Theta Power 🧠", 0.0, 100.0, 20.0)

with col2:
    gaze_x = st.slider("Gaze X 👁️", 0.0, 100.0, 50.0)
    gaze_y = st.slider("Gaze Y 👁️", 0.0, 100.0, 50.0)

# Predict button
if st.button("🔍 Predict Engagement"):
    sample = np.array([[alpha, beta, theta, gaze_x, gaze_y]])
    result = model.predict(sample)[0]

    # Color-coded output
    if result.lower() == "high":
        st.success(f"🔥 Engagement Level: {result}")
    elif result.lower() == "medium":
        st.warning(f"⚡ Engagement Level: {result}")
    else:
        st.error(f"😴 Engagement Level: {result}")

# Footer
st.markdown("---")
st.caption("AI-based system using EEG + Eye Tracking features")