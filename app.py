import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("encoder.pkl")

st.set_page_config(page_title="Student Dropout Prediction")

st.title("🎓 Student Dropout Prediction System")

attendance = st.slider("Attendance (%)", 0, 100, 70)
gpa = st.slider("GPA", 0.0, 10.0, 6.5)
assignments = st.slider("Assignments (%)", 0, 100, 60)
fee_delay = st.selectbox("Fee Delay", ["No", "Yes"])

if st.button("Predict"):
    fee_delay_encoded = le.transform([fee_delay])[0]

    input_data = np.array([[attendance, gpa, assignments, fee_delay_encoded]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Dropout (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Student is Safe (Probability: {probability:.2f})")