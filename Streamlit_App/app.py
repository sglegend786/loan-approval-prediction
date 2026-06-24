import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from reportlab.pdfgen import canvas
import io

# ---------------- PATH FIX (FINAL CORRECT) ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "Notebook", "loan_prediction_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "Notebook", "scaler.pkl")

# Page Configuration
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="centered"
)

# Sidebar
st.sidebar.title("🏦 Loan Prediction System")

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2910/2910768.png",
    width=120
)

st.sidebar.markdown("""
### About Project

This application predicts whether a loan application will be approved using a Machine Learning model.

### Model Information
- Algorithm: Random Forest Classifier
- Task: Binary Classification
- Output:
    - ✅ Loan Approved
    - ❌ Loan Not Approved
""")

st.sidebar.info("Enter applicant details and click Predict to get the result.")

# Custom CSS
st.markdown("""
<style>
body { background-color: #f5f7fb; }

.stApp { background-color: white; }

h1 {
    color: #1f4e79;
    text-align: center;
    font-size: 45px;
}

.stButton>button {
    width: 100%;
    height: 50px;
    background-color: #1f77b4;
    color: white;
    font-size: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL SAFELY ---------------- #
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    st.error("❌ Model loading failed!")
    st.code(str(e))
    st.stop()

# Title
st.markdown("""
<h1>🏦 Loan Approval Prediction System</h1>
<p style="text-align:center; font-size:20px;">
Predict whether your loan application will be approved or not.
</p>
""", unsafe_allow_html=True)

st.divider()

# Inputs
st.subheader("👤 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=5)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    annual_income = st.number_input("Annual Income")
    credit_score = st.number_input("Credit Score")
    loan_amount = st.number_input("Loan Amount")
    term = st.number_input("Loan Term")
    loan_status = st.selectbox("Loan Status", [0, 1])

# Encoding
gender = 1 if gender == "Male" else 0
marital_status = 1 if marital_status == "Married" else 0
education = 1 if education == "Graduate" else 0

property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Predict
if st.button("Predict Loan Approval"):

    input_data = np.array([[
        gender,
        marital_status,
        dependents,
        education,
        loan_status,
        annual_income,
        credit_score,
        loan_amount,
        term,
        property_area
    ]])

    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    approval_probability = probability[0][1] * 100
    result = "Approved" if prediction[0] == 1 else "Not Approved"

    st.subheader("📊 Approval Probability")
    st.metric("Loan Approval Probability", f"{approval_probability:.2f}%")

    summary = pd.DataFrame({
        "Applicant Income": [annual_income],
        "Loan Amount": [loan_amount],
        "Approval Probability (%)": [round(approval_probability, 2)],
        "Prediction": [result]
    })

    st.subheader("📋 Prediction Summary")
    st.table(summary)

    # PDF
    pdf = io.BytesIO()
    c = canvas.Canvas(pdf)

    c.drawString(100, 750, "Loan Prediction Report")
    c.drawString(100, 700, f"Income: {annual_income}")
    c.drawString(100, 670, f"Loan Amount: {loan_amount}")
    c.drawString(100, 640, f"Approval %: {approval_probability:.2f}")
    c.drawString(100, 610, f"Result: {result}")

    c.save()
    pdf.seek(0)

    st.download_button(
        "📄 Download PDF Report",
        pdf,
        "Loan_Prediction_Report.pdf",
        "application/pdf"
    )

    if prediction[0] == 1:
        st.balloons()
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")