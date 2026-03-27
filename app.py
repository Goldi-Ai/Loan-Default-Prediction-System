import streamlit as st
import pickle
import pandas as pd

# ================= LOAD MODEL =================
model = pickle.load(open("loan_model.pkl", "rb"))
features = pickle.load(open("model_features.pkl", "rb"))

st.set_page_config(page_title="CGU Bank Loan System", layout="wide")

# ================= SIDEBAR =================

st.sidebar.image("logo.png", width=150)

st.sidebar.title("CGU Bank")

st.sidebar.markdown("""
### About CGU Bank

CGU Bank is a demonstration banking system developed for the **Fundamentals of Data Science (FDS) Case Study Report**.

📍 Address  
C.V. Raman Global University  
Bidyanagar, Mahura  
Bhubaneswar, Odisha – 752054  

### Banking Services
✔ Personal Loans  
✔ Education Loans  
✔ Home Loans  
✔ Business Loans  

### Technology Used
- Python  
- Data Science  
- Machine Learning  
- Streamlit  
""")

# ================= HEADER =================

col1, col2 = st.columns([1,4])

with col1:
    st.image("logo.png", width=120)

with col2:
    st.title("CGU Bank Loan Approval System")
    st.write("Smart Loan Risk Prediction using Machine Learning")

st.divider()

# ================= LOAN CRITERIA =================

st.subheader("Loan Eligibility Criteria")

st.write("""
Applicants should meet the following criteria:

• Age between **18 – 65**  
• Stable employment  
• Good **credit score**  
• Sufficient **annual income**  
• Loan amount within repayment capacity
""")

st.divider()

# ================= APPLICATION FORM =================

st.header("Loan Application Form")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Full Name")
    age = st.number_input("Age", 18, 65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "Other"])

with col2:
    employment = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed"])
    annual_income = st.number_input("Annual Income")
    credit_score = st.number_input("Credit Score", 300, 900)
    loan_amount = st.number_input("Loan Amount")
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Car", "Business", "Personal"])

st.divider()

# ================= PREDICTION =================

if st.button("Predict Loan Approval"):

    input_data = {
        "age": age,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "loan_amount": loan_amount
    }

    input_df = pd.DataFrame([input_data])

    for col in features:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[features]

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1] * 100

    st.subheader("Loan Prediction Result")

    st.metric("Approval Probability", f"{probability:.2f}%")

    if prediction[0] == 1:

        st.success("✅ Loan Approved")

        st.info(f"""
        Congratulations **{name}**!

        Based on your financial profile, our AI system predicts that you are **likely to repay the loan successfully**.

        Your application has **low financial risk** and meets CGU Bank's lending criteria.
        """)

    else:

        st.error("❌ Loan Rejected")

        st.warning(f"""
        Dear **{name}**,

        Our system detected **high financial risk** based on the provided details.

        Suggestions to improve eligibility:
        • Improve credit score  
        • Reduce existing debt  
        • Increase annual income
        """)

    st.divider()

    st.caption("CGU Bank AI Loan Risk Analysis System | Developed for BCA 6th Semester ML Case Study")
