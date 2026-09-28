
import streamlit as st
import pandas as pd
import joblib


# Page Configuration

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="centered"
)



# Custom CSS

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 2rem;
    }

    /* Header */
    .header {
        text-align: center;
        padding: 10px 0 25px 0;
    }

    .header h1 {
        font-size: 2.4rem;
        margin-bottom: 5px;
    }

    .header p {
        color: #777;
        font-size: 1rem;
    }

    /* Input section */
    .input-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }

    /* Result boxes */
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #ddd;
    }

    .normal-result {
        background-color: #eaf7ee;
        border-color: #b7e4c7;
    }

    .fraud-result {
        background-color: #fdecec;
        border-color: #f5b5b5;
    }

    .result-title {
        font-size: 1.7rem;
        font-weight: bold;
        margin-bottom: 8px;
    }

    .probability {
        font-size: 1.2rem;
        font-weight: bold;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load Model


@st.cache_resource
def load_model():
    return joblib.load("src/fraud_model.pkl")


model = load_model()



# Header

st.markdown(
    """
    <div class="header">
        <h1>💳 Financial Fraud Detection</h1>
        <p>Machine Learning powered transaction risk analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)



# Transaction Input

st.markdown("### 🔍 Transaction Details")

st.markdown('<div class="input-card">', unsafe_allow_html=True)

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
)

amount = st.number_input(
    "Transaction Amount (₹)",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

old_balance = st.number_input(
    "Sender's Old Balance (₹)",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

st.markdown("</div>", unsafe_allow_html=True)


# Prediction

if st.button("🔍 Analyze Transaction", use_container_width=True):

    # Prepare input
    input_data = pd.DataFrame(
        [[transaction_type, amount, old_balance]],
        columns=["type", "amount", "oldbalanceOrg"]
    )

    # ML prediction
    prediction = model.predict(input_data)[0]

    # Fraud probability
    fraud_probability = model.predict_proba(input_data)[0][1]

    # Balance validation
    balance_issue = (
        transaction_type in
        ["PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT"]
        and amount > old_balance
    )


    # Result


    st.markdown("### 📊 Analysis Result")

    if prediction == 1:

        st.markdown(
            f"""
            <div class="result-box fraud-result">
                <div class="result-title">🚨 FRAUD TRANSACTION</div>
                <div class="probability">
                    ML Fraud Probability: {fraud_probability * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-box normal-result">
                <div class="result-title">✅ NORMAL TRANSACTION</div>
                <div class="probability">
                    ML Fraud Probability: {fraud_probability * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    
    # Probability Bar


    st.write("")

    st.progress(
        float(fraud_probability),
        text=f"Fraud Probability: {fraud_probability * 100:.2f}%"
    )


    # Rule-Based Validation

    if balance_issue:

        st.warning(
            "⚠️ **Rule-Based Warning:** "
            "The transaction amount exceeds the sender's recorded balance."
        )

    else:

        st.info(
            "ℹ️ No balance-related warning detected."
        )






# Footer


st.markdown(
    """
    <div class="footer">
        Financial Fraud Detection • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)

