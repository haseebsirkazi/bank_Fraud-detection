import streamlit as st
import pandas as pd
import pickle
import os

# Set page configuration for modern look
st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-container {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4CAF50;
    }
    .fraud-alert {
        background-color: rgba(255, 75, 75, 0.1);
        border: 1px solid #FF4B4B;
        padding: 20px;
        border-radius: 10px;
        color: #FF4B4B;
        text-align: center;
    }
    .legit-alert {
        background-color: rgba(76, 175, 80, 0.1);
        border: 1px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        color: #4CAF50;
        text-align: center;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data

# Main Title
st.markdown("<h1>🛡️ AI Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Monitor and detect suspicious banking transactions in real-time using machine learning.")
st.markdown("---")

# Sidebar for Input
st.sidebar.header("Transaction Details")
st.sidebar.markdown("Enter transaction specifics below to predict potential fraud.")

tx_type = st.sidebar.selectbox("Transaction Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=1500.0, step=100.0)
old_balance = st.sidebar.number_input("Origin Old Balance ($)", min_value=0.0, value=5000.0, step=100.0)
new_balance = st.sidebar.number_input("Origin New Balance ($)", min_value=0.0, value=3500.0, step=100.0)

analyze_btn = st.sidebar.button("Analyze Transaction", use_container_width=True, type="primary")

# Top Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="metric-container">
            <p>Total Scanned Today</p>
            <div class="metric-value" style="color: #00bcd4;">14,302</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="metric-container">
            <p>Fraud Detected</p>
            <div class="metric-value" style="color: #FF4B4B;">12</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class="metric-container">
            <p>System Accuracy</p>
            <div class="metric-value">99.8%</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("### Transaction Analysis Result")

# Load model
model_data = load_model()

if model_data is None:
    st.warning("⚠️ Model not found. Please run `python train_model.py` first to generate the machine learning model.")
else:
    model = model_data['model']
    le = model_data['label_encoder']

    if analyze_btn:
        with st.spinner("Analyzing transaction patterns..."):
            # Prepare Input
            try:
                encoded_type = le.transform([tx_type])[0]
                input_data = pd.DataFrame({
                    'type_encoded': [encoded_type],
                    'amount': [amount],
                    'oldbalanceOrg': [old_balance],
                    'newbalanceOrig': [new_balance]
                })

                # Predict
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]

                st.markdown("---")
                
                # Display Result
                if prediction == 1:
                    st.markdown(f"""
                        <div class="fraud-alert">
                            <h2>🚨 FRAUDULENT TRANSACTION DETECTED</h2>
                            <p>This transaction matches known fraudulent patterns.</p>
                            <p><strong>Risk Score:</strong> {probability*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="legit-alert">
                            <h2>✅ LEGITIMATE TRANSACTION</h2>
                            <p>This transaction appears normal.</p>
                            <p><strong>Risk Score:</strong> {probability*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.info("👈 Enter transaction details in the sidebar and click 'Analyze Transaction'.")

