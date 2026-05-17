# AI Bank Fraud Detection System

A modern, machine learning-powered web dashboard built with **Streamlit** to detect fraudulent banking transactions in real-time.

## Features
- **Real-Time Analysis**: Enter transaction details to instantly predict fraud probability.
- **Machine Learning**: Uses a Random Forest Classifier trained on banking transaction patterns.
- **Modern UI**: Clean, responsive dashboard designed for professional environments.

## Local Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/your-username/bank-fraud-clean.git
   cd bank-fraud-clean
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the Machine Learning Model**:
   Run the training script to generate `models/model.pkl`:
   ```bash
   python train_model.py
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   The application will open in your default browser at `http://localhost:8501`.

## Deployment to Render / Streamlit Community Cloud

This project is structured for immediate deployment.
1. Upload this repository to GitHub.
2. Link your GitHub repository to your preferred hosting provider (e.g., [Render](https://render.com) or [Streamlit Community Cloud](https://share.streamlit.io)).
3. **Important for Render**: Use the following start command in your Render web service configuration:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
   *(Ensure you also run `python train_model.py` as part of your build command if you don't commit the `models/model.pkl` file).*
