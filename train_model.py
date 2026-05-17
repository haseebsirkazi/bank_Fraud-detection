import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def create_synthetic_data(num_samples=5000):
    np.random.seed(42)
    
    # Synthetic Features
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    transaction_types = np.random.choice(types, num_samples)
    
    amounts = np.abs(np.random.normal(150000, 100000, num_samples))
    oldbalanceOrg = np.abs(np.random.normal(50000, 200000, num_samples))
    newbalanceOrig = np.where(transaction_types == 'CASH_IN', oldbalanceOrg + amounts, oldbalanceOrg - amounts)
    newbalanceOrig = np.clip(newbalanceOrig, 0, None)
    
    # Introduce Some Fraud Logic
    is_fraud = np.zeros(num_samples)
    
    for i in range(num_samples):
        # High transfer amounts and account emptying is highly suspicious
        if transaction_types[i] in ['TRANSFER', 'CASH_OUT'] and amounts[i] > 200000 and oldbalanceOrg[i] < amounts[i]:
            if np.random.rand() > 0.1: # 90% chance to flag as fraud
                is_fraud[i] = 1
        # Random occasional fraud
        elif np.random.rand() > 0.98:
            is_fraud[i] = 1

    df = pd.DataFrame({
        'type': transaction_types,
        'amount': amounts,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'isFraud': is_fraud
    })
    
    return df

def train_and_save_model():
    print("Generating synthetic dataset...")
    df = create_synthetic_data(10000)
    
    print("Preprocessing data...")
    # Label Encode 'type'
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    
    # Features and Target
    X = df[['type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate Accuracy
    acc = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {acc * 100:.2f}%")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the model and label encoder
    print("Saving model to models/model.pkl...")
    with open('models/model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': le}, f)
        
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
