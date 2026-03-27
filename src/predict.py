import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import joblib

class ReadmissionPredictor(nn.Module):
    def __init__(self, input_dim):
        super(ReadmissionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def preprocess_data_for_prediction(df, scaler, train_columns):
    df_processed = df.copy()
    
    # Fill missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    # Handle categoricals
    categorical_cols = ['gender', 'discharge_day_of_week', 'insurance_type']
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # Ensure columns match training
    for col in train_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
            
    df_processed = df_processed[train_columns]
    
    X = df_processed.astype(np.float32)
    X_scaled = scaler.transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Predict patient readmission")
    parser.add_argument("--input", type=str, required=True, help="Path to input test.csv")
    args = parser.parse_args()
    
    try:
        # Determine model path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'models')
        
        # Load artifacts
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        train_columns = joblib.load(os.path.join(model_dir, 'train_columns.pkl'))
        
        # Load data
        test_df = pd.read_csv(args.input)
        
        # Preprocess
        X_tensor = preprocess_data_for_prediction(test_df, scaler, train_columns)
        
        # Load model
        input_dim = X_tensor.shape[1]
        model = ReadmissionPredictor(input_dim)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
        model.eval()
        
        # Predict
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = outputs.squeeze().numpy()
            preds = (probs >= 0.5).astype(int)
        
        # Output predictions
        predictions = pd.DataFrame({
            'patient_id': test_df['patient_id'],
            'readmission_probability': probs,
            'prediction': preds
        })
        
        predictions.to_csv("predictions.csv", index=False)
        print("Predictions saved to predictions.csv")
    except Exception as e:
        print(f"Error predicting: {e}")

if __name__ == "__main__":
    main()