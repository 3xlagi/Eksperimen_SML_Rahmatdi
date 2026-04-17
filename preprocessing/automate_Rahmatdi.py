import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Memulai proses preprocessing...")
    
    # 1. Drop customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # 2. Fix TotalCharges data type
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 3. Handle Missing Values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # 4. Encode Target
    if 'Churn' in df.columns:
        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn'])
        
    # 5. One-Hot Encoding untuk fitur kategorik
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Preprocessing selesai. Bentuk data:", df.shape)
    return df

def save_data(df, output_path):
    # Pastikan folder tujuan ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data bersih berhasil disimpan ke {output_path}")

if __name__ == "__main__":
    # Tentukan path dinamis agar bisa dijalankan oleh GitHub Actions
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data_raw', 'Telco-Customer-Churn.csv')
    CLEAN_DATA_PATH = os.path.join(BASE_DIR, 'data_clean', 'Data_Cleaned.csv')
    
    # Eksekusi pipeline
    try:
        raw_df = load_data(RAW_DATA_PATH)
        clean_df = preprocess_data(raw_df)
        save_data(clean_df, CLEAN_DATA_PATH)
        print("Pipeline Otomatisasi Preprocessing Sukses!")
    except Exception as e:
        print(f"Terjadi error: {e}")