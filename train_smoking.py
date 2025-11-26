import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train():
    print("=== Memulai Pelatihan Model ===")

    # Pastikan dataset ada
    dataset_path = os.path.join(os.getcwd(), "train_dataset.csv")
    if not os.path.exists(dataset_path):
        print("ERROR: File 'train_dataset.csv' tidak ditemukan!")
        return

    # 1. Load dataset
    df = pd.read_csv(dataset_path)

    # 2. Preprocessing
    X = df.drop("smoking", axis=1)
    y = df["smoking"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Model
    print("Melatih model Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluasi
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"Akurasi: {acc:.4f}")
    print(classification_report(y_val, y_pred))

    # 5. Finalisasi model
    print("Melatih ulang dengan seluruh data...")
    model.fit(X, y)

    # Simpan model
    joblib.dump(model, "smoking_model.pkl")
    joblib.dump(X.columns.tolist(), "model_columns.pkl")

    print("=== Model berhasil disimpan ===")

if __name__ == "__main__":
    train()
