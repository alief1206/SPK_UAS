import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Perokok", page_icon="🫁")

# Fungsi load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "smoking_model.pkl")
    cols_path = os.path.join(os.getcwd(), "model_columns.pkl")

    if not os.path.exists(model_path):
        st.error("Model belum ditemukan! Jalankan train_model.py terlebih dahulu.")
        st.stop()

    model = joblib.load(model_path)
    cols = joblib.load(cols_path)
    return model, cols

model, model_columns = load_model()

# Tampilan
st.title("🫁 Sistem Prediksi Perokok")
st.write("Model menggunakan metode Random Forest untuk klasifikasi.")

# Form Input
st.sidebar.header("Input Data Pasien")

with st.form("form_input"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Umur", 10, 100, 30)
        height = st.number_input("Tinggi (cm)", 100, 250, 170)
    with col2:
        weight = st.number_input("Berat (kg)", 30, 200, 65)
        waist = st.number_input("Lingkar Pinggang", 50, 150, 80)
    with col3:
        dental_caries = st.selectbox("Karies Gigi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

    col4, col5 = st.columns(2)
    with col4:
        systolic = st.number_input("Sistolik", 80, 200, 120)
        relaxation = st.number_input("Diastolik", 50, 120, 80)
        cholesterol = st.number_input("Kolesterol", 100, 400, 200)
        triglyceride = st.number_input("Trigliserida", 30, 500, 150)
        hdl = st.number_input("HDL", 20, 100, 50)
        ldl = st.number_input("LDL", 50, 200, 100)
    with col5:
        fasting_sugar = st.number_input("Gula Darah Puasa", 50, 300, 100)
        hemoglobin = st.number_input("Hemoglobin", 10.0, 20.0, 15.0)
        urine_protein = st.selectbox("Protein Urine", [1, 2, 3, 4, 5, 6])
        creatinine = st.number_input("Serum Creatinine", 0.4, 2.0, 1.0)
        ast = st.number_input("AST", 10, 100, 25)
        alt = st.number_input("ALT", 10, 100, 25)
        gtp = st.number_input("GTP", 10, 200, 30)

    submitted = st.form_submit_button("🔍 Analisis Sekarang")

if submitted:
    # Data tidak diinput mata & telinga → default aman
    input_data = pd.DataFrame([{
        "age": age,
        "height(cm)": height,
        "weight(kg)": weight,
        "waist(cm)": waist,
        "eyesight(left)": 1.0,
        "eyesight(right)": 1.0,
        "hearing(left)": 1,
        "hearing(right)": 1,
        "systolic": systolic,
        "relaxation": relaxation,
        "fasting blood sugar": fasting_sugar,
        "Cholesterol": cholesterol,
        "triglyceride": triglyceride,
        "HDL": hdl,
        "LDL": ldl,
        "hemoglobin": hemoglobin,
        "Urine protein": urine_protein,
        "serum creatinine": creatinine,
        "AST": ast,
        "ALT": alt,
        "Gtp": gtp,
        "dental caries": dental_caries
    }])

    # Sesuaikan dengan kolom model
    input_data = input_data[model_columns]

    # Prediksi
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"🔥 HASIL: PEROKOK | Probabilitas: {prob:.1%}")
    else:
        st.success(f"💚 HASIL: BUKAN PEROKOK | Probabilitas: {(1-prob):.1%}")
