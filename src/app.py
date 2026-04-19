import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import random

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Predicción de Diabetes",
    page_icon="🩺",
    layout="centered"
)

# =========================
# ESTILOS PERSONALIZADOS
# =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.title-box {
    text-align: center;
    padding: 1rem 0 1.5rem 0;
}
.title-box h1 {
    color: white;
    margin-bottom: 0.4rem;
}
.title-box p {
    color: #cbd5e1;
    font-size: 1rem;
}
.result-success {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.35);
    padding: 1rem;
    border-radius: 14px;
    color: #dcfce7;
    text-align: center;
    font-size: 1.1rem;
    margin-top: 1rem;
}
.result-danger {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 1rem;
    border-radius: 14px;
    color: #fee2e2;
    text-align: center;
    font-size: 1.1rem;
    margin-top: 1rem;
}
.prob-box {
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.30);
    padding: 0.9rem;
    border-radius: 12px;
    color: #dbeafe;
    text-align: center;
    margin-top: 0.8rem;
}
.note-box {
    background: rgba(255,255,255,0.05);
    padding: 0.9rem;
    border-radius: 12px;
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-top: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CARGA DEL MODELO
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "diabetes_model.pkl"

model = joblib.load(MODEL_PATH)

# =========================
# FUNCIÓN PARA DATOS ALEATORIOS
# =========================
def generar_datos_aleatorios():
    return {
        "Pregnancies": random.randint(0, 20),
        "Glucose": random.randint(40, 250),
        "BloodPressure": random.randint(40, 140),
        "SkinThickness": random.randint(7, 99),
        "Insulin": random.randint(15, 900),
        "BMI": round(random.uniform(15, 60), 1),
        "DiabetesPedigreeFunction": round(random.uniform(0.05, 2.50), 2),
        "Age": random.randint(18, 90),
    }

# =========================
# ESTADO INICIAL
# =========================
if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "Pregnancies": 1,
        "Glucose": 100,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 25.0,
        "DiabetesPedigreeFunction": 0.50,
        "Age": 30,
    }

# =========================
# ENCABEZADO
# =========================
st.markdown("""
<div class="title-box">
    <h1>Predicción de Diabetes 🩺</h1>
    <p>Ingresa los datos clínicos del paciente para estimar el riesgo de diabetes usando un modelo XGBoost.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# BOTÓN DE DATOS ALEATORIOS
# =========================
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    if st.button("Rellenar aleatoriamente 🎲", use_container_width=True):
        st.session_state.form_data = generar_datos_aleatorios()
        st.rerun()

# =========================
# FORMULARIO
# =========================
st.subheader("Datos del paciente")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=int(st.session_state.form_data["Pregnancies"]),
        step=1
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=40,
        max_value=140,
        value=int(st.session_state.form_data["BloodPressure"]),
        step=1
    )

    insulin = st.number_input(
        "Insulin",
        min_value=15,
        max_value=900,
        value=int(st.session_state.form_data["Insulin"]),
        step=1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.05,
        max_value=2.50,
        value=float(st.session_state.form_data["DiabetesPedigreeFunction"]),
        step=0.01,
        format="%.2f"
    )

with col2:
    glucose = st.number_input(
        "Glucose",
        min_value=40,
        max_value=250,
        value=int(st.session_state.form_data["Glucose"]),
        step=1
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=7,
        max_value=99,
        value=int(st.session_state.form_data["SkinThickness"]),
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=15.0,
        max_value=60.0,
        value=float(st.session_state.form_data["BMI"]),
        step=0.1,
        format="%.1f"
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=90,
        value=int(st.session_state.form_data["Age"]),
        step=1
    )

# Guardar valores actuales
st.session_state.form_data = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age,
}

# =========================
# BOTÓN DE PREDICCIÓN
# =========================
if st.button("Predecir", use_container_width=True):
    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.subheader("Resultado del análisis")

    if prediction == 1:
        st.markdown(
            '<div class="result-danger">⚠️ Riesgo de diabetes detectado</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-success">✅ No se detecta diabetes</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="prob-box">Probabilidad estimada de diabetes: <strong>{probability:.2%}</strong></div>',
        unsafe_allow_html=True
    )

# =========================
# NOTA FINAL
# =========================
st.markdown("""
<div class="note-box">
Este modelo no garantiza el 100% de predicción real. Esta herramienta tiene fines educativos y no reemplaza una evaluación médica profesional.
</div>
""", unsafe_allow_html=True)