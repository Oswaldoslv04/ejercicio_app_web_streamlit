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
# MENSAJES DE PRUEBA
# =========================
st.write("App cargó correctamente 🚀")

# =========================
# CARGA DEL MODELO
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "diabetes_model.pkl"

st.write("Ruta del modelo:", MODEL_PATH)
st.write("¿Existe el modelo?", MODEL_PATH.exists())

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    st.success("Modelo cargado correctamente ✅")
else:
    st.error("No se encontró el modelo. Revisa la ruta y el archivo diabetes_model.pkl")
    st.stop()

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
st.title("Predicción de Diabetes 🩺")
st.write(
    "Ingresa los datos clínicos del paciente para estimar el riesgo de diabetes usando un modelo XGBoost."
)

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
        st.error("⚠️ Riesgo de diabetes detectado")
    else:
        st.success("✅ No se detecta diabetes")

    st.info(f"Probabilidad estimada de diabetes: {probability:.2%}")

# =========================
# NOTA FINAL
# =========================
st.caption(
    "Este modelo no garantiza el 100% de predicción real. Esta herramienta tiene fines educativos y no reemplaza una evaluación médica profesional."
)