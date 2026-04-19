# Predicción de Diabetes con Flask y XGBoost

Aplicación web interactiva desarrollada con Streamlit para estimar el riesgo de diabetes a partir de variables clínicas, utilizando un modelo de Machine Learning entrenado con XGBoost.

## Aplicación desplegada en Render


## Tecnologías utilizadas
- Python
- Streamlit
- XGBoost
- Scikit-learn
- NumPy
- Pandas
- Joblib
- Render

## Funcionalidades principales
- Ingreso manual de variables clínicas del paciente
- Validación de rangos básicos para cada variable
- Botón para rellenar datos aleatorios dentro de rangos razonables
- Predicción del riesgo de diabetes
- Visualización de la probabilidad estimada por el modelo
- Interfaz interactiva desarrollada en Streamlit

## Estructura del proyecto
- `src/app.py`: aplicación principal en Streamlit
- `models/diabetes_model.pkl`: modelo entrenado con XGBoost
- `src/explore.ipynb`: notebook con el proceso de exploración y modelado
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto

## Nota
Esta herramienta tiene fines educativos y no reemplaza una evaluación médica profesional.