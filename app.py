import json
import joblib
import numpy as np
import openai
import pandas as pd
import pickle
import sklearn
import streamlit as st

st.set_page_config(
    page_title="Billboard Machine Learning App", page_icon="🎵", layout="wide"
)

st.title("🎵 Billboard ML Predictor & Analysis App")

# Cargar metadatos desde el archivo JSON de configuración del modelo
try:
    with open("meta_model_data.json", "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    st.sidebar.header("📌 Meta Data del Modelo")
    st.sidebar.markdown(f"**Modelo:** {meta_data.get('model_name')}")
    st.sidebar.markdown(f"**Versión:** {meta_data.get('version')}")
    st.sidebar.markdown(f"**Target:** `{meta_data.get('target')}`")
    st.sidebar.markdown("**Features:**")
    for feat in meta_data.get("features", []):
        st.sidebar.markdown(f"- {feat}")
except FileNotFoundError:
    st.sidebar.warning("No se encontró el archivo meta_model_data.json")

# Cargar el archivo .pkl (dataset o modelo entrenado del .ipynb)
st.subheader("📊 Contenido del Archivo Pickle (.pkl)")

try:
    with open("billboard_model_data.pkl", "rb") as f:
        model_or_data = pickle.load(f)

    st.success("¡Archivo .pkl cargado exitosamente!")

    # Si el archivo es un DataFrame (como el dataset procesado en el .ipynb)
    if isinstance(model_or_data, pd.DataFrame):
        st.write("### Vista previa del DataFrame")
        st.dataframe(model_or_data.head())

        st.info(
            f"El dataset contiene {model_or_data.shape[0]} filas y"
            f" {model_or_data.shape[1]} columnas."
        )
    else:
        # Si es un modelo de Scikit-Learn entrenado guardado con pickle o joblib
        st.write("### Objeto del Modelo Cargado")
        st.write(f"Tipo de objeto: {type(model_or_data)}")
        st.write(
            "El modelo está listo para realizar inferencias con las librerías"
            " configuradas."
        )

except FileNotFoundError:
    st.info("Coloca tu archivo 'billboard_model_data.pkl' en el repositorio.")
except Exception as e:
    st.error(f"Error al cargar el archivo .pkl: {e}")
