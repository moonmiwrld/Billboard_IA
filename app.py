import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

st.set_page_config(page_title="Billboard Data Analysis", page_icon="🎵", layout="wide")

st.title("🎵 Billboard Data Analysis & Model Dashboard")

# Cargar metadatos JSON
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

# Cargar y desplegar archivo .pkl
st.subheader("📊 Análisis del Dataset (Pickle .pkl)")

try:
    with open("billboard_model_data.pkl", "rb") as f:
        df = pickle.load(f)

    st.success("¡Archivo .pkl cargado correctamente!")

    if isinstance(df, pd.DataFrame):
        st.write("### Vista previa de los datos")
        st.dataframe(df.head())

        # Visualización usando matplotlib y seaborn (librerías del ipynb)
        st.write("### Visualización de Datos (Seaborn & Matplotlib)")
        fig, ax = plt.subplots(figsize=(8, 4))
        if 'rank' in df.columns:
            sns.histplot(df['rank'], kde=True, ax=ax, color='purple')
            ax.set_title("Distribución de Rankings")
        else:
            sns.countplot(x=df.iloc[:, 0], ax=ax)
            ax.set_title("Frecuencia de datos")
        
        st.pyplot(fig)
    else:
        st.write(df)

except FileNotFoundError:
    st.info("Coloca tu archivo 'billboard_model_data.pkl' en el mismo directorio.")
except Exception as e:
    st.error(f"Error al cargar el archivo .pkl: {e}")
