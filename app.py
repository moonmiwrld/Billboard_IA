import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Predicción Billboard Top 1",
    page_icon="🎵",
    layout="centered"
)

# Encabezado principal
st.title("🎵 Billboard Top 1 Predictor")
st.markdown("### **Creadora: Luna Dubon**")
st.write("¡Bienvenido! Esta aplicación utiliza el último árbol de decisión entrenado para predecir si una canción tiene el potencial de alcanzar el puesto #1 en la lista Billboard.")

# Ruta del último archivo único consolidado exportado en la raíz
MODEL_PATH = Path("modelo_billboard.pkl")

@st.cache_resource
def cargar_recursos():
    if not MODEL_PATH.exists():
        return None, None
    # Cargar el diccionario unificado usando pickle
    with open(MODEL_PATH, "rb") as f:
        recursos = pickle.load(f)
    modelo = recursos["modelo"]
    columnas = recursos["columnas"]
    return modelo, columnas

modelo, columnas_entrenamiento = cargar_recursos()

if modelo is None or columnas_entrenamiento is None:
    st.error("❌ No se encontró el archivo consolidado del modelo 'modelo_billboard.pkl' en la raíz.")
else:
    st.sidebar.header("⚙️ Atributos de la Canción")
    duration_seg = st.sidebar.number_input("Duración (segundos)", min_value=1, max_value=5000, value=200, step=10)
    artist_type = st.sidebar.selectbox("Tipo de Artista", ["Male", "Female", "Mixed"])
    tempo = st.sidebar.selectbox("Tempo de la canción", ["Medium Tempo", "Slow Tempo", "Fast Tempo"])
    genre = st.sidebar.selectbox("Género musical", [
        "Urban", "Pop", "Traditional", "Alternative & Punk",
        "Electronica", "Rock", "Other", "Soundtrack", "Jazz"
    ])
    mood = st.sidebar.selectbox("Estado de ánimo (Mood)", [
        "Brooding", "Energizing", "Excited", "Yearning", "Romantic",
        "Peaceful", "Sophisticated", "Sentimental", "Defiant", "Easygoing",
        "Empowering", "Fiery", "Gritty", "Lively", "Melancholy", "Other",
        "Rowdy", "Sensual", "Upbeat", "Urgent"
    ])

    datos_usuario = pd.DataFrame([{
        "artist_type": artist_type,
        "tempo": tempo,
        "genre": genre,
        "mood": mood,
        "durationSeg": duration_seg
    }])

    datos_procesados = pd.get_dummies(datos_usuario)
    datos_procesados = datos_procesados.reindex(columns=columnas_entrenamiento, fill_value=0)

    st.subheader("🎧 Datos ingresados:")
    st.dataframe(datos_usuario)

    if st.button("🔮 Predecir éxito", use_container_width=True):
        prediccion = modelo.predict(datos_procesados)[0]
        st.markdown("--- ")
        if prediccion == 1:
            st.balloons()
            st.success("🎉 **¡SÍ PUEDE LLEGAR AL TOP 1!** El modelo estima que esta canción tiene las características de un hit masivo.")
        else:
            st.warning("⚠️ **No se predice que llegue al Top 1.**")
