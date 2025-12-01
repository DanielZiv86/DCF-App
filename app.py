# app.py
import streamlit as st

from app_theme import apply_global_styles, render_header
from views.bonos_hd import render_bonos_hd
from views.letras_boncaps import render_letras_boncaps
from views.bonos_cer import render_bonos_cer


# -------------------- Config básica de la app --------------------
st.set_page_config(
    page_title="DCF | Análisis de Bonos y Letras",
    layout="wide",
)

# -------------------- Estilos globales + Header --------------------
apply_global_styles()
render_header()

# -------------------- Sidebar --------------------
st.sidebar.title("Instrumentos")

tipo_instrumento = st.sidebar.radio(
    "Seleccioná el instrumento",
    options=[
        "Bonos soberanos HD",
        "Letras y Boncaps",
        "Bonos Ajustables CER",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Herramienta interna • DCF Inversiones")

# -------------------- Lógica principal --------------------
if tipo_instrumento == "Bonos soberanos HD":
    render_bonos_hd()

elif tipo_instrumento == "Letras y Boncaps":
    render_letras_boncaps()

elif tipo_instrumento == "Bonos Ajustables CER":
    render_bonos_cer()