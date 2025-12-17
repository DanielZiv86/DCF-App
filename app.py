#app.py

import streamlit as st
import market_cache

from app_theme import apply_global_styles, render_header
from views.bonos_hd import render_bonos_hd
from views.bopreales import render as render_bopreales
from views.letras_boncaps import render_letras_boncaps
from views.bonos_cer import render_bonos_cer
from views.ons_ytm import render as render_ons_ytm


# -------------------- Config básica de la app --------------------
st.set_page_config(
    page_title="DCF | Herramientas de Análisis de Mercado",
    layout="wide",
)

# -------------------- Estilos globales + Header --------------------
apply_global_styles()
render_header()

# -------------------- Sidebar --------------------
st.sidebar.title("Instrumentos")
st.sidebar.metric("Última actualización", market_cache.get_last_update_display())

# 1) Selector principal (Bonos tiene misma jerarquía)
main_view = st.sidebar.radio(
    "Seleccioná el instrumento",
    options=[
        "Bonos",
        "Letras y Boncaps",
        "Bonos Ajustables CER",
        "ONs",
    ],
    index=0,  # default: Bonos
    key="main_view",
)

# 2) Sub-vistas SOLO cuando main_view == "Bonos"
bonos_subview = st.session_state.get("bonos_subview", "Bonos Soberanos")

if main_view == "Bonos":
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='margin-left:12px'>", unsafe_allow_html=True)

    bonos_subview = st.sidebar.radio(
        "",
        options=[
            "Bonos Soberanos",
            "Bopreales",
        ],
        index=0,  # default: Bonos Soberanos
        key="bonos_subview",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Herramienta interna • DCF Inversiones")


# -------------------- Lógica principal --------------------
if main_view == "Bonos":
    if bonos_subview == "Bonos Soberanos":
        render_bonos_hd()
    elif bonos_subview == "Bopreales":
        render_bopreales()

elif main_view == "Letras y Boncaps":
    render_letras_boncaps()

elif main_view == "Bonos Ajustables CER":
    render_bonos_cer()

elif main_view == "ONs":
    render_ons_ytm()
