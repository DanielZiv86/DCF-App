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
# -------------------- Ajustes UI (sidebar) --------------------
st.markdown(
    """
    <style>
      /* Sidebar: achicar KPI 'Última actualización' */
      section[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 0.85rem; }
      section[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.35rem; }
      section[data-testid="stSidebar"] [data-testid="stMetricDelta"] { font-size: 0.75rem; }

      /* Badge pill */
      .dcf-badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 600;
        line-height: 1.2;
        margin-top: 0.25rem;
      }
      .dcf-open  { background: rgba(22, 163, 74, 0.18); border: 1px solid rgba(22, 163, 74, 0.55); }
      .dcf-closed{ background: rgba(239, 68, 68, 0.16); border: 1px solid rgba(239, 68, 68, 0.55); }
    </style>
    """,
    unsafe_allow_html=True,
)
render_header()

# -------------------- Sidebar --------------------
st.sidebar.title("Instrumentos")
st.sidebar.metric("Última actualización", market_cache.get_last_update_display())

# Badge Mercado abierto/cerrado (según horario definido en market_cache)
is_open = market_cache.is_market_open()
badge_text = "Mercado abierto" if is_open else "Mercado cerrado"
badge_class = "dcf-open" if is_open else "dcf-closed"
st.sidebar.markdown(
    f'<span class="dcf-badge {badge_class}">{badge_text}</span>',
    unsafe_allow_html=True,
)


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
        "Subtipo de bonos",
        options=[
            "Bonos Soberanos",
            "Bopreales",
        ],
        index=0,
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