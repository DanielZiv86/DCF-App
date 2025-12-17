# app_theme.py
import os
import base64
import streamlit as st
import plotly.graph_objects as go

# -------------------- Branding DCF --------------------
DCF_COLORS = {
    "bg_header":  "#053D57",
    "bg_alt":     "#111111",
    "text_header":"#FFFFFF",
    "text_body":  "#E3E6E9",
    "border":     "#3A4A52",
    "accent":     "#0E6881",
    "muted":      "#90B0BC",
}

# Título único de la app
APP_TITLE = "DCF Inversiones | Herramientas para análisis de mercado"


# -------------------- Utils --------------------
def load_logo_base64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -------------------- Estilos globales --------------------
def apply_global_styles():
    st.markdown(
        """
        <style>
            /* Altura aproximada de la barra superior de Streamlit (Deploy) */
            :root {
                --st-topbar-h: 64px;
            }

            /* Empuja todo el contenido debajo de la topbar */
            .block-container {
                padding-top: calc(0.8rem + var(--st-topbar-h)) !important;
            }

            /* Header DCF */
            .dcf-header {
                position: relative;
                z-index: 50;

                margin-top: 0px;
                margin-bottom: 18px;

                display: flex;
                align-items: center;
                gap: 12px;
            }

            .dcf-header-title {
                background: linear-gradient(90deg, #053D57, #0E6881);
                color: white;
                font-size: 1.4rem;
                font-weight: 600;
                padding: 10px 20px;
                border-radius: 10px;
                line-height: 1.2;
                white-space: nowrap;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


# -------------------- Header --------------------
def render_header():
    logo_base64 = load_logo_base64("dcf_logo.png")

    st.markdown(
        f"""
        <div class="dcf-header">
            {'<img src="data:image/png;base64,' + logo_base64 + '" style="height:44px;">' if logo_base64 else ''}
            <div class="dcf-header-title">{APP_TITLE}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------- Plotly Template --------------------
DCF_PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(
            color=DCF_COLORS["text_body"],
            family="Montserrat, Arial, sans-serif"
        ),
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        title_font=dict(color=DCF_COLORS["accent"], size=20),
        xaxis=dict(
            showgrid=True,
            gridcolor="#333333",
            color=DCF_COLORS["text_body"]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#333333",
            color=DCF_COLORS["text_body"]
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=DCF_COLORS["text_body"])
        ),
        colorway=[
            DCF_COLORS["accent"],
            "#4EA5B5",
            "#89C0CC",
            "#C5D2D8",
            "#FFFFFF",
        ],
    )
)
