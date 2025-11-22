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

def load_logo_base64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Template Plotly DARK corporativo
DCF_PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(color=DCF_COLORS["text_body"], family="Montserrat, Arial, sans-serif"),
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        title_font=dict(color=DCF_COLORS["accent"], size=20),
        xaxis=dict(
            showgrid=True,
            gridcolor="#333333",
            color=DCF_COLORS["text_body"],
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#333333",
            color=DCF_COLORS["text_body"],
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


def apply_global_styles():
    """Inyecta el CSS global de la app."""
    st.markdown(
        f"""
        <style>
            /* Fuente general y color de texto
               👉 OJO: sin el * para no pisar los iconos de Streamlit */
            html, body, [data-testid="stAppViewContainer"] {{
                font-family: "Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                color: {DCF_COLORS["text_body"]};
            }}

            /* Header DCF */
            .dcf-header {{
                background-color: rgba(5,61,87,0.25);
                padding: 12px 24px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                gap: 18px;
                margin-bottom: 20px;
                border: 1px solid {DCF_COLORS["border"]};
            }}
            .dcf-header h1 {{
                color: {DCF_COLORS["text_header"]};
                font-size: 26px;
                margin: 0;
            }}

            /* Sidebar con gradiente DCF */
            [data-testid="stSidebar"] > div:first-child {{
                background: linear-gradient(180deg, #053D57 0%, #021F2D 100%) !important;
            }}
            [data-testid="stSidebar"] * {{
                color: #FFFFFF !important;
            }}

            /* Tablas (dataframe) */
            thead tr th {{
                background-color: {DCF_COLORS["bg_header"]} !important;
                color: {DCF_COLORS["text_header"]} !important;
            }}
            tbody tr:nth-child(even) {{
                background-color: #0D0D0D !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Header con logo y título principal."""
    logo_base64 = load_logo_base64("dcf_logo.png")

    st.markdown(
        f"""
        <div class="dcf-header">
            {'<img src="data:image/png;base64,' + logo_base64 + '" style="height:55px;">' if logo_base64 else ''}
            <h1>DCF Inversiones · Análisis de Bonos y Letras</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
