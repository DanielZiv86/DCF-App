# views/bonos_cer.py

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from bonos_cer_data import get_bonos_cer
from app_theme import DCF_PLOTLY_TEMPLATE


@st.cache_data(ttl=300, show_spinner=False)
def load_bonos_cer():
    df, meta = get_bonos_cer()
    return df, meta


def _format_bonos_cer_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla sólo con: Ticker, Precio, Vencimiento, TIR_%
    con formato amigable.
    """
    df_disp = df.copy()

    # Precio en $
    if "Precio" in df_disp.columns:
        df_disp["Precio"] = df_disp["Precio"].map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "-"
        )

    # Vencimiento dd/mm/yyyy
    if "Vencimiento" in df_disp.columns:
        df_disp["Vencimiento"] = pd.to_datetime(
            df_disp["Vencimiento"], errors="coerce"
        ).dt.strftime("%d/%m/%Y")

    # TIR_% como porcentaje con 2 decimales
    if "TIR_%" in df_disp.columns:
        df_disp["TIR_%"] = df_disp["TIR_%"].map(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "-"
        )

    return df_disp[["Ticker", "Precio", "Vencimiento", "TIR_%"]]


def build_cer_tir_curve(df: pd.DataFrame) -> go.Figure:
    """
    Scatter TIR vs Fecha de vencimiento, con ticker como label,
    + curva de tendencia (polinomio de 2º grado) en función de los años
    hasta el vencimiento.
    """
    d = df.copy()

    # Tipos correctos
    d["Vencimiento"] = pd.to_datetime(d["Vencimiento"], errors="coerce")
    d["TIR_%"] = pd.to_numeric(d["TIR_%"], errors="coerce")

    # Quitamos nulos
    d = d.dropna(subset=["Vencimiento", "TIR_%"])

    # --- FILTRO DE OUTLIERS EN TIR ---
    # Rango "razonable" para TIR real de CER (ajustalo si querés)
    d = d[(d["TIR_%"] > -10) & (d["TIR_%"] < 40)]

    if len(d) < 2:
        # Caso degenerate: sólo scatter
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["Vencimiento"],
            y=d["TIR_%"],
            mode="markers+text",
            text=d["Ticker"],
            textposition="top center",
            name="Bonos CER",
            marker=dict(size=9),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Venc.: %{x|%d/%m/%Y}<br>"
                "TIR: %{y:.2f}%<extra></extra>"
            ),
        ))
        fig.update_layout(
            title="Curva CER: TIR real vs vencimiento",
            xaxis_title="Fecha de vencimiento",
            yaxis_title="TIR real (%)",
            template=DCF_PLOTLY_TEMPLATE,
            height=500,
            margin=dict(l=10, r=10, t=50, b=20),
        )
        return fig

    # ---------------------------------------------------------
    # 1) Scatter base
    # ---------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=d["Vencimiento"],
        y=d["TIR_%"],
        mode="markers+text",
        text=d["Ticker"],
        textposition="top center",
        name="Bonos CER",
        marker=dict(size=9),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Venc.: %{x|%d/%m/%Y}<br>"
            "TIR: %{y:.2f}%<extra></extra>"
        ),
    ))

    # ---------------------------------------------------------
    # 2) Curva de tendencia en función de años a vencimiento
    # ---------------------------------------------------------
    hoy = pd.Timestamp.today().normalize()
    d["years_to_maturity"] = (d["Vencimiento"] - hoy).dt.days / 365.25

    x_num = d["years_to_maturity"].to_numpy()
    y_num = d["TIR_%"].to_numpy()

    # Si tenemos al menos 3 puntos, usamos un polinomio de 2º grado
    if len(d) >= 3:
        coef = np.polyfit(x_num, y_num, deg=2)
        a2, a1, a0 = coef

        # Grid suave entre duración mínima y máxima
        x_grid = np.linspace(x_num.min(), x_num.max(), 100)
        y_fit = a2 * x_grid**2 + a1 * x_grid + a0

        # Convertimos nuevamente a fechas para ploteo
        x_dates = hoy + pd.to_timedelta(x_grid * 365.25, unit="D")
    else:
        # Fallback: recta simple si hay pocos datos
        a1, a0 = np.polyfit(x_num, y_num, deg=1)
        x_grid = np.linspace(x_num.min(), x_num.max(), 100)
        y_fit = a0 + a1 * x_grid
        x_dates = hoy + pd.to_timedelta(x_grid * 365.25, unit="D")

    fig.add_trace(go.Scatter(
        x=x_dates,
        y=y_fit,
        mode="lines",
        name="Tendencia (duration)",
        line=dict(width=2, dash="dot"),
        hoverinfo="skip",
    ))

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    fig.update_layout(
        title="Curva CER: TIR real vs vencimiento",
        xaxis_title="Fecha de vencimiento",
        yaxis_title="TIR real (%)",
        template=DCF_PLOTLY_TEMPLATE,
        height=500,
        margin=dict(l=10, r=10, t=50, b=20),
    )

    return fig



def render_bonos_cer():
    st.header("Bonos Ajustables CER")

    df, _meta = load_bonos_cer()

    tabla_col, graf_col = st.columns([1.3, 2.0])

    # ---------- Tabla ----------
    with tabla_col:
        st.subheader("Listado de Bonos Ajustables por CER")

        df_disp = _format_bonos_cer_for_display(df)

        n_rows = len(df_disp)
        table_height = (n_rows + 1) * 35

        st.dataframe(
            df_disp.set_index("Ticker"),
            width="stretch",
            height=table_height,
        )

    # ---------- Gráfico ----------
    with graf_col:
        st.subheader("Curva CER: TIR real vs vencimiento")
        fig = build_cer_tir_curve(df)
        st.plotly_chart(fig, width="stretch")

