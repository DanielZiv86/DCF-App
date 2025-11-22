# views/bonos_hd.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from bonistas_hd_data import get_multi_table
from app_theme import DCF_PLOTLY_TEMPLATE


@st.cache_data(ttl=300)
def load_all_bonds() -> pd.DataFrame:
    """Carga la tabla multi-mercado de Bonos HD."""
    return get_multi_table()


def build_curve_figure(df: pd.DataFrame, mercado: str) -> go.Figure:
    df = df.dropna(subset=["Duration", "TIR"])
    df = df[df["Duration"] > 0]

    x = df["Duration"].values.astype(float)
    y = df["TIR"].values.astype(float)
    tickers = df["Ticker"].values

    fig = go.Figure()

    # Puntos: hover = Ticker + TIR
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=tickers,
        textposition="top center",
        name="Bonos",
        marker=dict(size=9),
        hovertemplate="<b>%{text}</b><br>TIR: %{y:.2%}<extra></extra>",
    ))

    # Línea de tendencia log
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if mask.sum() >= 2:
        X = np.log(x[mask])
        b, a = np.polyfit(X, y[mask], 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = a + b * np.log(xs)

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(dash="dash"),
            name="Tendencia (log)",
            hovertemplate="Tendencia (log)<br>TIR: %{y:.2%}<extra></extra>",
        ))

    fig.update_layout(
        title=f"CURVA TIR SOBERANOS HD – {mercado}",
        xaxis_title="Duration (años)",
        yaxis_title="TIR (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
        ),
        template=DCF_PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(tickformat=".2%")
    return fig


def render_bonos_hd():
    """Dibuja toda la sección de 'Bonos soberanos HD'."""
    st.subheader("Curvas de TIR por tipo de mercado (HD)")

    col1, _ = st.columns([2, 3])
    with col1:
        mercado = st.radio(
            "Mercado",
            ["PESOS", "MEP", "CCL"],
            index=1,
            horizontal=True,
        )

    df_all = load_all_bonds()
    df_market = df_all[df_all["Mercado"] == mercado].copy()

    tabla_col, graf_col = st.columns([1.2, 2.0])

    with tabla_col:
        st.markdown(f"### Bonos soberanos HD – {mercado}")

        if df_market.empty:
            st.warning("No se encontraron datos para este mercado.")
        else:
            df_market = df_market.sort_values("Duration")

            df_show = df_market[["Ticker", "Precio", "Duration", "TIR"]].copy()
            df_show["Precio"] = df_show["Precio"].map(lambda x: f"${x:,.2f}")
            df_show["Duration"] = df_show["Duration"].map(lambda x: f"{x:.1f}")
            df_show["TIR"] = df_show["TIR"].map(lambda x: f"{x*100:.2f}%")

            n_rows = len(df_show)
            table_height = (n_rows + 1) * 35

            st.dataframe(
                df_show.set_index("Ticker"),
                use_container_width=True,
                height=table_height,
            )

    with graf_col:
        st.markdown("### Curva de TIR en moneda de emisión")
        if not df_market.empty:
            fig = build_curve_figure(df_market[["Ticker", "Duration", "TIR"]], mercado)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sin datos para graficar la curva.")
