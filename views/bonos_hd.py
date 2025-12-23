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
    df = df[df["Duration"] > 0].copy()

    # --- Clasificación Ley (heurística por prefijo) ---
    # Globales (Ley NY): GDxx
    # Bonares (Ley AR):  ALxx / AExx
    def classify_law(ticker: str) -> str:
        t = str(ticker).strip().upper()
        if t.startswith("GD"):
            return "Globales (Ley NY)"
        if t.startswith("AL") or t.startswith("AE") or t.startswith("AN"):
            return "Bonares (Ley AR)"
        return "Otros"

    df["Ley"] = df["Ticker"].map(classify_law)

    fig = go.Figure()

    def add_group(group_df: pd.DataFrame, name: str):
        if group_df.empty:
            return

        x = group_df["Duration"].astype(float).values
        y = group_df["TIR"].astype(float).values
        tickers = group_df["Ticker"].astype(str).values

        # Puntos del grupo (mismo color por ser un solo trace)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=tickers,
            textposition="top center",
            name=name,
            marker=dict(size=9),
            hovertemplate="<b>%{text}</b><br>Duration: %{x:.2f}<br>TIR: %{y:.2%}<extra></extra>",
        ))

        # Tendencia log del grupo: y = a + b*ln(x)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if mask.sum() >= 2:
            X = np.log(x[mask])
            b, a = np.polyfit(X, y[mask], 1)

            xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
            ys = a + b * np.log(xs)

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(dash="dash"),
                name=f"Tendencia (log) – {name}",
                hovertemplate=f"Tendencia (log) – {name}<br>TIR: %{{y:.2%}}<extra></extra>",
            ))

    # 2 grupos principales
    add_group(df[df["Ley"] == "Globales (Ley NY)"], "Globales (Ley NY)")
    add_group(df[df["Ley"] == "Bonares (Ley AR)"], "Bonares (Ley AR)")

    # Opcional: si querés mostrar “Otros” (BOPREAL, etc.), descomentá:
    # add_group(df[df["Ley"] == "Otros"], "Otros")

    fig.update_layout(
        title=dict(text=""),
        xaxis_title="Duration (años)",
        yaxis_title="TIR",
        template=DCF_PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=20),
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
                width="stretch",
                height=table_height,
            )

    with graf_col:
        st.markdown("### Curva de TIR en moneda de emisión")
        if not df_market.empty:
            fig = build_curve_figure(df_market[["Ticker", "Duration", "TIR"]], mercado)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Sin datos para graficar la curva.")
