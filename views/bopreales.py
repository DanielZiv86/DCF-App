# views/bopreales.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from bopreales_data import get_multi_table
from app_theme import DCF_PLOTLY_TEMPLATE


@st.cache_data(ttl=300)
def load_all_bopreales() -> pd.DataFrame:
    return get_multi_table()


# =========================
# Formatters (tabla)
# =========================
def fmt_ars(x):
    if pd.isna(x):
        return "-"
    return f"$ {float(x):,.0f}".replace(",", ".")

def fmt_usd(x):
    if pd.isna(x):
        return "-"
    s = f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"USD {s}"


# =========================
# Chart
# =========================
def build_curve_figure(df: pd.DataFrame) -> go.Figure:
    df = df.dropna(subset=["Duration", "TIR"])
    df = df[df["Duration"] > 0].copy()

    fig = go.Figure()

    x = df["Duration"].astype(float).values
    y = df["TIR"].astype(float).values
    tickers = df["Ticker"].astype(str).values

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=tickers,
        textposition="top center",
        name="BOPREAL",
        marker=dict(size=9),
        hovertemplate="<b>%{text}</b><br>Duration: %{x:.2f}<br>TIR: %{y:.2%}<extra></extra>",
    ))

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if mask.sum() >= 2:
        X = np.log(x[mask])
        b, a = np.polyfit(X, y[mask], 1)
        xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
        ys = a + b * np.log(xs)

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(dash="dash"),
            name="Tendencia (log)",
            hovertemplate="Tendencia (log)<br>TIR: %{y:.2%}<extra></extra>",
        ))

    fig.update_layout(
        template=DCF_PLOTLY_TEMPLATE,
        title="",
        xaxis_title="Duration (años)",
        yaxis_title="TIR",
        legend_title=None,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_yaxes(tickformat=".2%")
    return fig


# =========================
# Render
# =========================
def render():
    st.subheader("Bopreales - TIR")

    df_all = load_all_bopreales()

    tab_ars, tab_usd = st.tabs(["ARS", "USD"])
    for tab, mercado in [(tab_ars, "ARS"), (tab_usd, "USD")]:
        with tab:
            df = df_all[df_all["Mercado"] == mercado].copy()

            # ✅ ordenar por Duration asc (NaN al final)
            df["Duration"] = pd.to_numeric(df.get("Duration"), errors="coerce")
            df = df.sort_values(["Duration", "Ticker"], ascending=[True, True], na_position="last")

            # ✅ más lugar al gráfico (como Letras)
            col_tbl, col_fig = st.columns([1.1, 2.0])

            with col_tbl:
                show = df[["Ticker", "Precio", "TIR"]].copy()

                # Precio como texto (para $ / USD)
                if mercado == "ARS":
                    show["Precio"] = show["Precio"].apply(fmt_ars)
                else:
                    show["Precio"] = show["Precio"].apply(fmt_usd)

                # ✅ En la tabla la mostramos como porcentaje (5.00 en vez de 0.05)
                show["TIR"] = pd.to_numeric(show["TIR"], errors="coerce") * 100.0

                st.dataframe(
                    show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Ticker": st.column_config.TextColumn(width="small"),
                        "Precio": st.column_config.TextColumn(width="small"),
                        "TIR": st.column_config.NumberColumn("TIR", format="%.2f%%", width="small"),
                    },
                )

            with col_fig:
                st.plotly_chart(build_curve_figure(df), use_container_width=True)
