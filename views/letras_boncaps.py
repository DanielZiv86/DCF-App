# views/letras_boncaps.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly import figure_factory as ff

from letras_boncaps_data import get_letras_carry
from app_theme import DCF_PLOTLY_TEMPLATE


#@st.cache_data(ttl=300)
def load_letras_v2():
    """Carga tabla y datos crudos de Letras/Boncaps (nueva versión)."""
    return get_letras_carry()


def _reset_with_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reset index y garantiza una columna 'Ticker' a partir de:
    - 'symbol'
    - 'ticker'
    - 'index'
    - o, en última instancia, el propio índice.
    """
    df = df.copy().reset_index()

    candidates = [c for c in df.columns if c.lower() in ["ticker", "symbol", "index"]]
    if candidates:
        df = df.rename(columns={candidates[0]: "Ticker"})
    else:
        df["Ticker"] = df.index.astype(str)

    return df


def build_letras_rate_curve(carry: pd.DataFrame, col: str, label: str) -> go.Figure:
    """Gráfico TNA/TEM vs días al vencimiento. col: 'tna' o 'tem'."""
    df = _reset_with_ticker(carry)
    df = df.dropna(subset=["days_to_exp", col])
    df = df[df["days_to_exp"] > 0].sort_values("days_to_exp")

    x = df["days_to_exp"].values.astype(float)
    y = df[col].values.astype(float)
    tickers = df["Ticker"].values

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=tickers,
        textposition="top center",
        name=label,
        marker=dict(size=9),
        hovertemplate=(
            "<b>%{text}</b><br>" +
            f"{label}: " +
            "%{y:.2%}<extra></extra>"
        ),
    ))

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
            hovertemplate=(
                "Tendencia (log)<br>" +
                f"{label}: " +
                "%{y:.2%}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"{label} por plazo – Letras y Boncaps",
        xaxis_title="Días al vencimiento",
        yaxis_title=f"{label} (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
        ),
        template=DCF_PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(tickformat=".2%")
    return fig


def build_letras_bands_figure_plotly(carry: pd.DataFrame) -> go.Figure:
    """
    Gráfico de bandas de Carry-Trade en Plotly.
    Usa:
      - days_to_exp
      - finish_worst  (techo / banda superior)
      - finish_better (piso / banda inferior)
      - MEP_BREAKEVEN
    """
    df = carry.copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["days_to_exp"],
        y=df["finish_worst"],
        mode="lines",
        name="$ Banda Sup",
        line=dict(width=2),
    ))

    fig.add_trace(go.Scatter(
        x=df["days_to_exp"],
        y=df["finish_better"],
        mode="lines",
        name="$ Banda Inf",
        line=dict(width=2, dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=df["days_to_exp"],
        y=df["MEP_BREAKEVEN"],
        mode="markers+text",
        text=df.index,
        textposition="top center",
        name="MEP Breakeven",
        marker=dict(size=8),
    ))

    fig.update_layout(
        title="Carry-Trade – Líneas de Banda",
        xaxis_title="Días al vencimiento",
        yaxis_title="Precio proyectado ($)",
        template=DCF_PLOTLY_TEMPLATE,
        height=550,
        margin=dict(l=10, r=10, t=50, b=20),
    )

    return fig


def build_letras_scenarios_heatmap(carry: pd.DataFrame) -> go.Figure:
    """
    Heatmap de escenarios de carry, a partir de:
    - MEP_BREAKEVEN
    - finish_worst
    - days_to_exp
    - Ticker / symbol
    """
    df = carry.copy().reset_index()

    if "symbol" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"symbol": "Ticker"})

    if "Ticker" not in df.columns:
        if "index" in df.columns:
            df["Ticker"] = df["index"].astype(str)
        else:
            df["Ticker"] = df.index.astype(str)

    required_base = ["Ticker", "days_to_exp", "MEP_BREAKEVEN", "finish_worst"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"Faltan columnas base en carry_raw: {missing_base}")

    df = (
        df[required_base]
        .dropna(subset=["days_to_exp", "MEP_BREAKEVEN", "finish_worst"])
        .sort_values("days_to_exp")
    )

    df["days_to_exp"] = df["days_to_exp"].astype(float)
    df["MEP_BREAKEVEN"] = pd.to_numeric(df["MEP_BREAKEVEN"], errors="coerce")
    df["finish_worst"] = pd.to_numeric(df["finish_worst"], errors="coerce")

    scenario_prices = [1000, 1100, 1200, 1300, 1400]

    for price in scenario_prices:
        col_name = f"carry_{price}"
        df[col_name] = (df["MEP_BREAKEVEN"] / price) - 1

    df["carry_worst"] = (df["MEP_BREAKEVEN"] / df["finish_worst"]) - 1

    carry_cols = [
        "carry_1000",
        "carry_1100",
        "carry_1200",
        "carry_1300",
        "carry_1400",
        "carry_worst",
    ]

    z = df[carry_cols].astype(float).values * 100.0
    text = np.vectorize(lambda v: f"{v:.2f}%")(z)

    x_labels = [
        "MEP 1000",
        "MEP 1100",
        "MEP 1200",
        "MEP 1300",
        "MEP 1400",
        "Peor caso",
    ]
    y_labels = [
        f"{t} ({int(d)}d)"
        for t, d in zip(df["Ticker"], df["days_to_exp"])
    ]

    vmax = np.nanmax(z)
    vmin = np.nanmin(z)
    lim = max(abs(vmin), abs(vmax))

    fig = ff.create_annotated_heatmap(
        z=z,
        annotation_text=text,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0, "#8b0000"],
            [0.5, "#ffffcc"],
            [1.0, "#006400"],
        ],
        zmin=-lim,
        zmax=lim,
        showscale=True,
        hoverinfo="z",
        font_colors=["black", "white"],
    )

    fig.update_traces(
        hovertemplate=(
            "Letra: %{y}<br>"
            "Escenario: %{x}<br>"
            "Carry: %{z:.2f}%<extra></extra>"
        )
    )

    fig.data[0].colorbar.tickformat = ".0f"
    fig.data[0].colorbar.ticksuffix = "%"

    fig.update_layout(
        title="Escenarios de Carry-Trade por tipo de cambio MEP",
        xaxis_title="Escenario MEP futuro",
        yaxis_title="Letra (ticker – días al vencimiento)",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        template=DCF_PLOTLY_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    n_cols = len(carry_cols)
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            ann = fig.layout.annotations[i * n_cols + j]
            if val > 40 or val < -40:
                ann.font.color = "white"
            else:
                ann.font.color = "black"

    return fig


def render_letras_boncaps():
    """Dibuja toda la sección de 'Letras y Boncaps'."""
    st.subheader("Letras y Bonos Cortos – Carry MEP")

    tabla_letras, carry_raw, mep = load_letras_v2()

    tabla_col, graf_col = st.columns([1.2, 2.0])

    # ---------- Tabla ----------
    with tabla_col:
        st.markdown("### Letras y Boncaps activos")
        st.caption(f"💵 Tipo de cambio MEP utilizado: ${mep:,.2f}")

        df_show = tabla_letras.copy()

        if "Precio" in df_show:
            df_show["Precio"] = df_show["Precio"].map(lambda x: f"${x:,.2f}")
        if "Dias A Venc." in df_show:
            df_show["Dias A Venc."] = df_show["Dias A Venc."].astype(int)

        for col in ["TNA", "TEA", "TEM"]:
            if col in df_show:
                df_show[col] = df_show[col].map(lambda x: f"{x*100:.1f}%")

        for col in ["MEP BE", "$ Banda Sup"]:
            if col in df_show:
                df_show[col] = df_show[col].map(lambda x: f"${x:,.0f}")

        n_rows = len(df_show)
        table_height = (n_rows + 1) * 35

        st.dataframe(
            df_show.set_index("Ticker"),
            width="stretch",
            height=table_height,
        )
    # ---------- Gráficos ----------
    with graf_col:
        st.markdown("### Análisis gráfico")

        tab_tna, tab_tem, tab_carry, tab_esc = st.tabs(
            ["TNA", "TEM", "Carry-Trade", "Escenarios Carry-Trade"]
        )

        with tab_tna:
            fig_tna = build_letras_rate_curve(carry_raw, "tna", "TNA")
            st.plotly_chart(fig_tna, width="stretch")

        with tab_tem:
            fig_tem = build_letras_rate_curve(carry_raw, "tem", "TEM")
            st.plotly_chart(fig_tem, width="stretch")

        with tab_carry:
            fig_carry = build_letras_bands_figure_plotly(carry_raw)
            st.plotly_chart(fig_carry, width="stretch")

        with tab_esc:
            st.markdown("#### Escenarios de Carry-Trade por tipo de cambio MEP")
            fig_heat = build_letras_scenarios_heatmap(carry_raw)
            st.plotly_chart(fig_heat, width="stretch")
