# views/bonos_general.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from bonistas_hd_data import get_multi_table as get_soberanos_multi_table
from bopreales_data import get_multi_table as get_bopreales_multi_table
from app_theme import DCF_PLOTLY_TEMPLATE


# =========================
# Loaders
# =========================
@st.cache_data(ttl=300)
def load_all_soberanos() -> pd.DataFrame:
    return get_soberanos_multi_table()


@st.cache_data(ttl=300)
def load_all_bopreales() -> pd.DataFrame:
    return get_bopreales_multi_table()


# =========================
# Helpers
# =========================
def _classify_soberano(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if t.startswith("GD"):
        return "Globales (Ley NY)"
    if t.startswith(("AL", "AE", "AN")):
        return "Bonares (Ley AR)"
    return "Otros"


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _build_hover_extras(df: pd.DataFrame, *, is_soberanos: bool) -> np.ndarray:
    """
    Devuelve un array (len=df) con texto HTML adicional para el hover.
    - Si no existen columnas (Precio / Vencimiento / Ley), devuelve strings vacíos.
    """
    # columnas candidatas (robusto a naming distinto)
    precio_col = _first_existing_col(df, ["Precio", "price", "Price", "PX", "Px", "Precio Actual", "precio"])
    maturity_col = _first_existing_col(df, ["Maturity", "Vencimiento", "Fecha Vto", "Vto", "MaturityDate", "Venc."])
    # para soberanos podemos inferir ley por ticker, pero si existe una col, mejor
    ley_col = _first_existing_col(df, ["Ley", "Legislacion", "Legislación", "Law"])

    extras = []
    for _, row in df.iterrows():
        parts = []

        if precio_col and pd.notna(row.get(precio_col)):
            try:
                parts.append(f"Precio: {float(row.get(precio_col)):,.2f}")
            except Exception:
                parts.append(f"Precio: {row.get(precio_col)}")

        if maturity_col and pd.notna(row.get(maturity_col)):
            parts.append(f"Maturity: {row.get(maturity_col)}")

        if ley_col and pd.notna(row.get(ley_col)):
            parts.append(f"Ley: {row.get(ley_col)}")
        else:
            # fallback suave solo para soberanos si no hay columna Ley
            if is_soberanos:
                t = str(row.get("Ticker", "")).upper()
                if t.startswith("GD"):
                    parts.append("Ley: NY")
                elif t.startswith(("AL", "AE", "AN")):
                    parts.append("Ley: AR")

        extras.append("<br>".join(parts))

    return np.array(extras, dtype=object)


def _add_group_markers_and_trend(
    fig: go.Figure,
    df: pd.DataFrame,
    name: str,
    show_labels: bool,
    color: str | None = None,
    is_soberanos: bool = False,
):
    if df.empty:
        return

    df = df.copy()
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
    df["TIR"] = pd.to_numeric(df["TIR"], errors="coerce")
    df = df.dropna(subset=["Duration", "TIR"])
    df = df[df["Duration"] > 0]

    if df.empty:
        return

    x = df["Duration"].astype(float).values
    y = df["TIR"].astype(float).values
    tickers = df["Ticker"].astype(str).values

    marker_kwargs = dict(size=9)
    line_kwargs = dict(dash="dash")
    if color:
        marker_kwargs["color"] = color
        line_kwargs["color"] = color

    hover_extras = _build_hover_extras(df, is_soberanos=is_soberanos)

    # puntos (labels opcionales)
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers+text" if show_labels else "markers",
        text=tickers,  # hover siempre tiene ticker
        textposition="top center",
        textfont=dict(size=11),
        name=name,
        marker=marker_kwargs,
        cliponaxis=False,
        customdata=hover_extras,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Duration: %{x:.2f}<br>"
            "TIR: %{y:.2%}"
            "<br>%{customdata}"
            "<extra></extra>"
        ),
    ))

    # tendencia log: y = a + b ln(x)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if mask.sum() >= 2:
        X = np.log(x[mask])
        b, a = np.polyfit(X, y[mask], 1)
        xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 220)
        ys = a + b * np.log(xs)

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=line_kwargs,
            name=f"Tendencia (log) – {name}",
            hovertemplate=f"Tendencia (log) – {name}<br>TIR: %{{y:.2%}}<extra></extra>",
        ))


def build_general_curve_figure(
    df_soberanos: pd.DataFrame,
    df_bop: pd.DataFrame,
    show_labels: bool,
    height: int,
    show_globales: bool,
    show_bonares: bool,
    show_bopreal: bool,
) -> go.Figure:
    fig = go.Figure()

    # --- Soberanos: Globales y Bonares ---
    df_s = df_soberanos.copy()
    df_s["Grupo"] = df_s["Ticker"].map(_classify_soberano)

    if show_globales:
        _add_group_markers_and_trend(
            fig,
            df_s[df_s["Grupo"] == "Globales (Ley NY)"],
            "Globales (Ley NY)",
            show_labels=show_labels,
            color=None,  # color por template
            is_soberanos=True,
        )

    if show_bonares:
        _add_group_markers_and_trend(
            fig,
            df_s[df_s["Grupo"] == "Bonares (Ley AR)"],
            "Bonares (Ley AR)",
            show_labels=show_labels,
            color=None,  # color por template
            is_soberanos=True,
        )

    # --- BOPREAL (color distintivo) ---
    if show_bopreal:
        _add_group_markers_and_trend(
            fig,
            df_bop,
            "BOPREAL",
            show_labels=show_labels,
            color="orange",
            is_soberanos=False,
        )

    fig.update_layout(
        template=DCF_PLOTLY_TEMPLATE,
        title="",
        xaxis_title="Duration (años)",
        yaxis_title="TIR",
        legend_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=20, b=20),
        height=height,
    )
    fig.update_yaxes(tickformat=".2%")
    return fig


# =========================
# Render
# =========================
def render():
    st.markdown(
        "<h2 style='text-align: center;'>CURVA SOBERANOS Y BOPREAL</h2>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([1.4, 1.4, 1.2, 1.2])

    with col1:
        mercado_soberanos = st.radio(
            "Mercado Soberanos",
            ["PESOS", "MEP", "CCL"],
            index=1,
            horizontal=True,
        )

    with col2:
        default_bop = "ARS" if mercado_soberanos == "PESOS" else "USD"
        mercado_bop = st.radio(
            "Mercado BOPREAL",
            ["ARS", "USD"],
            index=0 if default_bop == "ARS" else 1,
            horizontal=True,
        )

    with col3:
        show_labels = st.toggle("Mostrar tickers", value=True)

    with col4:
        chart_height = st.slider("Altura gráfico", 520, 980, 720, 20)

    st.markdown("")

    v1, v2, v3 = st.columns([1, 1, 1])
    with v1:
        show_globales = st.toggle("Ver Globales", value=True)
    with v2:
        show_bonares = st.toggle("Ver Bonares", value=True)
    with v3:
        show_bopreal = st.toggle("Ver BOPREAL", value=True)

    df_s_all = load_all_soberanos()
    df_b_all = load_all_bopreales()

    df_s = df_s_all[df_s_all["Mercado"] == mercado_soberanos].copy()
    df_b = df_b_all[df_b_all["Mercado"] == mercado_bop].copy()

    if not (show_globales or show_bonares or show_bopreal):
        st.warning("Activá al menos una curva para graficar.")
        return

    if df_s.empty and df_b.empty:
        st.warning("No hay datos para graficar en la combinación seleccionada.")
        return

    fig = build_general_curve_figure(
        df_s,          # pasamos DF completo (para hover enriquecido)
        df_b,          # pasamos DF completo (para hover enriquecido)
        show_labels=show_labels,
        height=chart_height,
        show_globales=show_globales,
        show_bonares=show_bonares,
        show_bopreal=show_bopreal,
    )

    # ✅ Fix warning: use_container_width -> width
    st.plotly_chart(fig, width="stretch")
