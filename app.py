# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly import figure_factory as ff
import plotly.graph_objects as go
import base64
import os

# Datos de Bonos HD
from bonistas_hd_data import get_multi_table, get_hd_table

# Datos Letras / Boncaps
from letras_boncaps_data import get_letras_carry

# -------------------- Config básica de la app --------------------
st.set_page_config(
    page_title="DCF | Análisis de Bonos y Letras",
    layout="wide",
)

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

# -------------------- Estilos globales (CSS) --------------------
st.markdown(
    f"""
    <style>
        /* Fuente general y color de texto */
        html, body, [data-testid="stAppViewContainer"] * {{
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

        /* 🔥 Ocultar header nativo de Streamlit (donde aparece keyboard_double_arrow_right) */
        header[data-testid="stHeader"] {{
            display: none !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <style>

    /* --- FIX ABSOLUTO PARA REMOVER EL TEXTO "keyboard_double_arrow_right" --- */

    /* 1) Forzar que Streamlit use siempre Material Icons */
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    .material-icons, [class^="material-icons"], span[class*="material-icons"] {
        font-family: 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 24px !important;
        line-height: 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        display: inline-block !important;
        white-space: nowrap !important;
        direction: ltr !important;
    }

    /* 2) Prevención adicional: ocultar cualquier fallback textual */
    span[data-baseweb="icon"] {
        font-family: 'Material Icons' !important;
    }

    /* 3) Forzar reemplazo específico si Streamlit renderiza texto en vez del icono */
    span:contains("keyboard_double_arrow_right") {
        font-family: 'Material Icons' !important;
        visibility: hidden !important; /* opción 1 */
    }
    </style>
    """, unsafe_allow_html=True)



# Header con logo (el fondo negro general se mantiene)
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


# -------------------- Helpers de datos --------------------

@st.cache_data(ttl=300)
def load_all_bonds() -> pd.DataFrame:
    """Carga la tabla multi-mercado de Bonos HD."""
    return get_multi_table()


@st.cache_data(ttl=300)
def load_letras_v2():
    """Carga tabla y datos crudos de Letras/Boncaps (nueva versión)."""
    return get_letras_carry()


# -------------------- Gráficos Bonos HD --------------------

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
        hovertemplate=(
            "<b>%{text}</b><br>"      # Ticker
            "TIR: %{y:.2%}<extra></extra>"
        ),
    ))

    # Línea de tendencia log: hover = Tendencia (log) + TIR
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
                "Tendencia (log)<br>"
                "TIR: %{y:.2%}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"CURVA TIR SOBERANOS HD – {mercado}",
        xaxis_title="Duration (años)",
        yaxis_title="TIR (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5
        ),
        template=DCF_PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(tickformat=".2%")
    return fig


# -------------------- Gráficos Letras / Boncaps --------------------

def _reset_with_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reset index y garantiza una columna 'Ticker' a partir de:
    - 'symbol'
    - 'ticker'
    - 'index'
    - o, en última instancia, el propio índice.
    """
    df = df.copy().reset_index()

    # Buscar una columna que pueda ser el ticker
    candidates = [c for c in df.columns if c.lower() in ["ticker", "symbol", "index"]]
    if candidates:
        df = df.rename(columns={candidates[0]: "Ticker"})
    else:
        df["Ticker"] = df.index.astype(str)

    return df


def build_letras_rate_curve(carry: pd.DataFrame, col: str, label: str) -> go.Figure:
    """
    Gráfico TNA/TEM vs días al vencimiento.
    col: 'tna' o 'tem'
    """
    df = _reset_with_ticker(carry)
    df = df.dropna(subset=["days_to_exp", col])
    df = df[df["days_to_exp"] > 0].sort_values("days_to_exp")

    x = df["days_to_exp"].values.astype(float)
    y = df[col].values.astype(float)
    tickers = df["Ticker"].values

    fig = go.Figure()

    # Puntos con hover limpio: Ticker + tasa
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
            xanchor="center", x=0.5
        ),
        template=DCF_PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(tickformat=".2%")
    return fig

def build_letras_bands_figure_plotly(carry: pd.DataFrame) -> go.Figure:
    """
    Gráfico de bandas de Carry-Trade en Plotly, usando el template dark DCF.
    Usa:
      - days_to_exp
      - finish_worst  (techo / banda superior)
      - finish_better (piso / banda inferior)
      - MEP_BREAKEVEN
      - índice = symbol (ticker)
    """
    df = carry.copy()

    fig = go.Figure()

    # Banda Superior (techo)
    fig.add_trace(go.Scatter(
        x=df["days_to_exp"],
        y=df["finish_worst"],
        mode="lines",
        name="$ Banda Sup",
        line=dict(width=2),
    ))

    # Banda Inferior (piso)
    fig.add_trace(go.Scatter(
        x=df["days_to_exp"],
        y=df["finish_better"],
        mode="lines",
        name="$ Banda Inf",
        line=dict(width=2, dash="dash"),
    ))

    # MEP Breakeven (puntos + ticker)
    fig.add_trace(go.Scatter(
        x=df["days_to_exp"],
        y=df["MEP_BREAKEVEN"],
        mode="markers+text",
        text=df.index,               # el índice es el ticker (symbol)
        textposition="top center",
        name="MEP Breakeven",
        marker=dict(size=8),
    ))

    fig.update_layout(
        title="Carry-Trade – Líneas de Banda",
        xaxis_title="Días al vencimiento",
        yaxis_title="Precio proyectado ($)",
        template=DCF_PLOTLY_TEMPLATE,   # 👈 mismo estilo dark que TNA/TEM
        height=550,
        margin=dict(l=10, r=10, t=50, b=20),
    )

    return fig


def build_letras_scenarios_heatmap(carry: pd.DataFrame) -> go.Figure:
    """
    Heatmap de escenarios de carry, calculado directamente a partir de:
    - MEP_BREAKEVEN
    - finish_worst (techo de banda)
    - days_to_exp
    - Ticker / symbol
    """
    # Partimos del DF original
    df = carry.copy().reset_index()

    # Renombrar symbol -> Ticker si existe
    if "symbol" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"symbol": "Ticker"})

    # Si aún no hay Ticker, lo generamos desde el índice
    if "Ticker" not in df.columns:
        if "index" in df.columns:
            df["Ticker"] = df["index"].astype(str)
        else:
            df["Ticker"] = df.index.astype(str)

    # Columnas base que necesitamos
    required_base = ["Ticker", "days_to_exp", "MEP_BREAKEVEN", "finish_worst"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"Faltan columnas base en carry_raw: {missing_base}")

    # Nos quedamos solo con lo necesario y ordenamos
    df = (
        df[required_base]
        .dropna(subset=["days_to_exp", "MEP_BREAKEVEN", "finish_worst"])
        .sort_values("days_to_exp")
    )

    # Aseguramos tipo numérico
    df["days_to_exp"] = df["days_to_exp"].astype(float)
    df["MEP_BREAKEVEN"] = pd.to_numeric(df["MEP_BREAKEVEN"], errors="coerce")
    df["finish_worst"] = pd.to_numeric(df["finish_worst"], errors="coerce")

    # Escenarios de MEP
    scenario_prices = [1000, 1100, 1200, 1300, 1400]

    # Calculamos carry en cada escenario con la identidad:
    # carry_escenario = (MEP_BREAKEVEN / precio_escenario) - 1
    for price in scenario_prices:
        col_name = f"carry_{price}"
        df[col_name] = (df["MEP_BREAKEVEN"] / price) - 1

    # Peor caso: usamos el techo de la banda (finish_worst)
    df["carry_worst"] = (df["MEP_BREAKEVEN"] / df["finish_worst"]) - 1

    # Orden final de columnas de escenarios
    carry_cols = [
        "carry_1000",
        "carry_1100",
        "carry_1200",
        "carry_1300",
        "carry_1400",
        "carry_worst",
    ]

    # Matriz Z en porcentaje
    z = df[carry_cols].astype(float).values * 100.0  # ej: 45.3%

    # Texto anotado: dos decimales + símbolo %
    text = np.vectorize(lambda v: f"{v:.2f}%")(z)

    # Etiquetas X e Y
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

    # Rango simétrico para colores
    vmax = np.nanmax(z)
    vmin = np.nanmin(z)
    lim = max(abs(vmin), abs(vmax))

    fig = ff.create_annotated_heatmap(
        z=z,
        annotation_text=text,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0, "#8b0000"],   # rojo oscuro
            [0.5, "#ffffcc"],   # amarillo claro
            [1.0, "#006400"],   # verde oscuro
        ],
        zmin=-lim,
        zmax=lim,
        showscale=True,
        hoverinfo="z",
        font_colors=["black", "white"],  # base, luego lo ajustamos dinámico
    )

    # Hover más claro
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

    # ----------- Ajuste dinámico del color de texto según valor -----------
    # Texto blanco cuando |carry| > 40%, negro en el resto
    n_cols = len(carry_cols)
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            ann = fig.layout.annotations[i * n_cols + j]
            if val > 40 or val < -40:
                ann.font.color = "white"
            else:
                ann.font.color = "black"

    return fig


# -------------------- Sidebar --------------------

st.sidebar.title("Instrumentos")

tipo_instrumento = st.sidebar.radio(
    "Seleccioná el instrumento",
    options=[
        "Bonos soberanos HD",
        "Letras y Boncaps"
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Herramienta interna • DCF Inversiones")


# -------------------- Lógica principal --------------------

if tipo_instrumento == "Bonos soberanos HD":

    # ---------- Controles superiores ----------
    st.subheader("Curvas de TIR por tipo de mercado (HD)")

    col1, _ = st.columns([2, 3])
    with col1:
        mercado = st.radio(
            "Mercado",
            ["PESOS", "MEP", "CCL"],
            index=1,
            horizontal=True,
        )

    # ---------- Datos ----------
    df_all = load_all_bonds()
    df_market = df_all[df_all["Mercado"] == mercado].copy()

    # ---------- Tabla + Gráfico ----------
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


# =====================================================================
#                  SECCIÓN: LETRAS Y BONCAPS
# =====================================================================

elif tipo_instrumento == "Letras y Boncaps":

    st.subheader("Letras y Bonos Cortos – Carry MEP")

    # Cargar datos
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
            use_container_width=True,
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
            st.plotly_chart(fig_tna, use_container_width=True)

        with tab_tem:
            fig_tem = build_letras_rate_curve(carry_raw, "tem", "TEM")
            st.plotly_chart(fig_tem, use_container_width=True)

        with tab_carry:
            fig_carry = build_letras_bands_figure_plotly(carry_raw)
            st.plotly_chart(fig_carry, use_container_width=True)

        with tab_esc:
            st.markdown("#### Escenarios de Carry-Trade por tipo de cambio MEP")
            fig_heat = build_letras_scenarios_heatmap(carry_raw)
            st.plotly_chart(fig_heat, use_container_width=True)


