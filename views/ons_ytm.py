# views/ons_ytm.py
import os
import html as html_lib
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

from ons_ytm_data import load_bd_ons_xlsx, compute_ytm_table


# =========================
# Helpers de formato
# =========================
def fmt_ars(x):
    if pd.isna(x):
        return "-"
    return f"$ {x:,.0f}".replace(",", ".")


def fmt_usd(x):
    if pd.isna(x):
        return "-"
    return f"USD {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x):
    if pd.isna(x):
        return "-"
    return f"{x:.2f}%"


def fmt_num(x, dec=2):
    if pd.isna(x):
        return "-"
    s = f"{float(x):,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def days_30_360(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    d1d = min(d1.day, 30)
    d2d = min(d2.day, 30)
    return (d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (d2d - d1d)


# =========================
# HTML Table builder
# =========================
def build_html_table(
    df: pd.DataFrame,
    tooltip_text: str = "ON próxima a su vencimiento",
    row_bg_close: str = "rgba(255, 193, 7, 0.18)",
    row_border_close: str = "rgba(255, 193, 7, 0.55)",
) -> str:

    if df is None or df.empty:
        return "<div style='opacity:0.75'>Sin datos para mostrar.</div>"

    tooltip_esc = html_lib.escape(tooltip_text)

    html = f"""
    <style>
      body {{ margin: 0; padding: 0; background: transparent; }}
      .dcf-wrap {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }}

      .dcf-table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 14px;
      }}

      .dcf-table th, .dcf-table td {{
          padding: 10px 12px;
          border-bottom: 1px solid rgba(255,255,255,0.08);
          text-align: left;
          vertical-align: middle;
          color: rgba(255,255,255,0.92);
      }}

      /* Header gris */
      .dcf-table th {{
          color: #E5E7EB;
          font-weight: 600;
          position: sticky;
          top: 0;
          background: #1F2933;
          border-bottom: 1px solid #374151;
      }}

      /* Row alerta */
      .dcf-row-close {{
          background: {row_bg_close};
          outline: 1px solid {row_border_close};
          outline-offset: -1px;
      }}

      /* Tooltip */
      .dcf-tooltip {{
          position: relative;
          cursor: help;
      }}

      .dcf-tooltip .dcf-tooltiptext {{
          visibility: hidden;
          width: 220px;
          background-color: rgba(20, 20, 20, 0.95);
          color: #fff;
          text-align: center;
          border-radius: 8px;
          padding: 8px 10px;
          position: absolute;
          z-index: 10;
          left: 50%;
          transform: translateX(-50%);
          bottom: 125%;
          opacity: 0;
          transition: opacity 0.15s ease;
          border: 1px solid rgba(255,255,255,0.15);
          font-size: 12px;
          white-space: normal;
      }}

      .dcf-tooltip:hover .dcf-tooltiptext {{
          visibility: visible;
          opacity: 1;
      }}
    </style>

    <div class="dcf-wrap">
      <table class="dcf-table">
        <thead>
          <tr>
            <th style="width:22%">Ticker</th>
            <th style="width:18%">Precio ARS</th>
            <th style="width:20%">Precio USD</th>
            <th style="width:20%">Cupón</th>
            <th style="width:20%">TIR</th>
          </tr>
        </thead>
        <tbody>
    """

    for _, row in df.iterrows():
        is_close = bool(row.get("_is_close", False))
        cls = "dcf-row-close" if is_close else ""
        ticker = html_lib.escape(str(row.get("Ticker", "")))
        precio_ars = html_lib.escape(str(row.get("Precio ARS", "")))
        precio_usd = html_lib.escape(str(row.get("Precio USD", "")))
        cupon = html_lib.escape(str(row.get("Cupón", "")))
        tir = html_lib.escape(str(row.get("TIR", "")))

        if is_close:
            ticker_cell = f"""
            <span class="dcf-tooltip">{ticker}
                <span class="dcf-tooltiptext">{tooltip_esc}</span>
            </span>
            """
        else:
            ticker_cell = ticker

        html += f"""
        <tr class="{cls}">
        <td>{ticker_cell}</td>
        <td>{precio_ars}</td>
        <td>{precio_usd}</td>
        <td>{cupon}</td>
        <td>{tir}</td>
        </tr>
        """

    html += """
        </tbody>
      </table>
    </div>
    """
    return html


# =========================
# Detalle ticker
# =========================
def _infer_payment_frequency(future_dates: pd.Series) -> str:
    """
    Determina periodicidad en base a diferencias 30/360 entre fechas FUTURAS.
    Robusto: con 2 fechas ya infiere.
    """
    ds = pd.to_datetime(future_dates).sort_values().dropna().unique()
    if len(ds) < 2:
        return "—"

    diffs = []
    for i in range(1, len(ds)):
        d1 = pd.Timestamp(ds[i - 1])
        d2 = pd.Timestamp(ds[i])
        diffs.append(days_30_360(d1, d2))

    if not diffs:
        return "—"

    d = float(np.median(diffs))

    if 25 <= d <= 35:
        return "Mensual"
    if 80 <= d <= 100:
        return "Trimestral"
    if 170 <= d <= 190:
        return "Semestral"
    if 350 <= d <= 370:
        return "Anual"
    return "Irregular"


def render_ticker_detail(
    tkr: str,
    df_bd: pd.DataFrame,
    price_usd_mkt: float | None,
    price_ars: float | None,
    mep_venta: float | None,
    price_usd_mep_equiv: float | None,  # = price_dirty_usd_used
):
    sub = (
        df_bd[df_bd["ticker"] == tkr]
        .dropna(subset=["date"])
        .sort_values("date")
        .copy()
    )

    if sub.empty:
        st.info("No hay cashflows para este ticker.")
        return

    sub["date"] = pd.to_datetime(sub["date"])
    today = pd.Timestamp.today().normalize()

    # --- Legislación / lámina mínima (tomamos primer valor no nulo) ---
    def _first_non_null(col: str):
        if col not in sub.columns:
            return "-"
        s = sub[col].dropna()
        if s.empty:
            return "-"
        v = s.iloc[0]
        return "-" if pd.isna(v) else str(v)

    legislacion = _first_non_null("legislacion")
    lamina_min = _first_non_null("lamina_minima")

    # --- Numéricos ---
    sub["Principal"] = pd.to_numeric(sub.get("Principal"), errors="coerce").fillna(0.0)
    sub["Int."] = pd.to_numeric(sub.get("Int."), errors="coerce").fillna(0.0)

    # --- Solo flujos futuros ---
    future = sub[sub["date"] >= today].copy()
    if future.empty:
        st.info("No hay cashflows futuros.")
        return

    past = sub[sub["date"] < today].copy()

    # Fechas clave
    maturity_dt = future["date"].max()
    next_dt = future["date"].min()

    # Próximo pago: interés y amortización (capital) del próximo pago
    next_interest = float(future.loc[future["date"] == next_dt, "Int."].sum())
    next_amort = float(future.loc[future["date"] == next_dt, "Principal"].sum())

    # Pendientes totales
    capital_pendiente = float(future["Principal"].sum())
    interes_pendiente = float(future["Int."].sum())

    # Periodicidad
    periodicidad = _infer_payment_frequency(future["date"])

    # --- Cupón corrido (30/360) usando el próximo cupón como base ---
    cupon_corrido = 0.0
    if not past.empty:
        last_dt = past["date"].max()
        days_period = days_30_360(last_dt, next_dt)
        days_elapsed = days_30_360(last_dt, today)
        if days_period > 0 and days_elapsed > 0:
            cupon_corrido = next_interest * (days_elapsed / days_period)

    # Valor teórico = Capital pendiente + cupón corrido (dirty)
    valor_teorico = capital_pendiente + cupon_corrido

    # --- Paridades ---
    def _premium_pct(price: float | None, vt: float) -> float | None:
        if price is None or pd.isna(price) or vt <= 0:
            return None
        return (float(price) / vt - 1.0) * 100.0

    prem_d = _premium_pct(price_usd_mkt, valor_teorico)
    prem_mep = _premium_pct(price_usd_mep_equiv, valor_teorico)

    # Helper para "metric" con color rojo si +, verde si -
    # (Streamlit lo hace con delta_color="inverse": positivo=rojo, negativo=verde)
    def metric_premium(label: str, pct: float | None):
        if pct is None:
            st.metric(label, "—")
        else:
            st.metric(
                label,
                value=" ",
                delta=f"{pct:+.2f}%",
                delta_color="inverse",
                help="Premium (+) / Discount (-) vs Valor Teórico"
            )

    # =========================
    # KPIs – Línea 1
    # Maturity / Periodicidad / Legislación / Lámina mínima
    # =========================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Maturity", maturity_dt.date().isoformat())
    c2.metric("Periodicidad", periodicidad)
    c3.metric("Legislación", legislacion)
    c4.metric("Lámina mínima", lamina_min)

    # =========================
    # KPIs – Línea 2
    # Próximo pago / Días al próximo / Interés prox / Amortización prox
    # =========================
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Próximo pago", next_dt.date().isoformat())
    c6.metric("Días al próximo pago (30/360)", days_30_360(today, next_dt))
    c7.metric("Interés a pagar en prox. pago (USD)", fmt_num(next_interest))
    c8.metric("Amortización a pagar prox. pago (USD)", fmt_num(next_amort))

    # =========================
    # KPIs – Línea 3
    # Capital pendiente / Interés pendiente / Cupón corrido / Valor teórico
    # =========================
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Capital pendiente de pago (USD)", fmt_num(capital_pendiente))
    c10.metric("Interés pendiente de pago (USD)", fmt_num(interes_pendiente))
    c11.metric("Cupón corrido (USD)", fmt_num(cupon_corrido))
    c12.metric("Valor teórico (USD)", fmt_num(valor_teorico))

    # =========================
    # KPIs – Línea 4
    # Precio USD (D) / Paridad vs VT / Precio USD equiv MEP / Paridad MEP vs VT
    # =========================
    c13, c14, c15, c16 = st.columns(4)

    # Precio USD (Ticker D)
    if price_usd_mkt is None or pd.isna(price_usd_mkt):
        c13.metric("Precio USD (Ticker D)", "—")
    else:
        c13.metric("Precio USD (Ticker D)", fmt_num(price_usd_mkt))

    # Paridad D vs VT (con color)
    with c14:
        metric_premium("Paridad vs VT (Ticker D)", prem_d)

    # Precio USD equivalente por MEP (ARS / MEP)
    if price_usd_mep_equiv is None or pd.isna(price_usd_mep_equiv):
        c15.metric("Precio USD equivalente MEP", "—")
    else:
        c15.metric("Precio USD equivalente MEP", fmt_num(price_usd_mep_equiv))

    # Paridad MEP vs VT (con color)
    with c16:
        metric_premium("Paridad vs VT (MEP)", prem_mep)

    # =========================
    # Gráfico cashflows futuros (stack capital + interés)
    # =========================
    fig = go.Figure()
    fig.add_bar(x=future["date"], y=future["Int."], name="Interés (USD)")
    fig.add_bar(x=future["date"], y=future["Principal"], name="Capital (USD)")

    fig.update_layout(
        barmode="stack",
        height=420,
        xaxis_title="Fecha",
        yaxis_title="Cashflow (USD)",
        title=f"Cashflows futuros (USD) – {tkr}",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# View principal
# =========================
def render():
    st.title("ONs - Análisis TIR% en (USD)")

    bd_path = os.path.join("data", "BD ONs.xlsx")

    with st.sidebar:
        st.subheader("TIR vs Maturity")
        use_t_plus_1 = st.toggle("Settlement T+1", value=True)
        last_cf_close_days = st.number_input(
            "Alerta: vencimiento final <= (días)",
            min_value=0,
            max_value=365,
            value=10,
        )

    @st.cache_data(ttl=60 * 60)  # BD no cambia tan seguido
    def _load_bd(bd_path: str) -> pd.DataFrame:
        return load_bd_ons_xlsx(bd_path)

    @st.cache_data(ttl=60 * 15)
    def _compute(df_bd: pd.DataFrame, use_t_plus_1: bool, last_cf_close_days: int) -> pd.DataFrame:
        return compute_ytm_table(
            df_bd,
            use_t_plus_1=use_t_plus_1,
            last_cf_close_days=last_cf_close_days,
        )

    df_bd = _load_bd(bd_path)
    df_raw = _compute(df_bd, use_t_plus_1, last_cf_close_days)

    # Solo OK (incluye OK con alertas)
    df = df_raw[df_raw["status"].astype(str).str.startswith("OK")].copy()

    left, right = st.columns([0.9, 1.4])

    # =========================
    # Tabla (izquierda)
    # =========================
    with left:
        st.subheader("Tabla")

        if df.empty:
            st.info("No hay resultados OK para mostrar.")
        else:
            base = df[[
                "ticker",
                "price_dirty_ars",
                "price_dirty_usd_mkt",
                "tasa_cupon_pct",
                "ytm_pct",
                "status",
            ]].copy()

            base = base.rename(columns={
                "ticker": "Ticker",
                "price_dirty_ars": "Precio ARS",
                "price_dirty_usd_mkt": "Precio USD",
                "tasa_cupon_pct": "Cupón",
                "ytm_pct": "TIR",
                "status": "_status",
            })

            base["Precio ARS"] = base["Precio ARS"].apply(fmt_ars)
            base["Precio USD"] = base["Precio USD"].apply(fmt_usd)
            base["Cupón"] = base["Cupón"].apply(fmt_pct)
            base["TIR"] = base["TIR"].apply(fmt_pct)
            base["_is_close"] = base["_status"].str.contains("LAST_CF_TOO_CLOSE", regex=False)

            table_html = build_html_table(
                base[["Ticker", "Precio ARS", "Precio USD", "Cupón", "TIR", "_is_close"]]
            )
            components.html(table_html, height=280, scrolling=True)

    # =========================
    # Gráfico (derecha)
    # =========================
    with right:
        st.subheader("TIR vs Maturity")

        plot_df = df.dropna(subset=["ytm_pct", "last_cf_date"]).copy()
        if plot_df.empty:
            st.info("No hay suficientes datos para graficar.")
        else:
            plot_df["last_cf_date"] = pd.to_datetime(plot_df["last_cf_date"])

            fig = px.scatter(
                plot_df,
                x="last_cf_date",
                y="ytm_pct",
                text="ticker",
                hover_data=["ticker", "ytm_pct", "price_dirty_ars", "tasa_cupon_pct"],
            )

            # Tendencia grado 2 (>=3 puntos)
            if len(plot_df) >= 3:
                x_sorted = plot_df.sort_values("last_cf_date")
                x_num = x_sorted["last_cf_date"].astype("int64") / 1e9
                y = x_sorted["ytm_pct"].astype(float)

                c2, c1, c0 = np.polyfit(x_num, y, 2)
                y_trend = c2 * (x_num ** 2) + c1 * x_num + c0

                fig.add_trace(
                    go.Scatter(
                        x=x_sorted["last_cf_date"],
                        y=y_trend,
                        mode="lines",
                        name="Tendencia (grado 2)",
                        line=dict(dash="dot"),
                    )
                )

            fig.update_traces(textposition="top center")
            fig.update_layout(
                xaxis_title="Maturity",
                yaxis_title="TIR (%)",
                height=520,
                margin=dict(l=10, r=10, t=10, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # Detalle por ticker (ancho completo, debajo)
    # =========================
    st.markdown("---")
    st.subheader("Detalle por ticker")

    if df.empty:
        st.info("No hay tickers OK para mostrar detalle.")
        return

    tickers_ok = sorted(df["ticker"].dropna().unique().tolist())
    selected = st.selectbox(
        "Elegí un ticker para ver más información",
        options=tickers_ok,
        index=0,
    )

    row_sel = df[df["ticker"] == selected].copy()

    price_usd = (
        row_sel["price_dirty_usd_mkt"].dropna().iloc[0]
        if "price_dirty_usd_mkt" in row_sel.columns and not row_sel["price_dirty_usd_mkt"].dropna().empty
        else None
    )

    price_ars = (
        row_sel["price_dirty_ars"].dropna().iloc[0]
        if "price_dirty_ars" in row_sel.columns and not row_sel["price_dirty_ars"].dropna().empty
        else None
    )

    mep_venta = (
        row_sel["mep_venta"].dropna().iloc[0]
        if "mep_venta" in row_sel.columns and not row_sel["mep_venta"].dropna().empty
        else None
    )

    price_usd_mep_equiv = (
        row_sel["price_dirty_usd_used"].dropna().iloc[0]
        if "price_dirty_usd_used" in row_sel.columns and not row_sel["price_dirty_usd_used"].dropna().empty
        else None
    )

    with st.expander(f"Ver cashflows futuros: {selected}", expanded=True):
        render_ticker_detail(
            selected,
            df_bd,
            price_usd_mkt=price_usd,
            price_ars=price_ars,
            mep_venta=mep_venta,
            price_usd_mep_equiv=price_usd_mep_equiv,
        )
