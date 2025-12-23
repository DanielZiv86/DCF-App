# views/ons_ytm.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ons_ytm_data import load_bd_ons_xlsx, compute_ytm_table
from app_theme import DCF_PLOTLY_TEMPLATE


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


def _infer_payment_frequency(future_dates: pd.Series) -> str:
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


def _normalize_leg(x: str) -> str:
    """Normaliza legislación a un set corto para filtros y color."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = str(x).strip()
    if not s:
        return "—"

    s_low = s.lower()
    # Ajustá acá si en tu BD viene "ARG", "Argentina", "Ley AR", etc.
    if "arg" in s_low or s_low == "ar":
        return "Arg"
    if "ny" in s_low or "new york" in s_low:
        return "NY"
    return s  # fallback (por si tenés otros)


def _build_meta_from_bd(df_bd: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de la BD de cashflows (muchas filas por ticker),
    arma una tabla 'meta' 1 fila por ticker con:
      - legislacion
      - lamina_minima
    """
    bd = df_bd.copy()

    if "ticker" not in bd.columns:
        return pd.DataFrame(columns=["ticker", "legislacion", "lamina_minima"])

    def first_non_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if not s2.empty else np.nan

    cols = ["ticker"]
    if "legislacion" in bd.columns:
        cols.append("legislacion")
    if "lamina_minima" in bd.columns:
        cols.append("lamina_minima")

    bd = bd[cols].copy()

    meta = (
        bd.groupby("ticker", as_index=False)
        .agg(
            legislacion=("legislacion", first_non_null) if "legislacion" in bd.columns else ("ticker", lambda x: np.nan),
            lamina_minima=("lamina_minima", first_non_null) if "lamina_minima" in bd.columns else ("ticker", lambda x: np.nan),
        )
    )

    meta["legislacion"] = meta["legislacion"].apply(_normalize_leg)
    meta["lamina_minima"] = pd.to_numeric(meta["lamina_minima"], errors="coerce").astype("Int64")

    return meta


# =========================
# Detalle ticker
# =========================
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

    def _first_non_null(col: str):
        if col not in sub.columns:
            return "—"
        s = sub[col].dropna()
        if s.empty:
            return "—"
        v = s.iloc[0]
        return "—" if pd.isna(v) else str(v)

    legislacion = _normalize_leg(_first_non_null("legislacion"))
    lamina_min = _first_non_null("lamina_minima")

    sub["Principal"] = pd.to_numeric(sub.get("Principal"), errors="coerce").fillna(0.0)
    sub["Int."] = pd.to_numeric(sub.get("Int."), errors="coerce").fillna(0.0)

    future = sub[sub["date"] >= today].copy()
    if future.empty:
        st.info("No hay cashflows futuros.")
        return

    past = sub[sub["date"] < today].copy()

    maturity_dt = future["date"].max()
    next_dt = future["date"].min()

    next_interest = float(future.loc[future["date"] == next_dt, "Int."].sum())
    next_amort = float(future.loc[future["date"] == next_dt, "Principal"].sum())

    capital_pendiente = float(future["Principal"].sum())
    interes_pendiente = float(future["Int."].sum())

    periodicidad = _infer_payment_frequency(future["date"])

    cupon_corrido = 0.0
    if not past.empty:
        last_dt = past["date"].max()
        days_period = days_30_360(last_dt, next_dt)
        days_elapsed = days_30_360(last_dt, today)
        if days_period > 0 and days_elapsed > 0:
            cupon_corrido = next_interest * (days_elapsed / days_period)

    valor_teorico = capital_pendiente + cupon_corrido

    def _premium_pct(price: float | None, vt: float) -> float | None:
        if price is None or pd.isna(price) or vt <= 0:
            return None
        return (float(price) / vt - 1.0) * 100.0

    prem_d = _premium_pct(price_usd_mkt, valor_teorico)
    prem_mep = _premium_pct(price_usd_mep_equiv, valor_teorico)

    def metric_premium(label: str, pct: float | None):
        if pct is None:
            st.metric(label, "—")
        else:
            st.metric(
                label,
                value=" ",
                delta=f"{pct:+.2f}%",
                delta_color="inverse",
                help="Premium (+) / Discount (-) vs Valor Teórico",
            )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Maturity", maturity_dt.date().isoformat())
    c2.metric("Periodicidad", periodicidad)
    c3.metric("Legislación", legislacion)
    c4.metric("Lámina mínima", lamina_min)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Próximo pago", next_dt.date().isoformat())
    c6.metric("Días al próximo pago (30/360)", days_30_360(today, next_dt))
    c7.metric("Interés a pagar en prox. pago (USD)", fmt_num(next_interest))
    c8.metric("Amortización a pagar prox. pago (USD)", fmt_num(next_amort))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Capital pendiente de pago (USD)", fmt_num(capital_pendiente))
    c10.metric("Interés pendiente de pago (USD)", fmt_num(interes_pendiente))
    c11.metric("Cupón corrido (USD)", fmt_num(cupon_corrido))
    c12.metric("Valor teórico (USD)", fmt_num(valor_teorico))

    c13, c14, c15, c16 = st.columns(4)

    if price_usd_mkt is None or pd.isna(price_usd_mkt):
        c13.metric("Precio USD (Ticker D)", "—")
    else:
        c13.metric("Precio USD (Ticker D)", fmt_num(price_usd_mkt))

    with c14:
        metric_premium("Paridad vs VT (Ticker D)", prem_d)

    if price_usd_mep_equiv is None or pd.isna(price_usd_mep_equiv):
        c15.metric("Precio USD equivalente MEP", "—")
    else:
        c15.metric("Precio USD equivalente MEP", fmt_num(price_usd_mep_equiv))

    with c16:
        metric_premium("Paridad vs VT (MEP)", prem_mep)

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
        template=DCF_PLOTLY_TEMPLATE,
    )

    st.plotly_chart(fig, width="stretch")


# =========================
# View principal
# =========================
def render():
    st.title("ONs - Análisis TIR% en (USD)")
    bd_path = os.path.join("data", "BD ONs.xlsx")

    @st.cache_data(ttl=60 * 60)
    def _load_bd(path: str) -> pd.DataFrame:
        return load_bd_ons_xlsx(path)

    @st.cache_data(ttl=60 * 15)
    def _compute(df_bd: pd.DataFrame, use_t_plus_1: bool, last_cf_close_days: int) -> pd.DataFrame:
        return compute_ytm_table(
            df_bd,
            use_t_plus_1=use_t_plus_1,
            last_cf_close_days=last_cf_close_days,
        )

    # 1) Cargamos BD y metadatos (esto permite armar filtros sin depender de df)
    df_bd = _load_bd(bd_path)
    meta = _build_meta_from_bd(df_bd)

    # 2) Sidebar (filtros arriba, luego settings), sin depender de df
    with st.sidebar:
        st.subheader("Filtros ONs")

        # Legislación (desde meta)
        leg_opts = sorted(
            [x for x in meta["legislacion"].dropna().unique().tolist() if str(x).strip() != ""]
        ) if not meta.empty else []
        sel_legs = st.multiselect(
            "Legislación",
            options=leg_opts,
            default=leg_opts,
            help="La legislación define qué normas regulan la emisión y qué tribunales intervienen en caso de disputa.",
        )

        # Lámina mínima (discreto, desde meta)
        lam_opts = (
            meta["lamina_minima"].dropna().astype(int).unique().tolist()
            if (not meta.empty and "lamina_minima" in meta.columns)
            else []
        )
        lam_opts = sorted(lam_opts)

        preferred = [1, 100, 1000, 10000]
        lam_opts_sorted = [x for x in preferred if x in lam_opts] + [x for x in lam_opts if x not in preferred]

        sel_laminas = st.multiselect(
            "Lámina mínima (VN)",
            options=lam_opts_sorted,
            default=lam_opts_sorted,
            help="Cantidad de VN mínimos a comprar.",
        )

        st.divider()
        st.subheader("TIR vs Maturity")
        use_t_plus_1 = st.toggle("Settlement T+1", value=True)
        last_cf_close_days = st.number_input(
            "Alerta: vencimiento final <= (días)",
            min_value=0,
            max_value=365,
            value=10,
        )

    # 3) Computo final (depende de toggles)
    df_raw = _compute(df_bd, use_t_plus_1, last_cf_close_days)
    df = df_raw[df_raw["status"].astype(str).str.startswith("OK")].copy()

    # Merge meta (legislacion + lamina_minima)
    if not meta.empty and "ticker" in df.columns:
        df = df.merge(meta, on="ticker", how="left")
    else:
        df["legislacion"] = "—"
        df["lamina_minima"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # 4) Aplicar filtros (tabla + gráfico)
    df_f = df.copy()

    if sel_legs:
        df_f = df_f[df_f["legislacion"].isin(sel_legs)].copy()
    else:
        df_f = df_f.iloc[0:0].copy()

    df_f["lamina_minima"] = pd.to_numeric(df_f["lamina_minima"], errors="coerce").astype("Int64")

    if sel_laminas:
        df_f = df_f[df_f["lamina_minima"].isin(sel_laminas)].copy()
    else:
        df_f = df_f.iloc[0:0].copy()

    # Layout como Letras: [1.2, 2.0]
    left, right = st.columns([1.2, 2.0])

    # Alturas simétricas
    CHART_H = 550
    TABLE_H = CHART_H

    # =========================
    # Tabla (izquierda)
    # =========================
    with left:
        st.subheader("Tabla")

        if df_f.empty:
            st.info("No hay resultados para los filtros seleccionados.")
        else:
            base = df_f[[
                "ticker",
                "price_dirty_ars",
                "price_dirty_usd_mkt",
                "tasa_cupon_pct",
                "ytm_pct",
                "legislacion",
                "lamina_minima",
                "status",
            ]].copy()

            base = base.rename(columns={
                "ticker": "Ticker",
                "price_dirty_ars": "Precio ARS",
                "price_dirty_usd_mkt": "Precio USD",
                "tasa_cupon_pct": "Cupón",
                "ytm_pct": "TIR",
                "legislacion": "Legislación",
                "lamina_minima": "Lámina mínima",
                "status": "_status",
            })

            base["Precio ARS"] = base["Precio ARS"].apply(fmt_ars)
            base["Precio USD"] = base["Precio USD"].apply(fmt_usd)
            base["Cupón"] = base["Cupón"].apply(fmt_pct)
            base["TIR"] = base["TIR"].apply(fmt_pct)

            base["Lámina mínima"] = pd.to_numeric(base["Lámina mínima"], errors="coerce")
            base["Lámina mínima"] = base["Lámina mínima"].map(lambda v: "-" if pd.isna(v) else f"{int(v)}")

            # Orden sugerido (TIR desc)
            try:
                base["_tir_num"] = pd.to_numeric(df_f["ytm_pct"], errors="coerce")
                base = base.sort_values("_tir_num", ascending=False).drop(columns=["_tir_num"])
            except Exception:
                pass

            show = base[["Ticker", "Precio ARS", "Precio USD", "Cupón", "TIR", "Legislación", "Lámina mínima"]].copy()

            st.dataframe(
                show.set_index("Ticker"),
                width="stretch",
                height=TABLE_H,
            )

    # =========================
    # Gráfico (derecha)
    # =========================
    with right:
        st.subheader("TIR vs Maturity")

        plot_df = df_f.dropna(subset=["ytm_pct", "last_cf_date"]).copy()
        if plot_df.empty:
            st.info("No hay suficientes datos para graficar.")
        else:
            plot_df["last_cf_date"] = pd.to_datetime(plot_df["last_cf_date"])

            # Color: legislación Arg vs resto (naranja para Arg)
            plot_df["leg_color_group"] = plot_df["legislacion"].apply(
                lambda x: "Ley Arg" if _normalize_leg(x) == "Arg" else "Ley NY"
            )

            fig = px.scatter(
                plot_df,
                x="last_cf_date",
                y="ytm_pct",
                text="ticker",
                color="leg_color_group",
                color_discrete_map={
                    "Ley NY": "#9ecae1",     # celeste
                    "Ley Arg": "#f28e2b",  # naranja
                },
                hover_data=["ticker", "ytm_pct", "price_dirty_ars", "tasa_cupon_pct", "legislacion", "lamina_minima"],
                category_orders={"leg_color_group": ["Ley NY", "Ley Arg"]},
            )

            fig.update_layout(legend_title_text="")

            # Tendencia grado 2 (>=3 puntos) sobre TODOS los puntos filtrados
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
                        name="Tendencia",
                        line=dict(dash="dot"),
                        hovertemplate="Tendencia<br>TIR: %{y:.2f}%<extra></extra>",
                    )
                )

            fig.update_traces(textposition="top center")
            fig.update_layout(
                xaxis_title="Maturity",
                yaxis_title="TIR (%)",
                height=CHART_H,
                margin=dict(l=10, r=10, t=10, b=10),
                template=DCF_PLOTLY_TEMPLATE,
                legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            )

            st.plotly_chart(fig, width="stretch")

    # =========================
    # Detalle por ticker (ancho completo, debajo)
    # =========================
    st.markdown("---")
    st.subheader("Detalle por ticker")

    if df_f.empty:
        st.info("No hay tickers para mostrar detalle con los filtros actuales.")
        return

    tickers_ok = sorted(df_f["ticker"].dropna().unique().tolist())
    selected = st.selectbox(
        "Elegí un ticker para ver más información",
        options=tickers_ok,
        index=0,
    )

    row_sel = df_f[df_f["ticker"] == selected].copy()

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
