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
        leg_opts = (
            sorted([x for x in meta["legislacion"].dropna().unique().tolist() if str(x).strip() != ""])
            if (meta is not None and not meta.empty and "legislacion" in meta.columns)
            else []
        )
        sel_legs = st.multiselect(
            "Legislación",
            options=leg_opts,
            default=leg_opts,
            help="La legislación define qué normas regulan la emisión y qué tribunales intervienen en caso de disputa.",
        )

        # Lámina mínima (discreto, desde meta)
        lam_opts = (
            meta["lamina_minima"].dropna().astype(int).unique().tolist()
            if (meta is not None and not meta.empty and "lamina_minima" in meta.columns)
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
    df_raw = _compute(df_bd, use_t_plus_1, int(last_cf_close_days))
    df = df_raw[df_raw["status"].astype(str).str.startswith("OK")].copy()

    # Merge meta (legislacion + lamina_minima)
    if meta is not None and not meta.empty and "ticker" in df.columns:
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

    # =========================
    # KPIs + Insight (arriba)
    # =========================
    st.subheader("Resumen")

    if df_f.empty:
        st.info("No hay resultados para los filtros seleccionados.")
    else:
        tir = pd.to_numeric(df_f.get("ytm_pct"), errors="coerce")
        maturity = pd.to_datetime(df_f.get("last_cf_date"), errors="coerce")
        today = pd.Timestamp.today().normalize()

        tir_mean = float(tir.mean()) if tir.notna().any() else np.nan
        tir_p75 = float(tir.quantile(0.75)) if tir.notna().any() else np.nan

        years_to_mat = ((maturity - today).dt.days / 365.25) if maturity.notna().any() else pd.Series(dtype=float)
        years_mean = float(years_to_mat.mean()) if years_to_mat.notna().any() else np.nan

        mep_med = (
            float(pd.to_numeric(df_f.get("mep_venta"), errors="coerce").median())
            if "mep_venta" in df_f.columns
            else np.nan
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("ONs filtradas", f"{len(df_f):,}".replace(",", "."))
        k2.metric("TIR promedio", fmt_pct(tir_mean) if np.isfinite(tir_mean) else "—")
        k3.metric("TIR p75", fmt_pct(tir_p75) if np.isfinite(tir_p75) else "—", help="Percentil 75 de TIR (top 25% por retorno).")
        k4.metric("Maturity promedio", f"{fmt_num(years_mean, 2)} años" if np.isfinite(years_mean) else "—")
        k5.metric("MEP Venta (IOL)", fmt_num(mep_med, 2) if np.isfinite(mep_med) else "—", help="Referencia para convertir ARS a USD (equivalente MEP).")

        # Insight simple por legislación (si hay datos de ambos grupos)
        df_tmp = df_f.copy()
        df_tmp["_leg_norm"] = df_tmp["legislacion"].apply(_normalize_leg)
        tir_by = df_tmp.groupby("_leg_norm")["ytm_pct"].mean(numeric_only=True) if "ytm_pct" in df_tmp.columns else pd.Series(dtype=float)

        if ("NY" in tir_by.index) and ("Arg" in tir_by.index) and np.isfinite(tir_by.get("NY")) and np.isfinite(tir_by.get("Arg")):
            delta_bps = (float(tir_by["NY"]) - float(tir_by["Arg"])) * 100.0  # 1% = 100 bps
            better = "NY" if delta_bps > 0 else "Arg"
            st.info(
                f"📌 **Insight:** con los filtros actuales, las ONs bajo **ley {better}** rinden en promedio "
                f"**{abs(delta_bps):,.0f} bps** {'más' if delta_bps > 0 else 'menos'} que las de la otra legislación."
                .replace(",", ".")
            )
        else:
            st.caption("Tip: usá el filtro de Legislación para comparar comportamiento por marco legal.")

    # Layout como Letras: [1.2, 2.0]
    left, right = st.columns([1.2, 2.0])

    CHART_H = 550
    TABLE_H = CHART_H

    # =========================
    # Tabla + Buckets (izquierda)
    # =========================
    with left:
        st.subheader("Tabla")

        if df_f.empty:
            st.info("No hay resultados para los filtros seleccionados.")
        else:
            df_table = df_f.copy()

            # Asegurar tipos
            df_table["ytm_pct"] = pd.to_numeric(df_table.get("ytm_pct"), errors="coerce")
            df_table["tasa_cupon_pct"] = pd.to_numeric(df_table.get("tasa_cupon_pct"), errors="coerce")
            df_table["price_dirty_ars"] = pd.to_numeric(df_table.get("price_dirty_ars"), errors="coerce")
            df_table["price_dirty_usd_mkt"] = pd.to_numeric(df_table.get("price_dirty_usd_mkt"), errors="coerce")
            df_table["lamina_minima"] = pd.to_numeric(df_table.get("lamina_minima"), errors="coerce").astype("Int64")
            df_table["last_cf_date"] = pd.to_datetime(df_table.get("last_cf_date"), errors="coerce")

            df_table = df_table.sort_values("ytm_pct", ascending=False, na_position="last")

            # =========================
            # Alertas visuales + Buckets
            # =========================
            today = pd.Timestamp.today().normalize()
            mat_dt = df_table["last_cf_date"]
            years_to_mat = (mat_dt - today).dt.days / 365.25

            tir_num = df_table["ytm_pct"]
            lam_num = pd.to_numeric(df_table["lamina_minima"], errors="coerce")

            is_arg = df_table["legislacion"].apply(lambda x: _normalize_leg(x) == "Arg")
            is_short = years_to_mat.notna() & (years_to_mat <= 1.5)

            # ⭐ Lámina mínima = 1
            is_lam_one = lam_num.notna() & (lam_num == 1)

            # Baja lámina (para bucket): por defecto <= 100 VN
            is_low_lam = lam_num.notna() & (lam_num <= 100)

            near_mat = mat_dt.notna() & ((mat_dt - today).dt.days <= int(last_cf_close_days))

            top3_idx = (
                df_table["ytm_pct"]
                .dropna()
                .nlargest(3)
                .index
            )

            is_top3_tir = df_table.index.isin(top3_idx)


            def _alert_row(_is_arg: bool, _near: bool, _lam_one: bool, _top3: bool) -> str:
                marks = []
                if _near:
                    marks.append("⚠️")
                if _lam_one:
                    marks.append("⭐")
                if _top3:
                    marks.append("🎯")
                if _is_arg:
                    marks.append("🇦🇷")
                return " ".join(marks)

            df_table["⚑"] = [
                _alert_row(a, n, l1, t3)
                for a, n, l1, t3 in zip(
                    is_arg.tolist(),
                    near_mat.tolist(),
                    is_lam_one.tolist(),
                    is_top3_tir.tolist(),
                )
            ]


            # Buckets
            top_tir = df_table.loc[tir_num.notna()].nlargest(5, "ytm_pct").copy()
            low_lamina = df_table.loc[is_low_lam].sort_values("ytm_pct", ascending=False, na_position="last").head(5).copy()
            corto_plazo = df_table.loc[is_short].sort_values("ytm_pct", ascending=False, na_position="last").head(5).copy()
            ley_arg = df_table.loc[is_arg].sort_values("ytm_pct", ascending=False, na_position="last").head(5).copy()

            with st.expander("Rankings rápidos (buckets)", expanded=False):
                t1, t2, t3, t4 = st.tabs(["Top TIR", "Baja lámina", "Corto plazo", "Ley Arg"])

                def _bucket_table(dd: pd.DataFrame):
                    if dd.empty:
                        st.caption("Sin resultados con los filtros actuales.")
                        return
                    tmp = dd[["ticker", "ytm_pct", "price_dirty_usd_mkt", "lamina_minima", "legislacion"]].rename(
                        columns={
                            "ticker": "Ticker",
                            "ytm_pct": "TIR (%)",
                            "price_dirty_usd_mkt": "USD (D)",
                            "lamina_minima": "Lámina",
                            "legislacion": "Ley",
                        }
                    )
                    st.dataframe(
                        tmp,
                        width="stretch",
                        height=220,
                        hide_index=True,
                        column_config={
                            "Ticker": st.column_config.TextColumn(),
                            "TIR (%)": st.column_config.NumberColumn(format="%.2f"),
                            "USD (D)": st.column_config.NumberColumn(format="USD %.2f"),
                            "Lámina": st.column_config.NumberColumn(format="%d"),
                            "Ley": st.column_config.TextColumn(),
                        },
                    )

                with t1:
                    _bucket_table(top_tir)
                with t2:
                    _bucket_table(low_lamina)
                with t3:
                    _bucket_table(corto_plazo)
                with t4:
                    _bucket_table(ley_arg)

            # Tabla principal
            show = df_table[
                [
                    "⚑",
                    "ticker",
                    "price_dirty_ars",
                    "price_dirty_usd_mkt",
                    "tasa_cupon_pct",
                    "ytm_pct",
                    "legislacion",
                    "lamina_minima",
                ]
            ].rename(
                columns={
                    "⚑": "",
                    "ticker": "Ticker",
                    "price_dirty_ars": "Precio ARS",
                    "price_dirty_usd_mkt": "Precio USD (D)",
                    "tasa_cupon_pct": "Cupón (%)",
                    "ytm_pct": "TIR (%)",
                    "legislacion": "Legislación",
                    "lamina_minima": "Lámina mínima (VN)",
                }
            )

            st.dataframe(
                show,
                width="stretch",
                height=TABLE_H,
                hide_index=True,
                column_config={
                    "": st.column_config.TextColumn(help="Alertas: ⚠️ vencimiento cercano | ⭐ lámina mínima = 1 | 🎯 Top 3 TIR | 🇦🇷 ley Arg"),
                    "Ticker": st.column_config.TextColumn(help="Símbolo de la ON."),
                    "Precio ARS": st.column_config.NumberColumn(format="$ %.0f"),
                    "Precio USD (D)": st.column_config.NumberColumn(format="USD %.2f", help="Precio de la especie D (si existe en IOL)."),
                    "Cupón (%)": st.column_config.NumberColumn(format="%.2f"),
                    "TIR (%)": st.column_config.NumberColumn(format="%.2f", help="TIR anualizada en USD al precio actual."),
                    "Legislación": st.column_config.TextColumn(
                        help="La legislación define qué normas regulan la emisión y qué tribunales intervienen en caso de disputa."
                    ),
                    "Lámina mínima (VN)": st.column_config.NumberColumn(format="%d", help="Cantidad de VN mínimos a comprar."),
                },
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

            plot_df["leg_color_group"] = plot_df["legislacion"].apply(
                lambda x: "Ley Arg" if _normalize_leg(x) == "Arg" else "Ley NY"
            )

            lam = pd.to_numeric(plot_df.get("lamina_minima"), errors="coerce").fillna(1).astype(float)
            plot_df["lamina_size_log"] = np.log10(lam.clip(lower=1.0)) + 1.0

            plot_df["TIR (%)"] = pd.to_numeric(plot_df.get("ytm_pct"), errors="coerce")
            plot_df["Cupón (%)"] = pd.to_numeric(plot_df.get("tasa_cupon_pct"), errors="coerce")
            plot_df["Precio ARS"] = pd.to_numeric(plot_df.get("price_dirty_ars"), errors="coerce")
            plot_df["Precio USD (D)"] = pd.to_numeric(plot_df.get("price_dirty_usd_mkt"), errors="coerce")
            plot_df["Lámina mínima (VN)"] = pd.to_numeric(plot_df.get("lamina_minima"), errors="coerce")

            fig = px.scatter(
                plot_df,
                x="last_cf_date",
                y="ytm_pct",
                text="ticker",
                color="leg_color_group",
                size="lamina_size_log",
                size_max=18,
                color_discrete_map={
                    "Ley NY": "#9ecae1",
                    "Ley Arg": "#f28e2b",
                },
                hover_name="ticker",
                hover_data={
                    "TIR (%)": True,
                    "Cupón (%)": True,
                    "Precio ARS": True,
                    "Precio USD (D)": True,
                    "Lámina mínima (VN)": True,
                    "leg_color_group": False,
                    "ytm_pct": False,
                    "tasa_cupon_pct": False,
                    "price_dirty_ars": False,
                    "price_dirty_usd_mkt": False,
                    "lamina_minima": False,
                    "lamina_size_log": False,
                },
                category_orders={"leg_color_group": ["Ley NY", "Ley Arg"]},
            )
            fig.update_layout(legend_title_text="")

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
                yaxis_tickformat=".2f",
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
