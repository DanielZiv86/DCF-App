# ons_ytm_data.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import requests

from io import StringIO
# =========================
# Config
# =========================
URL_ONS = "https://iol.invertironline.com/mercado/cotizaciones/argentina/obligaciones-negociables/todos"
URL_MONEDAS = "https://iol.invertironline.com/mercado/cotizaciones/argentina/monedas"

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "es-AR,es;q=0.9"}


# =========================
# Helpers
# =========================
def _parse_number_ar(s: str) -> float:
    s = str(s).strip()
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    return float(s)


def get_mep_venta_iol() -> float:
    """MEP Venta tomado de 'Dólar MEP - Ley Local (AL30D) *'."""
    html = requests.get(URL_MONEDAS, headers=HEADERS, timeout=25).text
    df = pd.read_html(StringIO(html))[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    df["Moneda_norm"] = (
        df["Moneda"].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    target = "Dólar MEP - Ley Local (AL30D)"
    row = df[df["Moneda_norm"].str.contains(target, regex=False)]
    if row.empty:
        raise ValueError(f"No encontré la fila objetivo '{target}' en la tabla de IOL.")

    return _parse_number_ar(row.iloc[0]["Venta"])


def load_bd_ons_xlsx(bd_path: str) -> pd.DataFrame:
    """Carga y normaliza tu BD (cashflows en USD base 100) preservando columnas extra (ej: legislacion, lamina_minima)."""
    if not os.path.exists(bd_path):
        raise FileNotFoundError(f"No encuentro el archivo: {bd_path} (cwd={os.getcwd()})")

    df = pd.read_excel(bd_path)

    # 1) Normalizar nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]

    # Mapa para encontrar columnas aunque vengan con mayúsculas/acentos/espacios
    col_map = {c.lower().strip(): c for c in df.columns}

    # Principal puede venir mal escrito
    principal_col = None
    if "principal" in col_map:
        principal_col = col_map["principal"]
    elif "pricipal" in col_map:  # typo común
        principal_col = col_map["pricipal"]

    if principal_col is None:
        raise KeyError("No encontré la columna 'Principal' ni 'Pricipal' en BD ONs.xlsx")

    # Columnas mínimas requeridas para cálculos
    needed_lower = ["ticker", "date", "int.", "cf", "tasa de cupon", "dias"]
    missing = [c for c in needed_lower if c not in col_map]
    if missing:
        raise KeyError(f"Faltan columnas en BD: {missing}. Columnas disponibles: {df.columns.tolist()}")

    # Resolver nombres reales según Excel
    ticker_col = col_map["ticker"]
    date_col = col_map["date"]
    int_col = col_map["int."]
    cf_col = col_map["cf"]
    tasa_col = col_map["tasa de cupon"]
    dias_col = col_map["dias"]

    # 2) Renombrar a nombres estándar, pero SIN perder el resto de columnas
    rename = {
        ticker_col: "ticker",
        date_col: "date",
        principal_col: "Principal",
        int_col: "Int.",
        cf_col: "cf",
        tasa_col: "Tasa de Cupon",
        dias_col: "Dias",
    }

    # También normalizamos (si existen) legislacion y lamina_minima a nombres fijos
    if "legislacion" in col_map:
        rename[col_map["legislacion"]] = "legislacion"
    if "lamina_minima" in col_map:
        rename[col_map["lamina_minima"]] = "lamina_minima"

    df = df.rename(columns=rename).copy()

    # 3) Limpieza / tipos
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    for col in ["Principal", "Int.", "cf", "Tasa de Cupon", "Dias"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4) Orden y filtrado mínimo (NO eliminamos legislacion/lamina_minima)
    df = df.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    return df



def _to_usd_ticker_from_ars(ticker: str) -> Optional[str]:
    t = str(ticker).strip().upper()
    if len(t) >= 2 and t.endswith("O"):
        return t[:-1] + "D"
    return None


def get_prices_iol_ons(tickers_ars: List[str]) -> pd.DataFrame:
    """
    Trae precios ON desde IOL y devuelve:
      ticker (ARS), price_dirty_ars,
      ticker_usd (derivado), price_dirty_usd_mkt (si existe en tabla),
      daily_change_ars, daily_change_usd
    """
    html = requests.get(URL_ONS, headers=HEADERS, timeout=25).text
    iol = pd.read_html(StringIO(html))[0].copy()

    iol = iol[["Símbolo", "Último Operado", "Variación Diaria"]].rename(columns={
        "Símbolo": "ticker",
        "Último Operado": "price_dirty",
        "Variación Diaria": "daily_change"
    })

    iol["ticker"] = iol["ticker"].astype(str).str.strip().str.upper()

    # Parse número AR (sirve tanto para ARS como para USD en la tabla)
    # OJO: no borremos "-" porque puede ser "sin precio" -> NaN y se filtra después.
    iol["price_dirty"] = (
        iol["price_dirty"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    iol["price_dirty"] = pd.to_numeric(iol["price_dirty"], errors="coerce")

    iol["daily_change"] = iol["daily_change"].astype(str).str.strip()

    iol = iol.dropna(subset=["price_dirty"])
    iol = iol[iol["price_dirty"] > 0].reset_index(drop=True)

    # --- Armamos mapping ARS->USD tickers ---
    tickers_ars_norm = [str(t).strip().upper() for t in tickers_ars]
    map_ars_to_usd = {t: _to_usd_ticker_from_ars(t) for t in tickers_ars_norm}
    tickers_usd = sorted({v for v in map_ars_to_usd.values() if v})

    # --- Precios ARS (ticker O) ---
    iol_ars = (
        iol[iol["ticker"].isin(tickers_ars_norm)]
        .rename(columns={"price_dirty": "price_dirty_ars", "daily_change": "daily_change_ars"})
        .copy()
    )

    # --- Precios USD (ticker D) ---
    iol_usd = (
        iol[iol["ticker"].isin(tickers_usd)]
        .rename(columns={"ticker": "ticker_usd", "price_dirty": "price_dirty_usd_mkt", "daily_change": "daily_change_usd"})
        .copy()
    )

    # ✅ Fix escala: IOL suele mostrar el "Último" de la especie D *100 (ej 106,60 aparece 10.660)
    # Regla robusta: si es > 1000, casi seguro está en centavos -> /100
    if not iol_usd.empty and "price_dirty_usd_mkt" in iol_usd.columns:
        iol_usd["price_dirty_usd_mkt"] = np.where(
            iol_usd["price_dirty_usd_mkt"] > 1000,
            iol_usd["price_dirty_usd_mkt"] / 100.0,
            iol_usd["price_dirty_usd_mkt"]
        )

    # Base con todos los tickers ARS
    base = pd.DataFrame({"ticker": tickers_ars_norm})
    base["ticker_usd"] = base["ticker"].map(map_ars_to_usd)

    # Merge: ARS por ticker, USD por ticker_usd
    out = base.merge(
        iol_ars[["ticker", "price_dirty_ars", "daily_change_ars"]],
        on="ticker",
        how="left",
    ).merge(
        iol_usd[["ticker_usd", "price_dirty_usd_mkt", "daily_change_usd"]],
        on="ticker_usd",
        how="left",
    )

    out = out.sort_values("ticker").reset_index(drop=True)
    return out


def days_30_360(d1: date, d2: date) -> int:
    d1d = min(d1.day, 30)
    d2d = d2.day
    if d1.day == 31:
        d1d = 30
    if d2.day == 31 and d1d == 30:
        d2d = 30
    return (d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (d2d - d1d)


def solve_ytm_bisect(settle: date, flows: List[Tuple[date, float]], dirty_price: float,
                     lo: float = -0.90, hi: float = 2.00, iters: int = 200) -> float:
    """YTM tal que -P + PV(flows)=0, con 30/360."""
    def f(y: float) -> float:
        total = -dirty_price
        for dt, cf in flows:
            if dt < settle:
                continue
            t = days_30_360(settle, dt) / 360.0
            total += cf / ((1 + y) ** t)
        return total

    flo, fhi = f(lo), f(hi)
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        raise ValueError("No pude encerrar la raíz (revisar precio/flows o rango lo/hi).")

    for _ in range(iters):
        mid = (lo + hi) / 2
        fmid = f(mid)
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return (lo + hi) / 2


def _get_flows(df_bd: pd.DataFrame, ticker: str) -> List[Tuple[date, float]]:
    sub = (
        df_bd[df_bd["ticker"] == ticker]
        .dropna(subset=["date", "cf"])
        .sort_values("date")
    )
    return list(zip(sub["date"].tolist(), sub["cf"].astype(float).tolist()))


def _flow_dates_info(flows: List[Tuple[date, float]], settle: date) -> Tuple[Optional[date], Optional[date]]:
    if not flows:
        return None, None
    dates = [d for d, _ in flows]
    last_dt = max(dates) if dates else None
    future_dates = [d for d, _ in flows if d >= settle]
    next_dt = min(future_dates) if future_dates else None
    return next_dt, last_dt


def compute_ytm_table(
    df_bd: pd.DataFrame,
    settle: Optional[date] = None,
    use_t_plus_1: bool = True,
    last_cf_close_days: int = 10
) -> pd.DataFrame:
    """
    Devuelve tabla con:
    ticker, price_dirty_ars, tasa_cupon_pct, ytm_pct, next_cf_date, last_cf_date, status
    """
    tickers = sorted(df_bd["ticker"].unique().tolist())

    # Settlement
    if settle is None:
        settle = date.today()
        if use_t_plus_1:
            settle = settle + timedelta(days=1)

    # Precio + MEP
    iol = get_prices_iol_ons(tickers)
    mep_venta = get_mep_venta_iol()

    iol["mep_venta"] = mep_venta

    # <-- ESTE es el USD usado para calcular TIR (como al principio)
    iol["price_dirty_usd_used"] = iol["price_dirty_ars"] / iol["mep_venta"]

    # Cupón por ticker (tomamos el primero)
    coupon = (
        df_bd.groupby("ticker", as_index=False)["Tasa de Cupon"]
        .first()
        .rename(columns={"Tasa de Cupon": "tasa_cupon_dec"})
    )
    coupon["tasa_cupon_pct"] = coupon["tasa_cupon_dec"] * 100

    # Queremos incluir TODOS los tickers de BD aunque no tengan precio hoy:
    base = pd.DataFrame({"ticker": tickers})
    base = base.merge(coupon[["ticker", "tasa_cupon_pct"]], on="ticker", how="left")
    base = base.merge(iol, on="ticker", how="left")

    rows = []
    for _, r in base.iterrows():
        tkr = r["ticker"]
        # Defaults to avoid UnboundLocalError in any early-return branch
        status = "UNKNOWN"
        ytm = np.nan

        flows_all = _get_flows(df_bd, tkr)
        next_dt, last_dt = _flow_dates_info(flows_all, settle)

        alerts = []
        days_to_last = None
        if last_dt is not None:
            days_to_last = (last_dt - settle).days
            if last_dt < settle:
                alerts.append("LAST_CF_PASSED")
            elif days_to_last <= last_cf_close_days:
                alerts.append("LAST_CF_TOO_CLOSE")

        # Si no hay precio hoy:
        if pd.isna(r.get("price_dirty_ars")) or pd.isna(r.get("price_dirty_usd_used")):
            status = "NO_PRICE_TODAY"
            if alerts:
                status += " | " + ",".join(alerts)
            rows.append({
                "ticker": tkr,
                "price_dirty_ars": (float(r["price_dirty_ars"]) if not pd.isna(r.get("price_dirty_ars")) else np.nan),
                "price_dirty_usd_mkt": r.get("price_dirty_usd_mkt"),
                "price_dirty_usd_used": float(r.get("price_dirty_usd_used")) if not pd.isna(r.get("price_dirty_usd_used")) else np.nan,
                "mep_venta": float(r.get("mep_venta")) if not pd.isna(r.get("mep_venta")) else np.nan,
                "tasa_cupon_pct": r.get("tasa_cupon_pct"),
                "ytm_pct": (round(float(ytm) * 100, 2) if np.isfinite(ytm) else np.nan),
                "next_cf_date": next_dt,
                "last_cf_date": last_dt,
                "status": status,
            })
            continue

        price_usd = float(r["price_dirty_usd_used"])

        # Filtrar flujos futuros
        flows_future = [(d, cf) for d, cf in flows_all if d >= settle and abs(cf) > 1e-12]

        if len(flows_all) == 0:
            status = "NO_CASHFLOWS"
            if alerts:
                status += " | " + ",".join(alerts)
            rows.append({
                "ticker": tkr,
                "price_dirty_ars": (float(r["price_dirty_ars"]) if not pd.isna(r.get("price_dirty_ars")) else np.nan),
                "price_dirty_usd_mkt": r.get("price_dirty_usd_mkt"),
                "price_dirty_usd_used": float(r.get("price_dirty_usd_used")) if not pd.isna(r.get("price_dirty_usd_used")) else np.nan,
                "mep_venta": float(r.get("mep_venta")) if not pd.isna(r.get("mep_venta")) else np.nan,
                "tasa_cupon_pct": r.get("tasa_cupon_pct"),
                "ytm_pct": (round(float(ytm) * 100, 2) if np.isfinite(ytm) else np.nan),
                "next_cf_date": next_dt,
                "last_cf_date": last_dt,
                "status": status,
            })
            continue

        if len(flows_future) == 0:
            status = "NO_FUTURE_CASHFLOWS"
            if alerts:
                status += " | " + ",".join(alerts)
            rows.append({
                "ticker": tkr,
                "price_dirty_ars": (float(r["price_dirty_ars"]) if not pd.isna(r.get("price_dirty_ars")) else np.nan),
                "price_dirty_usd_mkt": r.get("price_dirty_usd_mkt"),
                "price_dirty_usd_used": float(r.get("price_dirty_usd_used")) if not pd.isna(r.get("price_dirty_usd_used")) else np.nan,
                "mep_venta": float(r.get("mep_venta")) if not pd.isna(r.get("mep_venta")) else np.nan,
                "tasa_cupon_pct": r.get("tasa_cupon_pct"),
                "ytm_pct": (round(float(ytm) * 100, 2) if np.isfinite(ytm) else np.nan),
                "next_cf_date": next_dt,
                "last_cf_date": last_dt,
                "status": status,
            })
            continue

        try:
            ytm = solve_ytm_bisect(settle, flows_future, dirty_price=price_usd)
            status = "OK"
            if alerts:
                status += " | " + ",".join(alerts)
            rows.append({
                "ticker": tkr,
                "price_dirty_ars": (float(r["price_dirty_ars"]) if not pd.isna(r.get("price_dirty_ars")) else np.nan),
                "price_dirty_usd_mkt": r.get("price_dirty_usd_mkt"),
                "price_dirty_usd_used": float(r.get("price_dirty_usd_used")) if not pd.isna(r.get("price_dirty_usd_used")) else np.nan,
                "mep_venta": float(r.get("mep_venta")) if not pd.isna(r.get("mep_venta")) else np.nan,
                "tasa_cupon_pct": r.get("tasa_cupon_pct"),
                "ytm_pct": (round(float(ytm) * 100, 2) if np.isfinite(ytm) else np.nan),
                "next_cf_date": next_dt,
                "last_cf_date": last_dt,
                "status": status,
            })
        except Exception as e:
            status = f"ERR: {e}"
            if alerts:
                status += " | " + ",".join(alerts)
            rows.append({
                "ticker": tkr,
                "price_dirty_ars": (float(r["price_dirty_ars"]) if not pd.isna(r.get("price_dirty_ars")) else np.nan),
                "price_dirty_usd_mkt": r.get("price_dirty_usd_mkt"),
                "price_dirty_usd_used": float(r.get("price_dirty_usd_used")) if not pd.isna(r.get("price_dirty_usd_used")) else np.nan,
                "mep_venta": float(r.get("mep_venta")) if not pd.isna(r.get("mep_venta")) else np.nan,
                "tasa_cupon_pct": r.get("tasa_cupon_pct"),
                "ytm_pct": (round(float(ytm) * 100, 2) if np.isfinite(ytm) else np.nan),
                "next_cf_date": next_dt,
                "last_cf_date": last_dt,
                "status": status,
            })

    out = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    return out
