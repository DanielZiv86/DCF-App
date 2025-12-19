# bopreales_data.py
# ========= BOPREAL - Cálculo robusto (sin SciPy) =========
#
# Fuente de precios: IOL (pd.read_html) -> filtra tickers ARS y USD (MEP)
# Fuente de cashflows: Excel (BD BOPREALES.xlsx) con flujos en USD
#
# Devuelve una tabla compatible con una vista tipo bonos:
#   Base, Ticker, Mercado, Precio, VarPct, Duration, TIR
#
# Importante:
# - Los cashflows están en USD (moneda de emisión), por lo que la TIR se calcula sobre un "precio en USD".
# - Mercado USD: se usa el precio del ticker USD (ej. BPA7D, BPB8D, etc.).
# - Mercado ARS: se usa el precio ARS convertido a USD con un FX MEP implícito (proxy):
#       FX_MEP ≈ AL30 / AL30D  (fallback GD30 / GD30D)
#       PriceUSD_ARS = Precio_ARS / FX_MEP
#
from __future__ import annotations

import re
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import streamlit as st

from market_cache import market_bucket  # tu helper

# =========================
# Config
# =========================
DEFAULT_BD_PATH = "data/BD BOPREALES.xlsx"
URL_IOL_BONOS = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"

REQUIRED_CF_COLS = ["Ticker", "Fecha", "Principal", "Int", "Cashflow"]


# =========================
# HTTP Session
# =========================
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "OPTIONS"),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=10))
    s.headers.update({"User-Agent": "Mozilla/5.0 (DCF Inversiones)"})
    return s


# =========================
# Cashflows
# =========================
def load_cashflows(path: str = DEFAULT_BD_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_CF_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en BD BOPREALES: {missing}. Tengo: {df.columns.tolist()}")

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")

    for col in ["Principal", "Int", "Cashflow"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["Fecha"]).copy()
    return df


def get_base_tickers_from_bd(path: str = DEFAULT_BD_PATH) -> list[str]:
    df = load_cashflows(path)
    tickers = df["Ticker"].astype(str).str.strip().str.upper().unique().tolist()
    return sorted([t for t in tickers if t])


# =========================
# Regla nemotécnica tickers USD (MEP)
# =========================
def bopreal_ars_to_usd_ticker(base_ars: str) -> str:
    """
    Regla:
    - 2026: BPY26 (ARS) -> BPY6D (USD)
    - 2027/2028: si empieza con "BPO", se elimina la "O" y se agrega "D" final
      ej: BPOA7 -> BPA7D, BPOD7 -> BPD7D, BPOC7 -> BPC7D, BPOA8 -> BPA8D
    """
    t = str(base_ars).strip().upper()

    if t == "BPY26":
        return "BPY6D"

    if t.startswith("BPO") and len(t) >= 4:
        return "BP" + t[3:] + "D"

    return t + "D"


# =========================
# Precios desde IOL
# =========================
def _parse_price_to_float(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"", "-", "–"}:
        return np.nan
    s = re.sub(r"[^\d\.,-]", "", s)
    if not s:
        return np.nan

    # "1.234,56" o "1,234.56"
    if "." in s and "," in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    else:
        if "," in s and "." not in s:
            s = s.replace(".", "").replace(",", ".")

    try:
        return float(s)
    except Exception:
        return np.nan


def _parse_var_pct(v) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace("%", "").replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def fetch_iol_prices_and_var(
    url: str,
    tickers: Iterable[str],
    timeout_s: int = 20,
    adjust_100x: bool = False,
) -> pd.DataFrame:
    """
    Devuelve: Ticker, Price, VarPct.
    Robusto a:
    - múltiples tablas en la página (iteramos todas)
    - tickers con '*' u otros caracteres (normaliza a A-Z0-9)
    - precios en formato 100x (si adjust_100x=True y Price > 500 => /100)
    """
    def _norm_ticker(x: str) -> str:
        x = str(x).strip().upper()
        return re.sub(r"[^A-Z0-9]", "", x)

    tickers_norm = {_norm_ticker(t) for t in tickers if str(t).strip()}
    if not tickers_norm:
        return pd.DataFrame(columns=["Ticker", "Price", "VarPct"])

    s = _session()
    r = s.get(url, timeout=timeout_s)
    r.raise_for_status()

    tables = pd.read_html(r.text)
    if not tables:
        return pd.DataFrame(columns=["Ticker", "Price", "VarPct"])

    frames = []
    for df in tables:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        ticker_col = next((c for c in df.columns if c.lower() in {"símbolo", "simbolo", "ticker", "especie"}), None)
        price_col  = next((c for c in df.columns if c.lower() in {
            "último operado", "ultimo operado",
            "último", "ultimo", "últ.", "ult.",
            "precio", "cierre", "último cierre", "ultimo cierre"
        }), None)
        var_col    = next((c for c in df.columns if c.lower() in {
            "variación diaria", "variacion diaria",
            "variación", "variacion", "var.", "var", "dif"
        }), None)

        if ticker_col is None or price_col is None:
            continue

        cols = [ticker_col, price_col] + ([var_col] if var_col else [])
        tmp = df[cols].copy()
        tmp.columns = ["TickerRaw", "PriceRaw"] + (["VarRaw"] if var_col else [])

        tmp["Ticker"] = tmp["TickerRaw"].map(_norm_ticker)
        tmp = tmp[tmp["Ticker"].isin(tickers_norm)].copy()
        if tmp.empty:
            continue

        tmp["Price"] = tmp["PriceRaw"].apply(_parse_price_to_float)

        if adjust_100x:
            tmp["Price"] = np.where(tmp["Price"] > 500, tmp["Price"] / 100.0, tmp["Price"])

        tmp["VarPct"] = tmp["VarRaw"].apply(_parse_var_pct) if "VarRaw" in tmp.columns else np.nan
        frames.append(tmp[["Ticker", "Price", "VarPct"]])

    if not frames:
        return pd.DataFrame(columns=["Ticker", "Price", "VarPct"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("Ticker").drop_duplicates(subset=["Ticker"], keep="last").reset_index(drop=True)
    return out


# =========================
# FX MEP implícito (proxy)
# =========================
def _pick_ref_pair(ars_map: dict, usd_map: dict, candidates=("AL30", "GD30")) -> Optional[str]:
    for ref in candidates:
        p_ars = ars_map.get(ref, np.nan)
        p_usd = usd_map.get(f"{ref}D", np.nan)
        if np.isfinite(p_ars) and np.isfinite(p_usd) and p_usd != 0:
            return ref
    return None


def _implicit_mep_fx(ars_map: dict, usd_map: dict, ref: str) -> float:
    # FX (ARS por USD) ≈ ref / refD
    p_ars = ars_map.get(ref, np.nan)
    p_usd = usd_map.get(f"{ref}D", np.nan)
    if not (np.isfinite(p_ars) and np.isfinite(p_usd)) or p_usd == 0:
        return np.nan
    return float(p_ars / p_usd)


# =========================
# Root solver (sin SciPy)
# =========================
def _bisect_root(f, a: float, b: float, maxiter: int = 200, tol: float = 1e-10) -> float:
    fa = f(a)
    fb = f(b)
    if np.isnan(fa) or np.isnan(fb):
        return np.nan
    if fa == 0:
        return a
    if fb == 0:
        return b
    if np.sign(fa) == np.sign(fb):
        return np.nan

    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(maxiter):
        mid = (lo + hi) / 2.0
        fmid = f(mid)
        if np.isnan(fmid):
            return np.nan
        if abs(fmid) < tol or (hi - lo) / 2.0 < tol:
            return mid
        if np.sign(fmid) == np.sign(flo):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
    return (lo + hi) / 2.0


# =========================
# XNPV / XIRR (TIR con fechas)
# =========================
def xnpv(rate: float, cashflows: np.ndarray, dates: pd.DatetimeIndex) -> float:
    if rate <= -1.0:
        return np.nan
    dates = pd.to_datetime(dates, errors="coerce")
    if dates.isna().any():
        return np.nan
    t0 = dates[0]
    years = (dates - t0).days / 365.0
    return float(np.sum(cashflows / (1.0 + rate) ** years))


def xirr(
    cashflows: np.ndarray,
    dates: pd.DatetimeIndex,
    guess_low: float = -0.9999,
    guess_high: float = 5.0,
) -> float:
    try:
        def f(r: float) -> float:
            return xnpv(r, cashflows, dates)

        f_low = f(guess_low)
        f_high = f(guess_high)
        if np.isnan(f_low) or np.isnan(f_high):
            return np.nan

        if np.sign(f_low) == np.sign(f_high):
            for high in [10.0, 20.0, 50.0, 100.0]:
                f_high = f(high)
                if np.isnan(f_high):
                    continue
                if np.sign(f_low) != np.sign(f_high):
                    guess_high = high
                    break

        if np.sign(f_low) == np.sign(f_high):
            return np.nan

        return float(_bisect_root(f, guess_low, guess_high, maxiter=250, tol=1e-10))
    except Exception:
        return np.nan


def macaulay_duration_act365(cashflows: np.ndarray, dates_full: pd.DatetimeIndex, y: float) -> float:
    if np.isnan(y) or y <= -1.0:
        return np.nan
    dates_full = pd.to_datetime(dates_full, errors="coerce")
    if dates_full.isna().any():
        return np.nan

    t0 = dates_full[0]
    times = (dates_full - t0).days / 365.0
    if len(times) != (len(cashflows) + 1):
        return np.nan

    disc = (1.0 + y) ** times
    pv = cashflows / disc[1:]
    pv_total = float(np.sum(pv))
    if pv_total <= 0:
        return np.nan

    return float(np.sum(times[1:] * pv) / pv_total)


def modified_duration_from_macaulay(macaulay: float, y: float) -> float:
    if np.isnan(macaulay) or np.isnan(y):
        return np.nan
    return float(macaulay / (1.0 + y))


# =========================
# Métricas por base (con PriceUSD ya definido)
# =========================
def _compute_ytm_and_duration_by_base(
    cf: pd.DataFrame,
    price_usd_by_base: pd.DataFrame,
    valuation_date: pd.Timestamp,
) -> pd.DataFrame:
    df = cf.merge(price_usd_by_base[["Ticker", "PriceUSD"]], on="Ticker", how="left")

    results = []
    for ticker, g in df.groupby("Ticker", sort=True):
        px = g["PriceUSD"].dropna().unique()
        price = float(px[0]) if len(px) else np.nan

        if np.isnan(price):
            results.append({"Ticker": ticker, "PriceUSD": np.nan, "TIR": np.nan, "Modified_Duration": np.nan})
            continue

        future = g[g["Fecha"] > valuation_date].copy().sort_values("Fecha")
        future = future.dropna(subset=["Fecha", "Cashflow"])

        if future.empty:
            results.append({"Ticker": ticker, "PriceUSD": price, "TIR": np.nan, "Modified_Duration": np.nan})
            continue

        dates_full = pd.to_datetime([valuation_date] + future["Fecha"].tolist(), errors="coerce")
        cfs_full = np.concatenate((np.array([-price], dtype=float), future["Cashflow"].astype(float).to_numpy()))

        tir = xirr(cfs_full, dates_full)

        if not np.isnan(tir):
            dur_mac = macaulay_duration_act365(
                cashflows=future["Cashflow"].astype(float).to_numpy(),
                dates_full=dates_full,
                y=tir
            )
            dur_mod = modified_duration_from_macaulay(dur_mac, tir)
        else:
            dur_mod = np.nan

        results.append({"Ticker": ticker, "PriceUSD": price, "TIR": tir, "Modified_Duration": dur_mod})

    return pd.DataFrame(results).sort_values("Ticker").reset_index(drop=True)


# =========================
# Implementación "real" (sin cache)
# =========================
def _get_multi_table_impl(
    bd_path: str = DEFAULT_BD_PATH,
    today: Optional[date] = None,
) -> pd.DataFrame:
    """
    Devuelve tabla multi-mercado para BOPREAL:
      Base, Ticker, Mercado, Precio, VarPct, Duration, TIR

    - Cashflows vienen en USD (BD).
    - Mercado USD: PriceUSD = precio ticker USD (D).
    - Mercado ARS: PriceUSD = precio ARS / FX_MEP (proxy AL30/AL30D o GD30/GD30D).
    """
    if today is None:
        today = date.today()
    valuation_date = pd.Timestamp(today)

    cf = load_cashflows(bd_path)
    base_tickers = get_base_tickers_from_bd(bd_path)

    ars_tks = base_tickers
    usd_tks = [bopreal_ars_to_usd_ticker(t) for t in base_tickers]

    prices_ars = fetch_iol_prices_and_var(URL_IOL_BONOS, ars_tks, adjust_100x=False)
    prices_usd = fetch_iol_prices_and_var(URL_IOL_BONOS, usd_tks, adjust_100x=True)

    ars_map = dict(zip(prices_ars["Ticker"], prices_ars["Price"]))
    ars_var = dict(zip(prices_ars["Ticker"], prices_ars["VarPct"]))

    usd_map = dict(zip(prices_usd["Ticker"], prices_usd["Price"]))
    usd_var = dict(zip(prices_usd["Ticker"], prices_usd["VarPct"]))

    # --- FX MEP implícito (proxy) usando AL30/AL30D o GD30/GD30D ---
    ref_ars_df = fetch_iol_prices_and_var(URL_IOL_BONOS, ["AL30", "GD30"], adjust_100x=False)
    ref_usd_df = fetch_iol_prices_and_var(URL_IOL_BONOS, ["AL30D", "GD30D"], adjust_100x=True)
    ref_ars_map = dict(zip(ref_ars_df["Ticker"], ref_ars_df["Price"]))
    ref_usd_map = dict(zip(ref_usd_df["Ticker"], ref_usd_df["Price"]))

    ref = _pick_ref_pair(ref_ars_map, ref_usd_map, candidates=("AL30", "GD30"))
    fx_mep = _implicit_mep_fx(ref_ars_map, ref_usd_map, ref) if ref else np.nan  # ARS por USD

    # PriceUSD por base para ARS y USD
    price_usd_by_base_ars = []
    price_usd_by_base_usd = []
    for base in base_tickers:
        tk_usd = bopreal_ars_to_usd_ticker(base)

        p_ars = ars_map.get(base, np.nan)
        p_usd = usd_map.get(tk_usd, np.nan)

        p_ars_usd = (p_ars / fx_mep) if (np.isfinite(p_ars) and np.isfinite(fx_mep) and fx_mep != 0) else np.nan

        price_usd_by_base_ars.append({"Ticker": base, "PriceUSD": p_ars_usd})
        price_usd_by_base_usd.append({"Ticker": base, "PriceUSD": p_usd})

    metrics_ars = _compute_ytm_and_duration_by_base(cf, pd.DataFrame(price_usd_by_base_ars), valuation_date)
    metrics_ars["Mercado"] = "ARS"
    metrics_usd = _compute_ytm_and_duration_by_base(cf, pd.DataFrame(price_usd_by_base_usd), valuation_date)
    metrics_usd["Mercado"] = "USD"
    metrics_all = pd.concat([metrics_ars, metrics_usd], ignore_index=True)

    rows = []
    for base in base_tickers:
        tk_ars = base
        tk_usd = bopreal_ars_to_usd_ticker(base)

        m_ars = metrics_all[(metrics_all["Ticker"] == base) & (metrics_all["Mercado"] == "ARS")]
        tir_ars = float(m_ars["TIR"].iloc[0]) if len(m_ars) else np.nan
        dur_ars = float(m_ars["Modified_Duration"].iloc[0]) if len(m_ars) else np.nan

        m_usd = metrics_all[(metrics_all["Ticker"] == base) & (metrics_all["Mercado"] == "USD")]
        tir_usd = float(m_usd["TIR"].iloc[0]) if len(m_usd) else np.nan
        dur_usd = float(m_usd["Modified_Duration"].iloc[0]) if len(m_usd) else np.nan

        rows.append({
            "Base": base,
            "Ticker": tk_ars,
            "Mercado": "ARS",
            "Precio": ars_map.get(tk_ars, np.nan),
            "VarPct": ars_var.get(tk_ars, np.nan),
            "Duration": dur_ars,
            "TIR": tir_ars,
        })

        rows.append({
            "Base": base,
            "Ticker": tk_usd,
            "Mercado": "USD",
            "Precio": usd_map.get(tk_usd, np.nan),
            "VarPct": usd_var.get(tk_usd, np.nan),
            "Duration": dur_usd,
            "TIR": tir_usd,
        })

    return pd.DataFrame(rows)


# =========================
# Cache wrapper (Streamlit Cloud)
# =========================
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _get_multi_table_cached(bucket: str, bd_path: str, today: Optional[date]) -> pd.DataFrame:
    # bucket solo sirve para invalidar cada 20 min dentro de horario
    return _get_multi_table_impl(bd_path=bd_path, today=today)

def get_multi_table(bd_path: str = DEFAULT_BD_PATH, today: Optional[date] = None) -> pd.DataFrame:
    bucket = market_bucket()
    return _get_multi_table_cached(bucket, bd_path, today)
