# bonistas_hd_data.py
# ========= Bonos HD (Hard Dollar) - Cálculo robusto sin webscraping de bonistas =========
#
# Fuente de precios: IOL (pd.read_html) -> filtra tickers base / D / C
# Fuente de cashflows: Excel (BD BONOS HD.xlsx) con flujos futuros en USD
#
# Devuelve una tabla multi-mercado compatible con la vista `views/bonos_hd.py`:
#   Base, Ticker, Mercado, Precio, Duration, TIR
#
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import requests
from requests.adapters import HTTPAdapter, Retry

# yfinance es opcional (fallback)
try:
    import yfinance as yf
except Exception:
    yf = None


# =========================
# Config
# =========================

DEFAULT_BD_PATH = "data/BD BONOS HD.xlsx"
URL_IOL_BONOS = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"

MERCADOS = [
    ("PESOS", ""),   # AL30
    ("MEP", "D"),    # AL30D
    ("CCL", "C"),    # AL30C
]

REQUIRED_CF_COLS = ["Ticker", "Fecha", "Principal", "Int", "Cashflow"]


# =========================
# Requests session
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
# Carga de cashflows
# =========================

def load_cashflows(path: str = DEFAULT_BD_PATH) -> pd.DataFrame:
    """
    Lee un Excel con cashflows por bono.

    Columnas requeridas:
      - Ticker: ej 'AL30' (SIN sufijo D/C)
      - Fecha: fecha del flujo
      - Principal, Int, Cashflow: numéricos (Cashflow puede ser Principal+Int)
    """
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_CF_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en BD: {missing}. Tengo: {df.columns.tolist()}")

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")

    for col in ["Principal", "Int", "Cashflow"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["Fecha"]).copy()
    return df


def get_base_tickers_from_bd(path: str = DEFAULT_BD_PATH) -> list[str]:
    df = load_cashflows(path)
    tickers = (
        df["Ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )
    tickers = sorted([t for t in tickers if t])
    return tickers


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

    # Caso "1.234,56" o "1,234.56"
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


def fetch_iol_bonds_prices(url: str, tickers: Iterable[str], timeout_s: int = 20) -> pd.DataFrame:
    s = _session()
    r = s.get(url, timeout=timeout_s)
    r.raise_for_status()

    tables = pd.read_html(r.text)
    if not tables:
        raise RuntimeError("No pude leer tablas con pd.read_html desde IOL.")

    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    ticker_col = next((c for c in df.columns if c.lower() in {"símbolo", "simbolo", "ticker", "especie"}), df.columns[0])
    price_col  = next((c for c in df.columns if c.lower() in {"último", "ultimo", "últ.", "ult.", "precio", "cierre"}), df.columns[1])

    out = df[[ticker_col, price_col]].rename(columns={ticker_col: "Ticker", price_col: "PriceRaw"}).copy()
    out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()

    tickers_set = {str(t).strip().upper() for t in tickers}
    out = out[out["Ticker"].isin(tickers_set)].copy()

    out["Price"] = out["PriceRaw"].apply(_parse_price_to_float)

    # regla práctica: si viene “*100”
    out["Price"] = np.where(out["Price"] > 500, out["Price"] / 100.0, out["Price"])
    out = out.dropna(subset=["Price"]).drop_duplicates(subset=["Ticker"], keep="last")

    return out[["Ticker", "Price"]].sort_values("Ticker").reset_index(drop=True)


def _to_yf_ticker(tk: str) -> str:
    tk = str(tk).strip().upper()
    return tk if tk.endswith(".BA") else f"{tk}.BA"


def fetch_yf_prices(tickers: Iterable[str], period: str = "5d") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame(columns=["Ticker", "Price"])

    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "Price"])

    yf_tickers = [_to_yf_ticker(t) for t in tickers]

    data = yf.download(
        yf_tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    rows = []
    for base_tk, yf_tk in zip(tickers, yf_tickers):
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ("Close", yf_tk) in data.columns:
                    close = data[("Close", yf_tk)].dropna()
                elif (yf_tk, "Close") in data.columns:
                    close = data[(yf_tk, "Close")].dropna()
                else:
                    close = pd.Series(dtype=float)
            else:
                close = data["Close"].dropna() if "Close" in data.columns else pd.Series(dtype=float)

            px = float(close.iloc[-1]) if len(close) else np.nan
        except Exception:
            px = np.nan

        if not np.isnan(px) and px > 500:
            px = px / 100.0

        rows.append({"Ticker": base_tk, "Price": px})

    out = pd.DataFrame(rows).dropna(subset=["Price"])
    return out


# =========================
# XNPV / XIRR (sin SciPy)
# =========================

def xnpv(rate: float, cashflows: np.ndarray, dates: pd.DatetimeIndex) -> float:
    """NPV con ACT/365 desde la primera fecha."""
    if rate <= -1.0:
        return np.nan
    dates = pd.to_datetime(dates, errors="coerce")
    if dates.isna().any():
        return np.nan

    t0 = dates[0]
    years = (dates - t0).days / 365.0
    return float(np.sum(cashflows / (1.0 + rate) ** years))


def _find_bracket_for_root(func, low: float, high: float, expansions: list[float]) -> Optional[tuple[float, float]]:
    """
    Intenta encontrar un intervalo [a,b] donde func(a) y func(b) tengan signos opuestos.
    """
    try:
        f_low = func(low)
        f_high = func(high)
        if np.isnan(f_low) or np.isnan(f_high):
            return None
        if np.sign(f_low) != np.sign(f_high):
            return (low, high)

        for h in expansions:
            f_h = func(h)
            if np.isnan(f_h):
                continue
            if np.sign(f_low) != np.sign(f_h):
                return (low, h)
        return None
    except Exception:
        return None


def _bisect_root(func, a: float, b: float, tol: float = 1e-9, maxiter: int = 200) -> float:
    fa = func(a)
    fb = func(b)
    if np.isnan(fa) or np.isnan(fb):
        return np.nan
    if np.sign(fa) == np.sign(fb):
        return np.nan

    for _ in range(maxiter):
        m = (a + b) / 2.0
        fm = func(m)
        if np.isnan(fm):
            return np.nan

        if abs(fm) < tol:
            return float(m)

        # conserva el sub-intervalo con cambio de signo
        if np.sign(fa) != np.sign(fm):
            b, fb = m, fm
        else:
            a, fa = m, fm

        if abs(b - a) < tol:
            return float((a + b) / 2.0)

    return float((a + b) / 2.0)


def xirr(cashflows: np.ndarray, dates: pd.DatetimeIndex, guess_low: float = -0.9999, guess_high: float = 5.0) -> float:
    """
    Resuelve XIRR por bisección (robusto, sin SciPy).
    """
    dates = pd.to_datetime(dates, errors="coerce")
    if dates.isna().any():
        return np.nan

    def f(r: float) -> float:
        return xnpv(r, cashflows, dates)

    bracket = _find_bracket_for_root(f, guess_low, guess_high, expansions=[10.0, 20.0, 50.0, 100.0])
    if bracket is None:
        return np.nan

    a, b = bracket
    return _bisect_root(f, a, b, tol=1e-10, maxiter=250)


# =========================
# Duration (Macaulay / Modified)
# =========================

def macaulay_duration_act365(cashflows: np.ndarray, dates_full: pd.DatetimeIndex, y: float) -> float:
    """
    Macaulay duration en años (ACT/365).
    `dates_full` incluye valuation_date en el primer elemento, luego fechas de cada cashflow futuro.
    `cashflows` = cashflows futuros (sin flujo t0).
    """
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
# Métricas por mercado
# =========================

def compute_ytm_and_duration(
    cf: pd.DataFrame,
    prices_by_ticker: pd.DataFrame,
    valuation_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Calcula TIR y Modified Duration por ticker base (sin D/C),
    usando cashflows en USD y un precio USD por ticker.
    """
    df = cf.merge(prices_by_ticker[["Ticker", "PriceUSD"]], on="Ticker", how="left")

    results = []
    for ticker, g in df.groupby("Ticker", sort=True):
        px = g["PriceUSD"].dropna().unique()
        price = float(px[0]) if len(px) else np.nan

        if np.isnan(price):
            results.append({"Ticker": ticker, "PriceUSD": np.nan, "TIR": np.nan, "Modified_Duration": np.nan, "note": "Sin precio USD"})
            continue

        future = g[g["Fecha"] > valuation_date].copy().sort_values("Fecha")
        future = future.dropna(subset=["Fecha", "Cashflow"])

        if future.empty:
            results.append({"Ticker": ticker, "PriceUSD": price, "TIR": np.nan, "Modified_Duration": np.nan, "note": "Sin cashflows futuros"})
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

        results.append({
            "Ticker": ticker,
            "PriceUSD": price,
            "TIR": tir,
            "Modified_Duration": dur_mod,
            "note": "" if not np.isnan(tir) else "No converge / revisar cashflows o precio",
        })

    return pd.DataFrame(results).sort_values("Ticker").reset_index(drop=True)


def _build_prices_multi(base_tickers: list[str]) -> pd.DataFrame:
    tickers_all = []
    for base in base_tickers:
        for _, suf in MERCADOS:
            tickers_all.append(f"{base}{suf}")

    prices = fetch_iol_bonds_prices(URL_IOL_BONOS, tickers_all)
    prices_map = dict(zip(prices["Ticker"], prices["Price"]))

    # Fallback yfinance para faltantes
    missing = [t for t in tickers_all if (t not in prices_map) or pd.isna(prices_map.get(t))]
    if missing:
        yf_prices = fetch_yf_prices(missing)
        if not yf_prices.empty:
            prices_map.update(dict(zip(yf_prices["Ticker"], yf_prices["Price"])))

    rows = []
    for base in base_tickers:
        price_p = prices_map.get(f"{base}", np.nan)
        price_d = prices_map.get(f"{base}D", np.nan)
        price_c = prices_map.get(f"{base}C", np.nan)

        usd_for_tir = price_d if np.isfinite(price_d) else (price_c if np.isfinite(price_c) else np.nan)

        for mercado, suf in MERCADOS:
            tk = f"{base}{suf}"
            if mercado == "PESOS":
                shown = price_p if np.isfinite(price_p) else usd_for_tir
            elif mercado == "MEP":
                shown = price_d if np.isfinite(price_d) else usd_for_tir
            else:
                shown = price_c if np.isfinite(price_c) else usd_for_tir

            rows.append({"Base": base, "Ticker": tk, "Mercado": mercado, "PrecioUSD": shown})

    return pd.DataFrame(rows)


def scrape_bonistas_multi_mercado(
    base_tickers: list[str] | None = None,
    today: date | None = None,
    bd_path: str = DEFAULT_BD_PATH,
) -> pd.DataFrame:
    """
    Compatibilidad con el nombre anterior: ya no scrapea bonistas.
    Calcula TIR y Duration desde cashflows + precios IOL.

    Columnas:
      Base, Ticker, Mercado, Precio, Duration, TIR
    """
    if base_tickers is None:
        base_tickers = get_base_tickers_from_bd(bd_path)

    valuation_date = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()

    cf = load_cashflows(bd_path)
    cf = cf[cf["Ticker"].isin([t.upper() for t in base_tickers])].copy()

    prices_multi = _build_prices_multi([t.upper() for t in base_tickers])

    # arma tabla de precio USD por ticker base: priorizamos MEP (D)
    prices_usd = (
        prices_multi[prices_multi["Mercado"] == "MEP"][["Base", "PrecioUSD"]]
        .rename(columns={"Base": "Ticker", "PrecioUSD": "PriceUSD"})
    )

    metrics = compute_ytm_and_duration(cf, prices_usd, valuation_date)

    out = prices_multi.merge(
        metrics[["Ticker", "PriceUSD", "TIR", "Modified_Duration"]],
        left_on="Base",
        right_on="Ticker",
        how="left",
    )
    out = out.drop(columns=["Ticker_y"]).rename(columns={"Ticker_x": "Ticker"})
    out = out.rename(columns={"PrecioUSD": "Precio", "Modified_Duration": "Duration"})

    out = out[["Base", "Ticker", "Mercado", "Precio", "Duration", "TIR"]].sort_values(["Base", "Mercado"]).reset_index(drop=True)
    return out


def get_multi_table(
    mercado: str | None = None,
    today: date | None = None,
    bd_path: str = DEFAULT_BD_PATH,
) -> pd.DataFrame:
    base_tickers = get_base_tickers_from_bd(bd_path)
    df = scrape_bonistas_multi_mercado(base_tickers, today=today, bd_path=bd_path)
    if mercado is not None:
        mercado = mercado.upper()
        df = df[df["Mercado"] == mercado]
    return df


def get_hd_table(today: date | None = None, bd_path: str = DEFAULT_BD_PATH) -> pd.DataFrame:
    df_mep = get_multi_table("MEP", today=today, bd_path=bd_path).copy()
    df_mep = df_mep.set_index("Ticker")[["Precio", "Duration", "TIR"]]
    return df_mep
