# letras_boncaps_data.py
import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter, Retry
from matplotlib import ticker as mtick
import plotly.graph_objects as go

# ===================== 0) Config =====================

# Base del módulo (carpeta donde está este .py)
BASE_DIR = Path(__file__).resolve().parent

# Ruta al Excel, relativa al módulo
XLSX_PATH = BASE_DIR / "data" / "Letras_Activas.xlsx"

# Cache también relativa al módulo (no al cwd)
CACHE_DIR = BASE_DIR / ".cache_data912"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

URL_MEP   = "https://dolarapi.com/v1/dolares/bolsa"
URL_NOTES = "https://data912.com/live/arg_notes"
URL_BONDS = "https://data912.com/live/arg_bonds"

TIMEOUT_S = 12
RETRIES   = 3
BACKOFF   = 1.5

# Paleta DCF (solo para el gráfico)
DCF_COLORS = {
    "bg_header":  "#053D57",
    "bg_alt":     "#F3F7F9",
    "text_header":"#FFFFFF",
    "text_body":  "#0B2239",
    "border":     "#C5D2D8",
    "accent":     "#0E6881",
    "muted":      "#90B0BC",
}

# ===================== Utils HTTP con retry + cache =====================
def build_session(retries:int = RETRIES, backoff:float = BACKOFF) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries, connect=retries, read=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "OPTIONS"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "DCF-Inversiones/1.0 (+script carry)"})
    return s


def fetch_json_with_cache(session: requests.Session, url:str, cache_file:Path, timeout:int = TIMEOUT_S):
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        cache_file.write_text(json.dumps(data, ensure_ascii=False))
        return data, "live"
    except Exception as e:
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            print(f"⚠️  Aviso: usando cache para {url} por error de red: {e}")
            return data, "cache"
        raise


# ===================== Core: cálculo de letras y boncaps =====================

def get_letras_carry(
    xlsx_path: Path | str = XLSX_PATH,
    cache_dir: Path = CACHE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Devuelve:
      - carry_view: tabla resumen para mostrar (Ticker, Precio, Días, TNA, TEA, TEM, MEP BE, Banda Sup)
      - carry: dataframe completo con columnas para el gráfico (days_to_exp, finish_worst, finish_better, MEP_BREAKEVEN)
      - mep: tipo de cambio MEP utilizado
    """
    xlsx_path = Path(xlsx_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Cargar Excel
    df_xls = pd.read_excel(
        xlsx_path,
        usecols=["Ticker", "Fecha Vencimiento", "Valor Final"],
        decimal=",",
    )
    df_xls.columns = [c.strip() for c in df_xls.columns]
    df_xls["Ticker"] = df_xls["Ticker"].astype(str).str.strip()
    df_xls["Fecha Vencimiento"] = pd.to_datetime(df_xls["Fecha Vencimiento"], dayfirst=True, errors="coerce")
    df_xls["Valor Final"] = pd.to_numeric(df_xls["Valor Final"], errors="coerce")

    df_xls = df_xls.dropna(subset=["Ticker", "Fecha Vencimiento", "Valor Final"])
    df_xls = (
        df_xls.sort_values(["Ticker", "Fecha Vencimiento"])
              .drop_duplicates(subset=["Ticker"], keep="last")
    )

    tickers = dict(zip(df_xls["Ticker"], df_xls["Fecha Vencimiento"].dt.date))
    payoff  = dict(zip(df_xls["Ticker"], df_xls["Valor Final"].astype(float)))

    # 2) Datos de mercado
    session = build_session()

    # MEP
    resp = session.get(URL_MEP, timeout=TIMEOUT_S)
    resp.raise_for_status()
    mep = float(resp.json()["venta"])

    # Data912 (con cache)
    notes, src_notes = fetch_json_with_cache(session, URL_NOTES, cache_dir / "arg_notes.json", timeout=TIMEOUT_S)
    bonds, src_bonds = fetch_json_with_cache(session, URL_BONDS, cache_dir / "arg_bonds.json", timeout=TIMEOUT_S)

    if src_notes == "cache" or src_bonds == "cache":
        print("ℹ️  Fuente data912 parcial/totalmente desde cache (servicio lento o intermitente).")

    df = pd.DataFrame(notes + bonds)

    # 3) Cartera filtrada
    carry = df.loc[df.symbol.isin(tickers.keys())].copy().set_index("symbol")

    for col in ("c", "px_bid", "px_ask"):
        if col not in carry.columns:
            carry[col] = pd.NA

    px_c   = pd.to_numeric(carry["c"], errors="coerce")
    px_bid = pd.to_numeric(carry["px_bid"], errors="coerce")
    px_ask = pd.to_numeric(carry["px_ask"], errors="coerce")

    carry["bond_price"] = px_c.round(2)
    carry["payoff"]     = carry.index.map(payoff)
    carry["expiration"] = carry.index.map(tickers)

    today = date.today()
    carry["days_to_exp"] = (pd.to_datetime(carry["expiration"]) - pd.Timestamp(today)).dt.days.clip(lower=0)

    # 4) Tasas
    valid_days = carry["days_to_exp"].astype("float")
    valid_days = valid_days.mask(valid_days <= 0, np.nan)

    ratio_c = carry["payoff"] / px_c
    carry["tna"] = ((ratio_c - 1) / valid_days * 365).replace([np.inf, -np.inf], np.nan)
    carry["tea"] = (ratio_c ** (365 / valid_days) - 1).replace([np.inf, -np.inf], np.nan)
    carry["tem"] = (ratio_c ** (1 / (valid_days / 30)) - 1).replace([np.inf, -np.inf], np.nan)

    carry["tem_bid"] = (carry["payoff"] / px_bid) ** (1 / (valid_days / 30)) - 1
    carry["tem_ask"] = (carry["payoff"] / px_ask) ** (1 / (valid_days / 30)) - 1

    # 5) Banda MLC continua
    ANCHOR = date(2025, 4, 11)
    days_since_anchor = (pd.to_datetime(carry["expiration"]) - pd.Timestamp(ANCHOR)).dt.days.clip(lower=0)
    months_cont = days_since_anchor / 30.0

    finish_worst_float  = 1400 * (1.01 ** months_cont)  # techo
    finish_better_float = 1000 * (0.99 ** months_cont)  # piso

    carry["finish_worst"]  = finish_worst_float.round().astype("Int64")
    carry["finish_better"] = finish_better_float.round().astype("Int64")

    # 5.b) Carry en escenarios de tipo de cambio MEP
    for price in [1000, 1100, 1200, 1300, 1400]:
        carry[f"carry_{price}"] = (carry["payoff"] / px_c) * (mep / price) - 1

    carry["carry_worst"] = (carry["payoff"] / px_c) * (mep / finish_worst_float) - 1

    # 6) MEP Breakeven
    carry["MEP_BREAKEVEN"] = (mep * (carry["payoff"] / px_c)).round(0)

    # Ordenar por días
    carry = carry.sort_values("days_to_exp")
    # 7) Vista de tabla para mostrar
    ordered_cols = [
        "symbol",         # -> Ticker
        "bond_price",     # -> Precio
        "days_to_exp",    # -> Días A Venc.
        "tna",            # -> TNA
        "tea",            # -> TEA
        "tem",            # -> TEM
        "MEP_BREAKEVEN",  # -> MEP BE
        "finish_worst",   # -> $ Banda Sup
    ]

    _tmp = carry.reset_index()
    cols_available = [c for c in ordered_cols if c in _tmp.columns]

    carry_view = (
        _tmp[cols_available]
        .rename(columns={
            "symbol": "Ticker",
            "bond_price": "Precio",
            "days_to_exp": "Dias A Venc.",
            "tna": "TNA",
            "tea": "TEA",
            "tem": "TEM",
            "MEP_BREAKEVEN": "MEP BE",
            "finish_worst": "$ Banda Sup",
        })
        .sort_values("Dias A Venc.")
    )

    return carry_view, carry, mep


# ===================== Gráfico de bandas =====================


def build_letras_bands_figure_plotly(carry: pd.DataFrame) -> go.Figure:
    """
    Gráfico de bandas de Carry-Trade en Plotly, estilo dark DCF.
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
        line=dict(color="#4EA5B5", width=2),
    ))

    # Banda Inferior (piso)
    fig.add_trace(go.Scatter(
        x=df["days_to_exp"],
        y=df["finish_better"],
        mode="lines",
        name="$ Banda Inf",
        line=dict(color="#89C0CC", width=2, dash="dash"),
    ))

    # MEP Breakeven (puntos + ticker)
    fig.add_trace(go.Scatter(
        x=df["days_to_exp"],
        y=df["MEP_BREAKEVEN"],
        mode="markers+text",
        text=df.index,               # el índice es el ticker (symbol)
        textposition="top center",
        name="MEP Breakeven",
        marker=dict(size=8, color="#0E6881"),
    ))

    # Layout dark consistente con la app (sin usar el template global)
    fig.update_layout(
        title="Carry-Trade – Líneas de Banda",
        xaxis_title="Días al vencimiento",
        yaxis_title="Precio proyectado ($)",
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        font=dict(color="#E3E6E9", family="Montserrat, Arial, sans-serif"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#333333",
            color="#E3E6E9",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#333333",
            color="#E3E6E9",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E3E6E9")
        ),
        height=550,
        margin=dict(l=10, r=10, t=50, b=20),
    )

    return fig
