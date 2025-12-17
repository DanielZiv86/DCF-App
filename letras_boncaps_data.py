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
import streamlit as st
from market_cache import market_bucket, record_data_timestamp

# ===================== 0) Config =====================

# Carpeta donde está este archivo .py
BASE_DIR = Path(__file__).resolve().parent

# Ruta fija al Excel (tal como está en tu disco)
XLSX_PATH = BASE_DIR / "data" / "Letras Activas.xlsx"

# Cache relativa al módulo
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

# ===================== Google Sheet BONOS / LETRAS =====================

GOOGLE_SHEET_ID = "1AJKhrMq_lDHRRbGQ5UNkIJSmkPsBi3ekFkVODjIarIg"

def _sheet_csv_url(sheet_name: str) -> str:
    """
    Devuelve la URL CSV pública para una hoja de Google Sheets.
    """
    return (
        f"https://docs.google.com/spreadsheets/d/"
        f"{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    )


def load_bonos_letras_prices() -> pd.DataFrame:
    """
    Lee las hojas BONOS y LETRAS del Google Sheet y devuelve
    un DataFrame indexado por Ticker con la última cotización (columna Last).

    Output: DataFrame con índice = Ticker y columna 'Last' (float).
    """
    sheets = ["BONOS", "LETRAS"]
    frames = []

    for sheet in sheets:
        url = _sheet_csv_url(sheet)
        try:
            df_raw = pd.read_csv(url)
        except Exception as e:
            print(f"⚠️ No pude leer la hoja {sheet} desde Google Sheets: {e}")
            continue

        if df_raw.empty:
            continue

        # Suponemos que la primera columna es el ticker
        first_col = df_raw.columns[0]
        df = df_raw.copy()
        df = df[df[first_col].notna()]  # saco filas vacías
        df = df.rename(columns={first_col: "Ticker"})

        # En este dataset de Rudolph, la estructura típica es:
        # Ticker | Bid Size | Bid | Ask | Ask Size | Last | Close | ...
        # => tomamos la 6ta columna (índice 5) como precio 'Last'
        if df.shape[1] <= 5:
            print(f"⚠️ La hoja {sheet} no tiene suficientes columnas para extraer precio.")
            continue

        price_col = df.columns[5]  # columna Last
        df = df[["Ticker", price_col]].rename(columns={price_col: "Last"})

        df["Ticker"] = df["Ticker"].astype(str).str.strip()

        # Números tipo 113,890.00  ->  113890.00
        df["Last"] = (
            df["Last"]
            .astype(str)
            .str.replace(",", "", regex=False)  # saco separador de miles
        )
        df["Last"] = pd.to_numeric(df["Last"], errors="coerce")

        df = df.dropna(subset=["Last"])
        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No pude cargar ninguna hoja de BONOS/LETRAS desde Google Sheets."
        )

    all_df = pd.concat(frames, ignore_index=True)
    # Si el mismo ticker aparece en varias hojas, me quedo con la primera
    all_df = all_df.drop_duplicates(subset=["Ticker"], keep="first")

    return all_df.set_index("Ticker")


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

    Los precios de mercado vienen del Google Sheet (hojas BONOS y LETRAS).
    El Excel local define: Ticker, Fecha Vencimiento, Valor Final (payoff).
    """
    xlsx_path = Path(xlsx_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de Letras.\n"
            f"Ruta esperada: {xlsx_path}"
        )

    # 1) Cargar Excel de letras activas
    df_xls = pd.read_excel(
        xlsx_path,
        usecols=["Ticker", "Fecha Vencimiento", "Valor Final"],
        decimal=",",
    )
    df_xls.columns = [c.strip() for c in df_xls.columns]
    df_xls["Ticker"] = df_xls["Ticker"].astype(str).str.strip()
    df_xls["Fecha Vencimiento"] = pd.to_datetime(
        df_xls["Fecha Vencimiento"], dayfirst=True, errors="coerce"
    )
    df_xls["Valor Final"] = pd.to_numeric(df_xls["Valor Final"], errors="coerce")

    df_xls = df_xls.dropna(subset=["Ticker", "Fecha Vencimiento", "Valor Final"])
    df_xls = (
        df_xls.sort_values(["Ticker", "Fecha Vencimiento"])
              .drop_duplicates(subset=["Ticker"], keep="last")
    )

    tickers = dict(zip(df_xls["Ticker"], df_xls["Fecha Vencimiento"].dt.date))
    payoff  = dict(zip(df_xls["Ticker"], df_xls["Valor Final"].astype(float)))

    # 2) Traer precios desde Google Sheets (BONOS + LETRAS)
    prices_df = load_bonos_letras_prices()   # índice = Ticker, col = Last

    # Armo dataframe base 'carry' indexado por ticker
    carry = pd.DataFrame(index=sorted(tickers.keys()))
    carry.index.name = "symbol"

    carry["bond_price"] = prices_df["Last"].reindex(carry.index)

    # Aviso si algún ticker del Excel no tiene precio en el Sheet
    missing_px = carry["bond_price"].isna()
    if missing_px.any():
        faltan = carry.index[missing_px].tolist()
        print("⚠️ Tickers sin precio en Google Sheets:", faltan)

    carry["payoff"] = pd.Series(payoff)
    carry["expiration"] = pd.to_datetime(
        pd.Series(tickers, name="expiration")
    )

    # 3) Tipo de cambio MEP (igual que antes)
    session = build_session()
    resp = session.get(URL_MEP, timeout=TIMEOUT_S)
    resp.raise_for_status()
    mep = float(resp.json()["venta"])

    # 4) Días al vencimiento
    today = date.today()
    carry["days_to_exp"] = (
        (carry["expiration"] - pd.Timestamp(today)).dt.days.clip(lower=0)
    )

    # 5) Tasas (TNA / TEA / TEM) en base a bond_price y payoff
    valid_days = carry["days_to_exp"].astype("float")
    valid_days = valid_days.mask(valid_days <= 0, np.nan)

    px_c = pd.to_numeric(carry["bond_price"], errors="coerce")

    ratio_c = carry["payoff"] / px_c
    carry["tna"] = ((ratio_c - 1) / valid_days * 365).replace([np.inf, -np.inf], np.nan)
    carry["tea"] = (ratio_c ** (365 / valid_days) - 1).replace([np.inf, -np.inf], np.nan)
    carry["tem"] = (ratio_c ** (1 / (valid_days / 30)) - 1).replace([np.inf, -np.inf], np.nan)

    # 6) Banda MLC continua (igual que antes)
    ANCHOR = date(2025, 4, 11)
    days_since_anchor = (
        (carry["expiration"] - pd.Timestamp(ANCHOR)).dt.days.clip(lower=0)
    )
    months_cont = days_since_anchor / 30.0

    finish_worst_float  = 1400 * (1.01 ** months_cont)  # techo
    finish_better_float = 1000 * (0.99 ** months_cont)  # piso

    carry["finish_worst"]  = finish_worst_float.round().astype("Int64")
    carry["finish_better"] = finish_better_float.round().astype("Int64")

    # 7) Carry en escenarios de tipo de cambio MEP (usando px_c)
    for price in [1000, 1100, 1200, 1300, 1400]:
        carry[f"carry_{price}"] = (carry["payoff"] / px_c) * (mep / price) - 1

    carry["carry_worst"] = (carry["payoff"] / px_c) * (mep / finish_worst_float) - 1

    # 8) MEP Breakeven
    carry["MEP_BREAKEVEN"] = (mep * (carry["payoff"] / px_c)).round(0)

    # Ordenar por días
    carry = carry.sort_values("days_to_exp")

    # 9) Vista resumida para la tabla de Streamlit
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


# =========================
# Cache wrappers (Streamlit Cloud)
# - Cache bucket changes every 20m between 11:01-18:00 AR
# - Outside market hours, bucket stays fixed, so no refetch
# =========================
def _dcf_cache_bucket() -> str:
    try:
        return market_bucket()
    except Exception:
        return 'no-bucket'

_dcf_get_letras_carry__impl = get_letras_carry
@st.cache_data(ttl=60*60*24, show_spinner=False)
def _dcf_get_letras_carry__cached(bucket: str, *args, **kwargs):
    return _dcf_get_letras_carry__impl(*args, **kwargs)
def get_letras_carry(*args, **kwargs):
    bucket = _dcf_cache_bucket()
    return _dcf_get_letras_carry__cached(bucket, *args, **kwargs)

"""_dcf_get_boncaps_curve__impl = get_boncaps_curve
@st.cache_data(ttl=60*60*24, show_spinner=False)
def _dcf_get_boncaps_curve__cached(bucket: str, *args, **kwargs):
    return _dcf_get_boncaps_curve__impl(*args, **kwargs)"""
def get_boncaps_curve(*args, **kwargs):
    """
    Alias de compatibilidad: si en algún lugar del código quedó el nombre viejo
    'get_boncaps_curve', lo redirigimos al flujo actual.
    """
    return get_letras_carry(*args, **kwargs)
