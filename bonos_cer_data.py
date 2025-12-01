# bonos_cer_data.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# =========================================
# Config
# =========================================

BONISTAS_URL_TEMPLATE = "https://bonistas.com/bono-cotizacion-rendimiento-precio-hoy/{ticker}"

TIMEOUT_S = 12
RETRIES = 3
BACKOFF = 1.5

# Lista de bonos ajustables CER
CER_TICKERS: List[str] = [
    "TZXD5",
    "TZXM6",
    "TZX26",
    "TX26",
    "TZXO6",
    "TZXD6",
    "TZXM7",
    "TZX27",
    "TX28",
    "TZXD7",
    "TZX28",
    "DIP0",
    "DICP",
    "TX31",
    "PARP",
    "PAP0",
    "CUAP",
]


@dataclass
class CERMeta:
    cer_index: Optional[float] = None
    last_update: Optional[str] = None


# ===================== Google Sheet BONOS =====================

GOOGLE_SHEET_ID = "1AJKhrMq_lDHRRbGQ5UNkIJSmkPsBi3ekFkVODjIarIg"


def _sheet_csv_url(sheet_name: str) -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/"
        f"{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    )


def load_bonos_prices_and_expirations() -> pd.DataFrame:
    """
    Lee la hoja BONOS del Google Sheet y devuelve un DF:
        índice = Ticker
        columnas: ['Last', 'Expiration']
    """
    url = _sheet_csv_url("BONOS")
    df_raw = pd.read_csv(url)

    if df_raw.empty:
        raise RuntimeError("Hoja BONOS del Google Sheet vino vacía.")

    # Primera columna = ticker
    first_col = df_raw.columns[0]
    df = df_raw.copy()
    df = df[df[first_col].notna()]
    df = df.rename(columns={first_col: "Ticker"})
    df["Ticker"] = df["Ticker"].astype(str).str.strip()

    # Columna de precio: tratamos de ubicar 'Last'
    price_cols = [c for c in df.columns if "last" in c.lower()]
    if price_cols:
        price_col = price_cols[0]
    else:
        # fallback: la 6ta columna como en Rudolph
        if df.shape[1] <= 5:
            raise RuntimeError("No se encontró columna de precio (Last) en hoja BONOS.")
        price_col = df.columns[5]

    # Columna de vencimiento: 'Expiration Date' o similar
    exp_cols = [c for c in df.columns if "expir" in c.lower()]
    if not exp_cols:
        raise RuntimeError("No se encontró columna de vencimiento (Expiration) en hoja BONOS.")
    exp_col = exp_cols[0]

    df = df[["Ticker", price_col, exp_col]].rename(
        columns={price_col: "Last", exp_col: "Expiration"}
    )

    # Normalizamos el precio
    df["Last"] = (
        df["Last"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df["Last"] = pd.to_numeric(df["Last"], errors="coerce")

    # Normalizamos vencimiento
    df["Expiration"] = pd.to_datetime(df["Expiration"], errors="coerce")

    df = df.dropna(subset=["Last", "Expiration"])

    # (Opcional, pero recomendable) limitar a los bonos CER
    df = df[df["Ticker"].isin(CER_TICKERS)]

    return df.set_index("Ticker")


# ===================== HTTP session con retry =====================

def build_session(
    retries: int = RETRIES,
    backoff: float = BACKOFF,
) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "OPTIONS", "GET"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "DCF-Inversiones/1.0 (DCF-App CER Bonistas Scraper)"})
    return s


# ===================== Helpers de parseo =====================

def _to_float(value: str) -> Optional[float]:
    """
    Extrae el primer número de un string tipo:
      '7,00 %', '7.00%', '7,00 % anual', '25.1%' etc.
    Devuelve None si no encuentra ningún número.
    """
    if value is None:
        return None

    v = str(value).strip()
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", v)
    if not m:
        return None

    num = m.group(0).replace(",", ".")
    try:
        return float(num)
    except Exception:
        return None

def _get_from_info(info: Dict[str, str], pattern: str) -> Optional[str]:
    for k, v in info.items():
        if re.search(pattern, k, flags=re.IGNORECASE):
            return v
    return None


def _scrape_bonistas_ticker(ticker: str, session: requests.Session) -> Optional[Dict]:
    """
    Obtiene Precio y TIR desde la página de Bonistas para un ticker CER.
    
    Estructura observada en Bonistas:
      - Tabla 0 (9 x 2): 
            0                1
        0   Precio         241.30
        1   Variación ...
        ...
      - Tabla 1 (8 x 2):
            0                1
        0   TIR            27.25%
        1   Duration ...
        ...

    Tomamos:
      - Precio  = tabla[0], fila cuyo col0 contiene 'Precio', col1
      - TIR_%   = tabla[1], fila cuyo col0 contiene 'TIR', col1
    """
    url = BONISTAS_URL_TEMPLATE.format(ticker=ticker)
    try:
        resp = session.get(url, timeout=TIMEOUT_S)
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[WARN] Error {e} al obtener datos de {ticker} en Bonistas ({url})")
        return None
    except Exception as e:
        print(f"[WARN] Error genérico {e} al obtener datos de {ticker} ({url})")
        return None

    html = resp.text

    try:
        tables = pd.read_html(html)
    except ValueError:
        print(f"[WARN] pd.read_html no encontró tablas para {ticker} en Bonistas.")
        return None

    if len(tables) < 2:
        print(f"[WARN] Para {ticker} se esperaban al menos 2 tablas (Precio y TIR) y se encontraron {len(tables)}.")
        return None

    # ------------------------------------------------------------------
    # Tabla 0: Precio
    # ------------------------------------------------------------------
    tbl_price = tables[0]
    price_val = None

    if tbl_price.shape[1] >= 2:
        col0 = tbl_price.iloc[:, 0].astype(str)
        mask_precio = col0.str.contains("precio", case=False, regex=False)

        if mask_precio.any():
            price_val = tbl_price.loc[mask_precio, tbl_price.columns[1]].iloc[0]
        else:
            print(f"[INFO] No se encontró fila 'Precio' en tabla 0 para {ticker}.")

    # ------------------------------------------------------------------
    # Tabla 1: TIR
    # ------------------------------------------------------------------
    tbl_tir = tables[1]
    tir_val = None

    if tbl_tir.shape[1] >= 2:
        col0 = tbl_tir.iloc[:, 0].astype(str)
        # fila cuyo texto contenga 'TIR'
        mask_tir = col0.str.contains(r"\bTIR\b", case=False, regex=True)

        if mask_tir.any():
            tir_val = tbl_tir.loc[mask_tir, tbl_tir.columns[1]].iloc[0]
        else:
            print(f"[INFO] No se encontró fila 'TIR' en tabla 1 para {ticker}.")

    # ------------------------------------------------------------------
    # Validación final
    # ------------------------------------------------------------------
    if tir_val is None:
        print(f"[WARN] No se pudo extraer TIR para {ticker} desde Bonistas.")
        return None

    row_out = {
        "Ticker": ticker,
        "Precio_Bonistas": _to_float(price_val),
        "TIR_%": _to_float(tir_val),
    }

    # Debug opcional
    # print(f"[DEBUG CER {ticker}] Precio_raw={price_val!r}, TIR_raw={tir_val!r} -> "
    #       f"Precio={row_out['Precio_Bonistas']}, TIR={row_out['TIR_%']}")

    return row_out


# ===================== Core público =====================

def get_bonos_cer(session: Optional[requests.Session] = None) -> Tuple[pd.DataFrame, CERMeta]:
    """
    Devuelve un DataFrame con columnas:
        ['Ticker', 'Precio', 'Vencimiento', 'TIR_%']

    - Ticker, Precio y Vencimiento se toman SIEMPRE del Google Sheet (hoja BONOS),
      restringido a CER_TICKERS.
    - TIR_% se intenta obtener desde Bonistas; si no se puede, queda NaN para ese ticker.
    """
    if session is None:
        session = build_session()

    # --- 1) Base: precios y vencimientos desde Google Sheet (sólo CER_TICKERS) ---
    df_sheet = load_bonos_prices_and_expirations()   # índice = Ticker
    # nos quedamos sólo con los CER_TICKERS
    df_sheet = df_sheet[df_sheet.index.isin(CER_TICKERS)]

    # renombramos a nombres finales
    df = df_sheet.rename(columns={
        "Last": "Precio",
        "Expiration": "Vencimiento",
    }).reset_index()   # ahora tenemos col "Ticker"

    # --- 2) Scraping de TIR desde Bonistas, por cada ticker CER ---
    tir_map: Dict[str, Optional[float]] = {}

    for t in CER_TICKERS:
        # si el ticker no está en el sheet, lo salteamos
        if t not in df_sheet.index:
            continue

        data = _scrape_bonistas_ticker(t, session)
        if data is None:
            # dejamos ese ticker sin TIR
            print(f"[INFO] No se pudo obtener TIR para {t} (Bonistas).")
            continue

        tir_val = data.get("TIR_%")
        tir_map[t] = tir_val

    # asignamos la TIR mapeando por ticker
    df["TIR_%"] = df["Ticker"].map(tir_map)

    # --- 3) Orden y columnas finales ---
    df = df[["Ticker", "Precio", "Vencimiento", "TIR_%"]]
    df = df.sort_values("Vencimiento")

    meta = CERMeta(cer_index=None, last_update=None)
    return df, meta