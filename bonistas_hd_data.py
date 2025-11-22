# bonistas_hd_data.py
# ========= Scraper bonistas -> multi cotización (Pesos / MEP / CCL) =========
import re
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry


# ----------------------- Helpers scraping -----------------------

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


def _to_float(s: str | None) -> float | None:
    """Normaliza números tipo 1.234,56% o 10,5 a float."""
    if s is None:
        return None
    s = str(s).strip().lstrip("=+ ")
    # normaliza 1.234,56 -> 1234.56 si aplica
    if re.search(r"\d,\d{2}%?$", s):
        s = s.replace(".", "").replace(",", ".")
    # deja sólo dígitos, signo, punto y %
    s = re.sub(r"[^0-9\.\-%]", "", s)
    if not s:
        return None
    try:
        return float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
    except Exception:
        return None


def _find_metric(soup: BeautifulSoup, label: str) -> float | None:
    """
    Busca en el HTML el texto 'label' (ej. 'Precio', 'TIR')
    y devuelve el número que está al lado.
    """
    lbl = label.lower().strip()

    # 1) Buscar el nodo de texto exacto y leer el siguiente
    for node in soup.find_all(string=True):
        if node.strip().lower() == lbl:
            # hermano siguiente en la misma fila / div
            sib = node.parent.find_next_sibling()
            if sib and sib.get_text(strip=True):
                v = _to_float(sib.get_text(strip=True))
                if v is not None:
                    return v
            # o el siguiente texto en el flujo
            nxt = node.find_next(string=True)
            if nxt and nxt.strip() and nxt != node:
                v = _to_float(nxt.strip())
                if v is not None:
                    return v

    # 2) Fallback por regex sobre todo el texto de la página
    m = re.search(
        rf"{re.escape(label)}\s*([=+\-\d\.,%]+)",
        soup.get_text(" ", strip=True),
        flags=re.I,
    )
    return _to_float(m.group(1)) if m else None


def scrape_bonistas_metrics(ticker: str, timeout: int = 12) -> dict:
    """
    Devuelve un dict con:
      { 'Ticker': tk, 'Precio': float | None, 'TIR': float | None }
    """
    url = f"https://bonistas.com/bono-cotizacion-rendimiento-precio-hoy/{ticker}"
    s = _session()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    return {
        "Ticker": ticker,
        "Precio": _find_metric(soup, "Precio"),
        "TIR": _find_metric(soup, "TIR"),
    }


# ----------------------- Duration desde el ticker -----------------------

def _maturity_from_ticker(tk: str) -> date | None:
    """
    Extrae YY del ticker (…YY, …YYD o …YYC) y devuelve 09/07/20YY.
    Ej: AL30, AL30D o AL30C -> 09/07/2030
    """
    tk = tk.upper()
    # últimos dos dígitos de año + opcional D/C al final
    m = re.search(r"(\d{2})([DC])?$", tk)
    if not m:
        return None
    yy = int(m.group(1))
    year = 2000 + yy
    return date(year, 7, 9)


def _duration_years_from_ticker(tk: str, today: date | None = None) -> float | None:
    if today is None:
        today = date.today()
    mat = _maturity_from_ticker(tk)
    if not mat:
        return None
    days = (mat - today).days
    if days < 0:
        days = 0
    return round(days / 365.0, 1)


# ----------------------- Multi cotización: PESOS / MEP / CCL -----------------------

# Tickers base sin sufijo (AL30, GD30, etc.)
BASE_TICKERS = [
    "AL29", "AL30", "AE38", "AL35", "AL41",
    "GD29", "GD30", "GD35", "GD38", "GD41", "GD46",
]

# Definición de mercados y sufijos
MERCADOS = [
    ("PESOS", ""),   # AL30
    ("MEP",   "D"),  # AL30D
    ("CCL",   "C"),  # AL30C
]


def scrape_bonistas_multi_mercado(
    base_tickers: list[str] | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """
    Devuelve una tabla con los mismos bonos en 3 mercados:
    - PESOS  -> ticker base (ej: AL30)
    - MEP    -> ticker con 'D' final (ej: AL30D)
    - CCL    -> ticker con 'C' final (ej: AL30C)

    Columnas:
      Base, Ticker, Mercado, Precio, Duration, TIR
    """
    if base_tickers is None:
        base_tickers = BASE_TICKERS

    rows: list[dict] = []
    for base in base_tickers:
        for mercado, sufijo in MERCADOS:
            tk = f"{base}{sufijo}"
            try:
                m = scrape_bonistas_metrics(tk)
                duration = _duration_years_from_ticker(base, today=today)
                rows.append({
                    "Base": base,
                    "Ticker": tk,
                    "Mercado": mercado,
                    "Precio": m["Precio"],
                    "Duration": duration,
                    "TIR": m["TIR"],
                })
            except Exception as e:
                print(f"⚠️ Error con {tk}: {e}")
                rows.append({
                    "Base": base,
                    "Ticker": tk,
                    "Mercado": mercado,
                    "Precio": None,
                    "Duration": _duration_years_from_ticker(base, today=today),
                    "TIR": None,
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["Base", "Mercado"])
    return df


def get_multi_table(
    mercado: str | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """
    Helper para obtener la tabla multi-cotización.
    Si `mercado` es:
      - None   -> devuelve todo
      - 'PESOS', 'MEP' o 'CCL' -> filtra por ese mercado
    """
    df = scrape_bonistas_multi_mercado(BASE_TICKERS, today=today)
    if mercado is not None:
        mercado = mercado.upper()
        df = df[df["Mercado"] == mercado]
    return df


def get_hd_table(today: date | None = None) -> pd.DataFrame:
    """
    Devuelve la tabla HD MEP en el formato simple anterior:
      índice = Ticker
      columnas = ['Precio', 'Duration', 'TIR']
    Internamente usa la tabla multi-mercado filtrada por MEP.
    """
    df_mep = get_multi_table("MEP", today=today).copy()
    df_mep = df_mep.set_index("Ticker")[["Precio", "Duration", "TIR"]]
    return df_mep


# ----------------------- Test rápido -----------------------

"""if __name__ == "__main__":
    print("=== Multi cotización (primeras filas) ===")
    df_all = get_multi_table()
    print(df_all.head(), "\n")"""
