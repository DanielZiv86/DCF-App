# bandas_cambiarias.py
# ============================================================
# Módulo único y consistente para cálculo de Bandas Cambiarias
# (Nuevo método desde 01/01/2026: IPC INDEC con rezago T-2, fallback REM, fallback TAIL)
#
# Regla de actualización mensual:
#   - Para el mes m, usar IPC INDEC publicado de (m-2) si existe.
#   - Si IPC (m-2) no está disponible, usar REM del mes m.
#   - Si tampoco hay REM, usar fallback "TAIL" (decaimiento suave).
#
# Actualización de bandas (criterio acordado):
#   - Techo: techo * (1 + pi)
#   - Piso:  piso  * (1 - pi)
#
# Compatibilidad:
#   - Conserva funciones legacy: load_ipc_from_excel, load_rem_from_excel,
#     get_rem_inflacion_mensual_promedio, build_pi_series_for_bands,
#     compute_bands_monthly, compute_bands, bandas_for_dates, BandasResult
#
# Importante:
#   - Normaliza porcentajes desde Excel: 2.1, "2,10%", 0.021, etc.
#   - Parseo robusto de mes: "2026-01", "01/2026", "ene-26", etc.
# ============================================================

from __future__ import annotations

import datetime as dt
import re
import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd


# ============================================================
# Helpers de parsing (mes y %)
# ============================================================

_MONTHS_ES = {
    "ene": 1, "enero": 1,
    "feb": 2, "febrero": 2,
    "mar": 3, "marzo": 3,
    "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "junio": 6,
    "jul": 7, "julio": 7,
    "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "septiembre": 9, "set": 9,
    "oct": 10, "octubre": 10,
    "nov": 11, "noviembre": 11,
    "dic": 12, "diciembre": 12,
}


def _parse_yyyymm(x) -> str:
    """Convierte varias representaciones de fecha/mes a 'YYYY-MM'."""
    if pd.isna(x):
        raise ValueError("Mes vacío en fuente IPC/REM.")

    if isinstance(x, (dt.date, dt.datetime, pd.Timestamp)):
        ts = pd.to_datetime(x)
        return ts.strftime("%Y-%m")

    s = str(x).strip().lower()

    # formatos tipo "ene-26", "sept-26"
    m = re.match(r"^([a-zñ]{3,9})[-/\s]?(\d{2,4})$", s)
    if m:
        mon_txt = m.group(1)
        yy = m.group(2)
        if mon_txt in _MONTHS_ES:
            mm = _MONTHS_ES[mon_txt]
            yy_i = int(yy)
            if yy_i < 100:
                yy_i += 2000
            return f"{yy_i:04d}-{mm:02d}"

    # intenta parseo estándar (soporta "2025-11-01", "2025/11", "11/2025", etc.)
    try:
        ts = pd.to_datetime(s, dayfirst=True, errors="raise")
        return ts.strftime("%Y-%m")
    except Exception:
        pass

    # "202511"
    if len(s) == 6 and s.isdigit():
        return f"{s[:4]}-{s[4:]}"

    # "2025-11" o "2025/11"
    if len(s) == 7 and s[4] in "-/":
        return s[:4] + "-" + s[5:7]

    raise ValueError(f"No pude interpretar mes: {x!r}")


def _parse_pct_to_percent(x) -> float:
    """
    Devuelve porcentaje en puntos porcentuales.
    Acepta:
      - 2.5  -> 2.5
      - 0.025 -> 2.5  (si parece estar en forma decimal, típico Excel con formato %)
      - "2,5%" / "2.5%" -> 2.5
      - "-" o vacío -> NaN
    """
    if pd.isna(x):
        return float("nan")

    if isinstance(x, str):
        s = x.strip()
        if s in {"-", ""}:
            return float("nan")
        s = s.replace("%", "").replace(" ", "")

        # Manejo de miles y decimales en ES:
        # "1.234,56" -> "1234.56"
        if re.search(r"\d+\.\d{3},\d+", s):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        try:
            v = float(s)
        except Exception:
            return float("nan")
    else:
        try:
            v = float(x)
        except Exception:
            return float("nan")

    # Si parece decimal (0.021 = 2.1%) lo convertimos a %
    if 0 <= v <= 1.5:
        return v * 100.0
    return v


# ============================================================
# Helpers de fechas (API legacy)
# ============================================================

def month_add(yyyymm: str, n: int) -> str:
    y, m = map(int, yyyymm.split("-"))
    base = dt.date(y, m, 1)
    mm = (base.month - 1) + n
    yy = base.year + mm // 12
    mm = mm % 12 + 1
    return f"{yy:04d}-{mm:02d}"


def month_range(start_yyyymm: str, end_yyyymm: str) -> list[str]:
    out = []
    cur = start_yyyymm
    while cur <= end_yyyymm:
        out.append(cur)
        cur = month_add(cur, 1)
    return out


def yyyymm_from_date(d: dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


# ============================================================
# Carga de IPC real (INDEC) - Excel
# ============================================================

def load_ipc_from_excel(
    path: str | Path,
    sheet_name: str | int = 0,
    mes_col: str = "mes",
    ipc_col: str = "ipc",
) -> pd.DataFrame:
    """
    Espera una planilla con columnas de Mes e IPC.
    - mes_col: nombre de columna (ej: "Mes")
    - ipc_col: nombre de columna (ej: "IPC")
    """
    df = pd.read_excel(path, sheet_name=sheet_name).copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    mes_col = mes_col.lower().strip()
    ipc_col = ipc_col.lower().strip()

    if mes_col not in df.columns:
        raise ValueError(f"No encuentro columna '{mes_col}' en IPC. Columnas: {list(df.columns)}")
    if ipc_col not in df.columns:
        raise ValueError(f"No encuentro columna '{ipc_col}' en IPC. Columnas: {list(df.columns)}")

    out = df[[mes_col, ipc_col]].copy()
    out.columns = ["mes", "ipc_pct"]
    out["mes"] = out["mes"].map(_parse_yyyymm)
    out["ipc_pct"] = out["ipc_pct"].map(_parse_pct_to_percent)

    return out.dropna(subset=["mes", "ipc_pct"]).sort_values("mes").reset_index(drop=True)


# ============================================================
# Carga REM desde Excel local
# ============================================================

def load_rem_from_excel(
    path: str | Path,
    sheet_name: str | int = 0,
    mes_col: str = "mes",
    rem_col: str = "rem",
    **kwargs,
) -> pd.DataFrame:
    """
    Espera una planilla con columnas de Mes y REM.
    - mes_col: nombre de columna (ej: "Mes")
    - rem_col: nombre de columna (ej: "REM")
    """
    df = pd.read_excel(path, sheet_name=sheet_name, **kwargs).copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    mes_col = mes_col.lower().strip()
    rem_col = rem_col.lower().strip()

    if mes_col not in df.columns:
        raise ValueError(f"No encuentro columna '{mes_col}' en REM. Columnas: {list(df.columns)}")
    if rem_col not in df.columns:
        raise ValueError(f"No encuentro columna '{rem_col}' en REM. Columnas: {list(df.columns)}")

    out = df[[mes_col, rem_col]].copy()
    out.columns = ["mes", "rem_ipc_promedio_pct"]
    out["mes"] = out["mes"].map(_parse_yyyymm)
    out["rem_ipc_promedio_pct"] = out["rem_ipc_promedio_pct"].map(_parse_pct_to_percent)

    out = out.dropna(subset=["mes", "rem_ipc_promedio_pct"])
    out = out.sort_values("mes").drop_duplicates("mes").reset_index(drop=True)
    return out


def get_rem_inflacion_mensual_promedio(
    path: str | Path = Path("data") / "BD REM.xlsx",
    sheet_name: str | int = 0,
    mes_col: str = "mes",
    rem_col: str = "rem",
    **kwargs,
) -> pd.DataFrame:
    # Importante: respetar kwargs (tu error original venía de perder sheet_name/kwargs)
    return load_rem_from_excel(path, sheet_name=sheet_name, mes_col=mes_col, rem_col=rem_col, **kwargs)


# ============================================================
# Serie de inflación para bandas (IPC T-2 + REM + TAIL)
# ============================================================

def build_pi_series_for_bands(
    start_month: str,
    end_month: str,
    df_ipc: pd.DataFrame,
    df_rem: pd.DataFrame,
    tail_floor_pct: float = 1.0,
    tail_decay_step_pp: float = 0.10,
    tail_decay_every_n_months: int = 3,
) -> pd.DataFrame:
    months = month_range(start_month, end_month)

    df_ipc = df_ipc.copy()
    df_rem = df_rem.copy()
    df_ipc["mes"] = df_ipc["mes"].map(_parse_yyyymm)
    df_rem["mes"] = df_rem["mes"].map(_parse_yyyymm)

    ipc_map = dict(zip(df_ipc["mes"], df_ipc["ipc_pct"]))
    rem_map = dict(zip(df_rem["mes"], df_rem["rem_ipc_promedio_pct"]))

    out = []
    last_pi: Optional[float] = None
    tail_counter = 0

    for m in months:
        m_minus_2 = month_add(m, -2)

        if m_minus_2 in ipc_map and pd.notna(ipc_map[m_minus_2]):
            pi = float(ipc_map[m_minus_2])
            src = f"IPC INDEC (T-2: {m_minus_2})"
            tail_counter = 0

        elif m in rem_map and pd.notna(rem_map[m]):
            pi = float(rem_map[m])
            src = f"REM (m: {m})"
            tail_counter = 0

        else:
            # fallback: usa último pi y lo baja cada N meses hasta un piso
            if last_pi is None or pd.isna(last_pi):
                pi = float(tail_floor_pct)
            else:
                tail_counter += 1
                if tail_counter % tail_decay_every_n_months == 1:
                    pi = max(float(tail_floor_pct), float(last_pi) - float(tail_decay_step_pp))
                else:
                    pi = float(last_pi)
            src = "TAIL"

        last_pi = pi
        out.append({"mes": m, "pi_pct": pi, "fuente": src})

    return pd.DataFrame(out)


# ============================================================
# Cálculo mensual de bandas
# ============================================================

def compute_bands_monthly(
    df_pi: pd.DataFrame,
    piso_inicial: float,
    techo_inicial: float,
    metodo_piso: str = "multiply",  # "multiply" => piso*(1-pi); "divide" => piso/(1+pi)
) -> pd.DataFrame:
    piso = float(piso_inicial)
    techo = float(techo_inicial)

    rows = []
    for _, r in df_pi.iterrows():
        pi = float(r["pi_pct"]) / 100.0

        piso_ini = piso
        techo_ini = techo

        techo = techo * (1 + pi)
        if metodo_piso == "divide":
            piso = piso / (1 + pi)
        else:
            piso = piso * (1 - pi)

        rows.append(
            {
                "mes": r["mes"],
                "pi_pct": float(r["pi_pct"]),
                "fuente": r.get("fuente", ""),
                "piso_ini": piso_ini,
                "techo_ini": techo_ini,
                "piso_fin": piso,
                "techo_fin": techo,
            }
        )

    return pd.DataFrame(rows)


def compute_bands(
    df_pi: pd.DataFrame,
    piso_inicial: float,
    techo_inicial: float,
    metodo_piso: str = "multiply",
) -> pd.DataFrame:
    df = compute_bands_monthly(df_pi, piso_inicial, techo_inicial, metodo_piso)
    out = df[["mes", "pi_pct", "fuente", "piso_fin", "techo_fin"]].copy()
    out = out.rename(columns={"piso_fin": "piso", "techo_fin": "techo"})
    return out


# ============================================================
# Bandas por fecha (soporta df mensual con piso_fin/techo_fin o piso/techo)
# ============================================================

def bandas_for_dates(
    fechas: Iterable,
    df_monthly: pd.DataFrame,
    bandas_res: str = "fin_mes",
    df_dias_hab=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Devuelve bandas por fecha.

    bandas_res:
      - "fin_mes" (default): usa valores de fin de mes (piso_fin/techo_fin o piso/techo).
      - "pro_rata_cal": prorratea linealmente dentro del mes por días calendario,
                        usando (ini -> fin) del mes.
      - "pro_rata_hab": prorratea linealmente dentro del mes por días hábiles (Mon-Fri),
                        usando (ini -> fin) del mes.

    Nota:
      - Para prorrateo, df_monthly debe contener piso_ini/techo_ini y piso_fin/techo_fin.
        Si solo tiene piso/techo (fin de mes), se cae a "fin_mes".
    """
    fechas_ts = pd.to_datetime(list(fechas))

    df = df_monthly.copy()
    if "mes" not in df.columns:
        raise ValueError("df_monthly debe tener una columna 'mes' con formato YYYY-MM.")
    df = df.set_index("mes")

    has_ini = ("piso_ini" in df.columns) and ("techo_ini" in df.columns)
    has_fin = ("piso_fin" in df.columns) and ("techo_fin" in df.columns)

    piso_fin_col = "piso_fin" if "piso_fin" in df.columns else ("piso" if "piso" in df.columns else None)
    techo_fin_col = "techo_fin" if "techo_fin" in df.columns else ("techo" if "techo" in df.columns else None)
    if piso_fin_col is None or techo_fin_col is None:
        raise ValueError(f"df_monthly no tiene columnas esperadas. Columnas: {list(df.columns)}")

    bandas_res = (bandas_res or "fin_mes").strip().lower()

    def _month_start_end(mes: str):
        if mes not in df.index:
            raise KeyError(f"No hay bandas calculadas para mes {mes}.")
        row = df.loc[mes]
        if has_ini and has_fin:
            piso_ini = float(row["piso_ini"]); piso_fin = float(row["piso_fin"])
            techo_ini = float(row["techo_ini"]); techo_fin = float(row["techo_fin"])
        else:
            piso_ini = float(row[piso_fin_col]); piso_fin = float(row[piso_fin_col])
            techo_ini = float(row[techo_fin_col]); techo_fin = float(row[techo_fin_col])
        return piso_ini, piso_fin, techo_ini, techo_fin

    def _frac_calendar(f: pd.Timestamp) -> float:
        last_day = calendar.monthrange(f.year, f.month)[1]
        if last_day <= 1:
            return 1.0
        return (f.day - 1) / (last_day - 1)

    def _frac_business(f: pd.Timestamp) -> float:
        start = pd.Timestamp(f.year, f.month, 1)
        end = pd.Timestamp(f.year, f.month, calendar.monthrange(f.year, f.month)[1])
        bdays = pd.bdate_range(start=start, end=end)
        if len(bdays) <= 1:
            return 1.0
        f2 = f
        if f2.weekday() >= 5:
            f2 = f2 - pd.Timedelta(days=(f2.weekday() - 4))
        bdays_upto = bdays[bdays <= f2]
        if len(bdays_upto) == 0:
            return 0.0
        return (len(bdays_upto) - 1) / (len(bdays) - 1)

    rows = []
    for f in fechas_ts:
        mes = f.strftime("%Y-%m")
        piso_ini, piso_fin, techo_ini, techo_fin = _month_start_end(mes)

        if bandas_res == "fin_mes" or not (has_ini and has_fin):
            piso = piso_fin
            techo = techo_fin
        elif bandas_res in {"pro_rata_cal", "prorrata_cal", "cal"}:
            frac = _frac_calendar(f)
            piso = piso_ini + (piso_fin - piso_ini) * frac
            techo = techo_ini + (techo_fin - techo_ini) * frac
        elif bandas_res in {"pro_rata_hab", "prorrata_hab", "hab", "habil"}:
            frac = _frac_business(f)
            piso = piso_ini + (piso_fin - piso_ini) * frac
            techo = techo_ini + (techo_fin - techo_ini) * frac
        else:
            raise ValueError(f"bandas_res inválido: {bandas_res!r}")

        rows.append({"fecha": f.date(), "mes": mes, "piso": float(piso), "techo": float(techo)})

    return pd.DataFrame(rows)


# ============================================================
# Helpers legacy
# ============================================================

def end_month_from_maturities(df, maturity_col="vencimiento") -> str:
    fechas = pd.to_datetime(df[maturity_col])
    return fechas.max().strftime("%Y-%m")


@dataclass
class BandasResult:
    df_bandas_full: pd.DataFrame
    df_pi_full: pd.DataFrame
