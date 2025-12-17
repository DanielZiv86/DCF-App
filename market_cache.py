from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore

TZ_AR = ZoneInfo("America/Argentina/Buenos_Aires")

# Horario mercado AR (según tu definición)
MARKET_OPEN = time(11, 1)   # 11:01
MARKET_CLOSE = time(18, 0)  # 18:00
STEP_MINUTES = 20


def now_ar() -> datetime:
    return datetime.now(TZ_AR)


def is_market_open(dt: datetime | None = None) -> bool:
    dt = dt or now_ar()
    t = dt.timetz().replace(tzinfo=None)
    return MARKET_OPEN <= t <= MARKET_CLOSE


def _today_at(t: time, dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day, t.hour, t.minute, t.second, tzinfo=TZ_AR)


def _prev_trading_day_close(dt: datetime) -> datetime:
    # Asumimos mercado L-V. Si es sábado/domingo, retrocedemos al viernes.
    d = dt.date()
    while True:
        wd = d.weekday()  # 0=Mon ... 6=Sun
        if wd < 5:
            break
        d = d - timedelta(days=1)
    return datetime(d.year, d.month, d.day, MARKET_CLOSE.hour, MARKET_CLOSE.minute, tzinfo=TZ_AR)


def _next_trading_day_open(dt: datetime) -> datetime:
    d = dt.date()
    while True:
        wd = d.weekday()
        if wd < 5:
            break
        d = d + timedelta(days=1)
    return datetime(d.year, d.month, d.day, MARKET_OPEN.hour, MARKET_OPEN.minute, tzinfo=TZ_AR)


def market_bucket_dt(dt: datetime | None = None) -> datetime:
    """Devuelve el 'bucket' de cache vigente.

    - Dentro de mercado: el último corte de 20 minutos desde 11:01 (capado a 18:00)
    - Fuera de mercado:
        - antes de la apertura: cierre del día hábil anterior (18:00)
        - después del cierre: cierre del mismo día (18:00)
    """
    dt = dt or now_ar()

    # Ajuste a día hábil para bucket fuera de mercado
    wd = dt.weekday()
    if wd >= 5:
        # fin de semana -> usamos el cierre del viernes
        return _prev_trading_day_close(dt)

    open_dt = _today_at(MARKET_OPEN, dt)
    close_dt = _today_at(MARKET_CLOSE, dt)

    if dt < open_dt:
        return _prev_trading_day_close(dt)

    if dt >= close_dt:
        return close_dt

    # Dentro de mercado: buckets cada 20 min desde 11:01
    delta_min = int((dt - open_dt).total_seconds() // 60)
    step = (delta_min // STEP_MINUTES) * STEP_MINUTES
    bucket = open_dt + timedelta(minutes=step)
    if bucket > close_dt:
        bucket = close_dt
    return bucket


def market_bucket(dt: datetime | None = None) -> str:
    b = market_bucket_dt(dt)
    return b.strftime('%Y-%m-%d %H:%M')


def record_data_timestamp(ts: datetime | str | None) -> None:
    """Guarda un timestamp de proveedor (IOL u otro) para mostrar KPI si está disponible."""
    if st is None or ts is None:
        return

    try:
        if isinstance(ts, str):
            # Admitimos ISO o 'YYYY-mm-dd HH:MM'
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except Exception:
                dt = datetime.strptime(ts, '%Y-%m-%d %H:%M')
                dt = dt.replace(tzinfo=TZ_AR)
        else:
            dt = ts

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_AR)

        st.session_state['provider_last_ts'] = dt.astimezone(TZ_AR).isoformat()
    except Exception:
        # Si no se puede parsear, no rompemos
        return


def get_last_update_display() -> str:
    """Texto para KPI: prioriza timestamp de proveedor, si existe; si no, el bucket del cache."""
    if st is not None:
        iso = st.session_state.get('provider_last_ts')
        if iso:
            try:
                dt = datetime.fromisoformat(iso)
                return dt.astimezone(TZ_AR).strftime('%d/%m/%Y %H:%M')
            except Exception:
                pass

    b = market_bucket_dt()
    return b.strftime('%d/%m/%Y %H:%M')
