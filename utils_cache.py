from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

AR_TZ = ZoneInfo("America/Argentina/Buenos_Aires")

@dataclass(frozen=True)
class CachePolicy:
    ttl_seconds: int
    mode: str
    now_ar: datetime
    next_refresh_ar: datetime

def compute_ttl_iol_schedule(
    now: datetime | None = None,
    start_time: time = time(11, 1),   # 11:01
    end_time: time = time(18, 0),     # 18:00
    step_minutes: int = 20,           # cada 20 minutos
    safety_pad_seconds: int = 5,       # margen para que el backend termine de publicar
    min_ttl_seconds: int = 15,         # piso
) -> CachePolicy:
    now_ar = (now or datetime.now(tz=AR_TZ)).astimezone(AR_TZ)
    today = now_ar.date()

    start_dt = datetime.combine(today, start_time, tzinfo=AR_TZ)
    end_dt = datetime.combine(today, end_time, tzinfo=AR_TZ)

    # Fuera de ventana: cache hasta mañana 11:01
    if now_ar < start_dt:
        next_dt = start_dt
        ttl = int((next_dt - now_ar).total_seconds())
        return CachePolicy(max(ttl, 60), "market_closed_preopen", now_ar, next_dt)

    if now_ar >= end_dt:
        next_dt = datetime.combine(today + timedelta(days=1), start_time, tzinfo=AR_TZ)
        ttl = int((next_dt - now_ar).total_seconds())
        return CachePolicy(max(ttl, 60), "market_closed_afterclose", now_ar, next_dt)

    # Dentro de ventana: calcular próximo tick (11:01 + k*20m)
    elapsed = now_ar - start_dt
    elapsed_minutes = int(elapsed.total_seconds() // 60)

    # k = cantidad de pasos completos transcurridos
    k = elapsed_minutes // step_minutes
    next_dt = start_dt + timedelta(minutes=(k + 1) * step_minutes)

    # Si por alguna razón next_dt pasa 18:00, cache hasta el cierre
    if next_dt > end_dt:
        next_dt = end_dt

    # TTL hasta (próximo tick + safety_pad)
    ttl = int((next_dt - now_ar).total_seconds()) + safety_pad_seconds
    ttl = max(ttl, min_ttl_seconds)

    return CachePolicy(ttl, "market_open_20min_ticks", now_ar, next_dt)
