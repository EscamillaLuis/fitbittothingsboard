from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

DEFAULT_TIMEZONE = os.getenv("FITBIT_DEFAULT_TIMEZONE", "America/Mexico_City")
DEFAULT_DATA_DIR = os.getenv("FITBIT_DATA_DIR", "fitbit_data")
BASELINE_CACHE_NAME = "baselines_cache.json"
BASELINE_DAYS = 14
WINDOW_MINUTES = 30

_DATE_PATTERN = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_.*\.json$")

logger = logging.getLogger(__name__)


class MetricBaseline(TypedDict):
    median: float
    p10: float
    p90: float
    count: int


class ClientBaseline(TypedDict, total=False):
    HR: MetricBaseline
    HRV: MetricBaseline
    SpO2: MetricBaseline


class BaselineCache(TypedDict, total=False):
    generated_at: str
    window_days: int
    clients: Dict[str, ClientBaseline]


def _resolve_tz(tz_name: Optional[str]) -> ZoneInfo:
    candidate = tz_name or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(candidate)
    except ZoneInfoNotFoundError:
        if candidate != DEFAULT_TIMEZONE:
            logger.warning("Zona horaria desconocida '%s'. Usando '%s'.", candidate, DEFAULT_TIMEZONE)
        return ZoneInfo(DEFAULT_TIMEZONE)


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) else None
    return None


def _parse_clock_time(date_str: str, clock: str) -> Optional[datetime]:
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    raw = clock.strip()
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M"):
        try:
            time_obj = datetime.strptime(raw, fmt).time()
            return datetime.combine(day, time_obj)
        except ValueError:
            continue
    return None


def _parse_sample_dt(raw_value: Any, fallback_date: str, tzinfo: ZoneInfo) -> Optional[datetime]:
    if not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    if not text:
        return None
    if "T" not in text and "-" not in text:
        return _parse_clock_time(fallback_date, text)
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt_obj = datetime.fromisoformat(iso_text)
    except ValueError:
        return _parse_clock_time(fallback_date, text)
    if dt_obj.tzinfo is not None:
        dt_obj = dt_obj.astimezone(tzinfo).replace(tzinfo=None)
    return dt_obj


def _extract_hr_samples(data: Mapping[str, Any], date_str: str, tzinfo: ZoneInfo) -> List[Tuple[datetime, float]]:
    heart = data.get("Ritmo_Cardiaco")
    dataset: Any = None
    if isinstance(heart, Mapping):
        nested = heart.get("activities-heart-intraday")
        if isinstance(nested, Mapping):
            dataset = nested.get("dataset")
        if not isinstance(dataset, list):
            dataset = heart.get("dataset")
    if not isinstance(dataset, list):
        return []

    out: List[Tuple[datetime, float]] = []
    for entry in dataset:
        if not isinstance(entry, Mapping):
            continue
        value = _safe_float(entry.get("value"))
        if value is None:
            continue
        sample_dt = _parse_sample_dt(entry.get("time"), date_str, tzinfo)
        if sample_dt is None:
            continue
        out.append((sample_dt, value))
    return out


def _extract_hrv_samples(data: Mapping[str, Any], date_str: str, tzinfo: ZoneInfo) -> List[Tuple[datetime, float]]:
    minutes: List[Mapping[str, Any]] = []
    intraday = data.get("HRV_intraday")
    if isinstance(intraday, Mapping):
        minute_block = intraday.get("minutes")
        if isinstance(minute_block, list):
            minutes.extend(m for m in minute_block if isinstance(m, Mapping))

    if not minutes:
        hrv_daily = data.get("HRV")
        if isinstance(hrv_daily, list):
            for item in hrv_daily:
                if not isinstance(item, Mapping):
                    continue
                minute_block = item.get("minutes")
                if isinstance(minute_block, list):
                    minutes.extend(m for m in minute_block if isinstance(m, Mapping))

    out: List[Tuple[datetime, float]] = []
    for minute in minutes:
        rmssd = _safe_float(minute.get("rmssd"))
        if rmssd is None:
            nested_value = minute.get("value")
            if isinstance(nested_value, Mapping):
                rmssd = _safe_float(nested_value.get("rmssd"))
        if rmssd is None:
            continue
        ts_raw = minute.get("time")
        if not isinstance(ts_raw, str):
            ts_raw = minute.get("minute")
        sample_dt = _parse_sample_dt(ts_raw, date_str, tzinfo)
        if sample_dt is None:
            continue
        out.append((sample_dt, rmssd))
    return out


def _extract_spo2_samples(data: Mapping[str, Any], date_str: str, tzinfo: ZoneInfo) -> List[Tuple[datetime, float]]:
    spo2 = data.get("SpO2")
    minute_entries: List[Mapping[str, Any]] = []

    if isinstance(spo2, list):
        minute_entries.extend(item for item in spo2 if isinstance(item, Mapping))
    elif isinstance(spo2, Mapping):
        for key in ("minutes", "minuteSpO2"):
            block = spo2.get(key)
            if isinstance(block, list):
                minute_entries.extend(item for item in block if isinstance(item, Mapping))

    out: List[Tuple[datetime, float]] = []
    for entry in minute_entries:
        value = _safe_float(entry.get("value"))
        if value is None:
            continue
        minute_raw: Any = None
        for field in ("minute", "time", "timestamp"):
            if isinstance(entry.get(field), str):
                minute_raw = entry.get(field)
                break
        sample_dt = _parse_sample_dt(minute_raw, date_str, tzinfo)
        if sample_dt is None:
            continue
        out.append((sample_dt, value))
    return out


def _aggregate_metric_30m(samples: Iterable[Tuple[datetime, float]]) -> Dict[datetime, float]:
    buckets: Dict[datetime, List[float]] = defaultdict(list)
    for sample_dt, value in samples:
        start = sample_dt.replace(
            minute=(sample_dt.minute // WINDOW_MINUTES) * WINDOW_MINUTES,
            second=0,
            microsecond=0,
        )
        window_end = start + timedelta(minutes=WINDOW_MINUTES)
        buckets[window_end].append(value)
    return {window_end: float(mean(values)) for window_end, values in buckets.items() if values}


def _build_window_rows(client_id: str, data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    date_str = str(data.get("Fecha") or "").strip()
    if not date_str:
        return []
    tz_name = data.get("Timezone")
    tzinfo = _resolve_tz(tz_name if isinstance(tz_name, str) else None)

    hr_30m = _aggregate_metric_30m(_extract_hr_samples(data, date_str, tzinfo))
    hrv_30m = _aggregate_metric_30m(_extract_hrv_samples(data, date_str, tzinfo))
    spo2_30m = _aggregate_metric_30m(_extract_spo2_samples(data, date_str, tzinfo))

    all_windows = sorted(set(hr_30m) | set(hrv_30m) | set(spo2_30m))
    rows: List[Dict[str, Any]] = []
    for window_end in all_windows:
        rows.append(
            {
                "client_id": client_id,
                "window_end": window_end.isoformat(timespec="seconds"),
                "HR": hr_30m.get(window_end),
                "HRV": hrv_30m.get(window_end),
                "SpO2": spo2_30m.get(window_end),
            }
        )
    return rows


def _parse_file_date(path: Path) -> Optional[date]:
    match = _DATE_PATTERN.match(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group("date"), "%Y-%m-%d").date()
    except ValueError:
        return None


def _collect_recent_files(data_path: Path) -> Dict[str, List[Path]]:
    today = date.today()
    cutoff = today - timedelta(days=BASELINE_DAYS - 1)
    out: Dict[str, List[Path]] = {}

    for client_dir in data_path.iterdir():
        if not client_dir.is_dir():
            continue
        client_id = client_dir.name
        latest_per_day: Dict[date, Path] = {}
        for json_path in client_dir.rglob("*.json"):
            if json_path.name == BASELINE_CACHE_NAME:
                continue
            file_date = _parse_file_date(json_path)
            if file_date is None or file_date < cutoff or file_date > today:
                continue
            previous = latest_per_day.get(file_date)
            if previous is None or json_path.stat().st_mtime > previous.stat().st_mtime:
                latest_per_day[file_date] = json_path
        if latest_per_day:
            out[client_id] = [latest_per_day[d] for d in sorted(latest_per_day)]
    return out


def _score_high(value: float, baseline: Optional[float], p90: Optional[float]) -> Optional[float]:
    if baseline is None or p90 is None:
        return None
    denom = p90 - baseline
    if denom <= 0:
        return None
    raw = (value - baseline) / denom
    return max(0.0, min(2.0, raw))


def _score_low(value: float, baseline: Optional[float], p10: Optional[float]) -> Optional[float]:
    if baseline is None or p10 is None:
        return None
    denom = baseline - p10
    if denom <= 0:
        return None
    raw = (baseline - value) / denom
    return max(0.0, min(2.0, raw))


def _load_cache(cache_path: Path) -> BaselineCache:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            return raw
    except Exception:
        logger.exception("No se pudo leer caché de baselines: %s", cache_path)
    return {}


def _baseline_value(client_baseline: Mapping[str, Any], metric: str, key: str) -> Optional[float]:
    metric_data = client_baseline.get(metric)
    if not isinstance(metric_data, Mapping):
        return None
    return _safe_float(metric_data.get(key))


def actualizar_baselines_historicos(data_dir: str) -> bool:
    data_path = Path(data_dir)
    cache_path = data_path / BASELINE_CACHE_NAME

    if not data_path.exists():
        logger.warning("No existe el directorio de datos Fitbit: %s", data_path)
        return False

    try:
        files_by_client = _collect_recent_files(data_path)
        rows: List[Dict[str, Any]] = []

        for client_id, paths in files_by_client.items():
            for json_path in paths:
                try:
                    with json_path.open("r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                except Exception:
                    logger.exception("No se pudo leer %s", json_path)
                    continue
                if not isinstance(payload, dict):
                    continue
                rows.extend(_build_window_rows(client_id, payload))

        cache: BaselineCache = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_days": BASELINE_DAYS,
            "clients": {},
        }

        if rows:
            df = pd.DataFrame(rows)
            clients_payload: Dict[str, ClientBaseline] = {}

            for client, group in df.groupby("client_id"):
                client_baseline: ClientBaseline = {}
                for metric in ("HR", "HRV", "SpO2"):
                    series = pd.to_numeric(group[metric], errors="coerce").dropna()
                    if series.empty:
                        continue
                    client_baseline[metric] = {
                        "median": float(series.median()),
                        "p10": float(series.quantile(0.10)),
                        "p90": float(series.quantile(0.90)),
                        "count": int(series.shape[0]),
                    }
                if client_baseline:
                    clients_payload[str(client)] = client_baseline

            cache["clients"] = clients_payload

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)

        logger.info(
            "Baselines intradía actualizados: %s (%d usuarios)",
            cache_path,
            len(cache.get("clients", {})),
        )
        return True
    except Exception:
        logger.exception("Error actualizando baselines históricos en %s", data_path)
        return False


def evaluar_hora_actual(client_id: str, data_de_hoy: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cache_path = Path(DEFAULT_DATA_DIR) / BASELINE_CACHE_NAME
        cache = _load_cache(cache_path)
        clients = cache.get("clients", {})
        if not isinstance(clients, dict):
            data_de_hoy["Alertas_Intradia"] = []
            return data_de_hoy

        client_baseline = clients.get(client_id)
        if not isinstance(client_baseline, Mapping):
            data_de_hoy["Alertas_Intradia"] = []
            return data_de_hoy

        rows = _build_window_rows(client_id, data_de_hoy)
        if not rows:
            data_de_hoy["Alertas_Intradia"] = []
            return data_de_hoy

        alerts: List[Dict[str, Any]] = []
        yellow_streak = 0

        for row in sorted(rows, key=lambda item: str(item.get("window_end"))):
            hr_val = _safe_float(row.get("HR"))
            hrv_val = _safe_float(row.get("HRV"))
            spo2_val = _safe_float(row.get("SpO2"))

            risk_components: List[float] = []
            reason_components: List[str] = []

            if hr_val is not None:
                risk_hr = _score_high(
                    hr_val,
                    _baseline_value(client_baseline, "HR", "median"),
                    _baseline_value(client_baseline, "HR", "p90"),
                )
                if risk_hr is not None:
                    risk_components.append(risk_hr)
                    if risk_hr >= 0.9:
                        reason_components.append("HR")

            if hrv_val is not None:
                risk_hrv = _score_low(
                    hrv_val,
                    _baseline_value(client_baseline, "HRV", "median"),
                    _baseline_value(client_baseline, "HRV", "p10"),
                )
                if risk_hrv is not None:
                    risk_components.append(risk_hrv)
                    if risk_hrv >= 0.9:
                        reason_components.append("HRV")

            if spo2_val is not None:
                risk_spo2 = _score_low(
                    spo2_val,
                    _baseline_value(client_baseline, "SpO2", "median"),
                    _baseline_value(client_baseline, "SpO2", "p10"),
                )
                if risk_spo2 is not None:
                    risk_components.append(risk_spo2)
                    if risk_spo2 >= 0.9:
                        reason_components.append("SpO2")

            if not risk_components:
                yellow_streak = 0
                continue

            risk_30m = float(mean(risk_components))
            yellow = risk_30m >= 0.9
            red = (risk_30m >= 1.2) or (yellow and yellow_streak >= 1)
            alert_level = 2 if red else (1 if yellow else 0)
            yellow_streak = yellow_streak + 1 if yellow else 0

            reason_text = "; ".join(reason_components) if reason_components else "Sin señales fuertes vs baseline intradía"
            alerts.append(
                {
                    "window_end": row.get("window_end"),
                    "risk_30m": round(risk_30m, 4),
                    "alert_level_30m": alert_level,
                    "reason_30m": reason_text,
                }
            )

        data_de_hoy["Alertas_Intradia"] = alerts
        return data_de_hoy
    except Exception:
        logger.exception("Error evaluando alertas intradía para %s", client_id)
        data_de_hoy["Alertas_Intradia"] = []
        return data_de_hoy


__all__ = ["actualizar_baselines_historicos", "evaluar_hora_actual"]
