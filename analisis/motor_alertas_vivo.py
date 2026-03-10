from __future__ import annotations

import json
import logging
import math
import numbers
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
MIN_COV_3D = 100
MIN_COV_7D = 200
PERSIST_YELLOW_RATE = 0.01
PERSIST_RED_RATE = 0.03

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
    Steps: MetricBaseline
    Sedentary: MetricBaseline
    Sleep: MetricBaseline
    history_14d: List[Dict[str, Any]]


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
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
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


def _extract_steps_samples(data: Mapping[str, Any], date_str: str, tzinfo: ZoneInfo) -> List[Tuple[datetime, float]]:
    activities = data.get("Actividades")
    if not isinstance(activities, Mapping):
        return []
    dataset = activities.get("steps")
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


def _extract_sedentary_samples(data: Mapping[str, Any], date_str: str, tzinfo: ZoneInfo) -> List[Tuple[datetime, float]]:
    activities = data.get("Actividades")
    if not isinstance(activities, Mapping):
        return []
    dataset = activities.get("calories")
    if not isinstance(dataset, list):
        return []

    out: List[Tuple[datetime, float]] = []
    for entry in dataset:
        if not isinstance(entry, Mapping):
            continue
        sample_dt = _parse_sample_dt(entry.get("time"), date_str, tzinfo)
        if sample_dt is None:
            continue
        level = _safe_float(entry.get("level"))
        sedentary_flag = 1.0 if level == 0.0 else 0.0
        out.append((sample_dt, sedentary_flag))
    return out


def _extract_sleep_daily(data: Mapping[str, Any]) -> Optional[float]:
    sleep_summary = data.get("Resumen_Sue\u00f1o", [])
    if not isinstance(sleep_summary, list) or not sleep_summary:
        return None

    entries = [entry for entry in sleep_summary if isinstance(entry, Mapping)]
    if not entries:
        return None

    main_entries = [entry for entry in entries if entry.get("isMainSleep") is True]
    candidates = main_entries or entries

    for key in ("minutesAsleep", "timeInBed", "duration", "efficiency", "score"):
        best_value: Optional[float] = None
        for entry in candidates:
            value = _safe_float(entry.get(key))
            if value is None:
                continue
            if key == "duration":
                value = value / 60000.0
            if best_value is None or value > best_value:
                best_value = value
        if best_value is not None:
            return best_value
    return None


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


def _aggregate_metric_30m_sum(samples: Iterable[Tuple[datetime, float]]) -> Dict[datetime, float]:
    buckets: Dict[datetime, List[float]] = defaultdict(list)
    for sample_dt, value in samples:
        start = sample_dt.replace(
            minute=(sample_dt.minute // WINDOW_MINUTES) * WINDOW_MINUTES,
            second=0,
            microsecond=0,
        )
        window_end = start + timedelta(minutes=WINDOW_MINUTES)
        buckets[window_end].append(value)
    return {window_end: float(sum(values)) for window_end, values in buckets.items() if values}


def _build_window_rows(client_id: str, data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    date_str = str(data.get("Fecha") or "").strip()
    if not date_str:
        return []
    tz_name = data.get("Timezone")
    tzinfo = _resolve_tz(tz_name if isinstance(tz_name, str) else None)

    hr_samples = _extract_hr_samples(data, date_str, tzinfo)
    hrv_samples = _extract_hrv_samples(data, date_str, tzinfo)
    spo2_samples = _extract_spo2_samples(data, date_str, tzinfo)
    steps_samples = _extract_steps_samples(data, date_str, tzinfo)
    sedentary_samples = _extract_sedentary_samples(data, date_str, tzinfo)

    hr_30m = _aggregate_metric_30m(hr_samples)
    hrv_30m = _aggregate_metric_30m(hrv_samples)
    spo2_30m = _aggregate_metric_30m(spo2_samples)
    steps_30m = _aggregate_metric_30m_sum(steps_samples)
    sedentary_30m = _aggregate_metric_30m(sedentary_samples)

    all_windows = sorted(set(hr_30m) | set(hrv_30m) | set(spo2_30m) | set(steps_30m) | set(sedentary_30m))
    rows: List[Dict[str, Any]] = []
    for window_end in all_windows:
        rows.append(
            {
                "client_id": client_id,
                "window_end": window_end.isoformat(timespec="seconds"),
                "HR": hr_30m.get(window_end),
                "HRV": hrv_30m.get(window_end),
                "SpO2": spo2_30m.get(window_end),
                "Steps": steps_30m.get(window_end),
                "Sedentary": sedentary_30m.get(window_end),
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


def _alert_levels_from_risk(risk: pd.Series, cov: pd.Series, min_cov: int) -> pd.Series:
    ok = cov.fillna(0) >= min_cov
    yellow = (risk >= 0.9) & ok
    yellow2 = yellow.rolling(window=2, min_periods=2).sum().ge(2)
    red = ((risk >= 1.2) & ok) | yellow2
    levels = pd.Series(0, index=risk.index, dtype="int64")
    levels.loc[yellow] = 1
    levels.loc[red] = 2
    return levels


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


def _build_history_14d(group: pd.DataFrame, client_baseline: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if group is None or group.empty:
        return []

    history_rows: List[Dict[str, Any]] = []
    baseline_hr = _baseline_value(client_baseline, "HR", "median")
    baseline_hr_p90 = _baseline_value(client_baseline, "HR", "p90")
    baseline_hrv = _baseline_value(client_baseline, "HRV", "median")
    baseline_hrv_p10 = _baseline_value(client_baseline, "HRV", "p10")
    baseline_spo2 = _baseline_value(client_baseline, "SpO2", "median")
    baseline_spo2_p10 = _baseline_value(client_baseline, "SpO2", "p10")

    for record in group.to_dict("records"):
        window_end = record.get("window_end")
        if not window_end:
            continue

        hr_val = _safe_float(record.get("HR"))
        hrv_val = _safe_float(record.get("HRV"))
        spo2_val = _safe_float(record.get("SpO2"))

        risk_components: List[float] = []
        if hr_val is not None:
            risk_hr = _score_high(hr_val, baseline_hr, baseline_hr_p90)
            if risk_hr is not None:
                risk_components.append(risk_hr)
        if hrv_val is not None:
            risk_hrv = _score_low(hrv_val, baseline_hrv, baseline_hrv_p10)
            if risk_hrv is not None:
                risk_components.append(risk_hrv)
        if spo2_val is not None:
            risk_spo2 = _score_low(spo2_val, baseline_spo2, baseline_spo2_p10)
            if risk_spo2 is not None:
                risk_components.append(risk_spo2)

        comp_count = len(risk_components)
        risk_30m = float(mean(risk_components)) if risk_components else None
        history_rows.append(
            {
                "window_end": window_end,
                "risk_30m": risk_30m,
                "comp_count_30m": comp_count,
            }
        )

    if not history_rows:
        return []

    history_df = pd.DataFrame(history_rows)
    history_df["window_end"] = pd.to_datetime(history_df["window_end"], errors="coerce")
    history_df = history_df.dropna(subset=["window_end"])
    if history_df.empty:
        return []

    history_df = history_df.sort_values("window_end").set_index("window_end")
    history_df["alert_level_30m"] = _alert_levels_from_risk(
        history_df["risk_30m"],
        history_df["comp_count_30m"],
        min_cov=1,
    )

    history_df = history_df.reset_index()
    out: List[Dict[str, Any]] = []
    for row in history_df.itertuples(index=False):
        risk_val = row.risk_30m
        out.append(
            {
                "window_end": row.window_end.strftime("%Y-%m-%dT%H:%M:%S"),
                "risk_30m": None if pd.isna(risk_val) else float(risk_val),
                "alert_level_30m": int(row.alert_level_30m) if not pd.isna(row.alert_level_30m) else 0,
            }
        )
    return out


def actualizar_baselines_historicos(data_dir: str) -> bool:
    data_path = Path(data_dir)
    cache_path = data_path / BASELINE_CACHE_NAME

    if not data_path.exists():
        logger.warning("No existe el directorio de datos Fitbit: %s", data_path)
        return False

    try:
        files_by_client = _collect_recent_files(data_path)
        rows: List[Dict[str, Any]] = []
        sleep_rows: List[Dict[str, Any]] = []

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
                sleep_val = _extract_sleep_daily(payload)
                if sleep_val is not None:
                    sleep_rows.append({"client_id": client_id, "Sleep": sleep_val})

        cache: BaselineCache = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_days": BASELINE_DAYS,
            "clients": {},
        }

        if rows or sleep_rows:
            df = pd.DataFrame(rows)
            sleep_df = pd.DataFrame(sleep_rows)
            clients_payload: Dict[str, ClientBaseline] = {}
            clients_seen = set()
            if not df.empty and "client_id" in df:
                clients_seen.update(str(client) for client in df["client_id"].dropna().unique())
            if not sleep_df.empty and "client_id" in sleep_df:
                clients_seen.update(str(client) for client in sleep_df["client_id"].dropna().unique())

            for client in sorted(clients_seen):
                client_baseline: ClientBaseline = {}
                group = df.loc[df["client_id"] == client] if not df.empty else pd.DataFrame()
                client_sleep = (
                    sleep_df.loc[sleep_df["client_id"] == client, "Sleep"]
                    if not sleep_df.empty and "client_id" in sleep_df and "Sleep" in sleep_df
                    else pd.Series(dtype=float)
                )

                for key in ("HR", "HRV", "SpO2", "Steps", "Sedentary", "Sleep"):
                    if key == "Sleep":
                        series = pd.to_numeric(client_sleep, errors="coerce").dropna()
                    elif not group.empty and key in group:
                        series = pd.to_numeric(group[key], errors="coerce").dropna()
                    else:
                        series = pd.Series(dtype=float)
                    if series.empty:
                        continue
                    client_baseline[key] = {
                        "median": float(series.median()),
                        "p10": float(series.quantile(0.10)),
                        "p90": float(series.quantile(0.90)),
                        "count": int(series.shape[0]),
                    }
                if client_baseline:
                    history_14d = _build_history_14d(group, client_baseline)
                    client_baseline["history_14d"] = history_14d
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

        history_raw = client_baseline.get("history_14d")
        history_df = pd.DataFrame(history_raw) if isinstance(history_raw, list) else pd.DataFrame()
        if not history_df.empty:
            history_df["window_end"] = pd.to_datetime(history_df.get("window_end"), errors="coerce")
            history_df["risk_30m"] = pd.to_numeric(history_df.get("risk_30m"), errors="coerce")
            history_df["alert_level_30m"] = pd.to_numeric(history_df.get("alert_level_30m"), errors="coerce")
            history_df = history_df.dropna(subset=["window_end"])

        rows = _build_window_rows(client_id, data_de_hoy)
        sleep_val = _extract_sleep_daily(data_de_hoy)
        if not rows:
            data_de_hoy["Alertas_Intradia"] = []
            return data_de_hoy

        alerts: List[Dict[str, Any]] = []
        alerts_roll_rows: List[Dict[str, Any]] = []
        yellow_streak = 0

        for row in sorted(rows, key=lambda item: str(item.get("window_end"))):
            hr_val = _safe_float(row.get("HR"))
            hrv_val = _safe_float(row.get("HRV"))
            spo2_val = _safe_float(row.get("SpO2"))
            steps_val = _safe_float(row.get("Steps"))
            sedentary_val = _safe_float(row.get("Sedentary"))

            risk_components: List[float] = []
            reason_components: List[str] = []
            risk_steps_val: Optional[float] = None
            risk_sedentary_val: Optional[float] = None
            risk_sleep_val: Optional[float] = None

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

            if steps_val is not None:
                risk_steps_val = _score_low(
                    steps_val,
                    _baseline_value(client_baseline, "Steps", "median"),
                    _baseline_value(client_baseline, "Steps", "p10"),
                )

            if sedentary_val is not None:
                risk_sedentary_val = _score_high(
                    sedentary_val,
                    _baseline_value(client_baseline, "Sedentary", "median"),
                    _baseline_value(client_baseline, "Sedentary", "p90"),
                )

            if sleep_val is not None:
                risk_sleep_val = _score_low(
                    sleep_val,
                    _baseline_value(client_baseline, "Sleep", "median"),
                    _baseline_value(client_baseline, "Sleep", "p10"),
                )

            comp_count = len(risk_components)
            if not risk_components and risk_steps_val is None and risk_sedentary_val is None:
                yellow_streak = 0
                continue

            if risk_components:
                risk_30m = float(mean(risk_components))
                yellow = risk_30m >= 0.9
                red = (risk_30m >= 1.2) or (yellow and yellow_streak >= 1)
                alert_level = 2 if red else (1 if yellow else 0)
                yellow_streak = yellow_streak + 1 if yellow else 0
            else:
                risk_30m = 0.0
                alert_level = 0

            reason_text = "; ".join(reason_components) if reason_components else "Sin señales fuertes"
            out = {
                "window_end": row.get("window_end"),
                "risk_30m": round(risk_30m, 4),
                "alert_level_30m": alert_level,
                "reason_30m": reason_text,
            }
            out["risk_steps"] = None if risk_steps_val is None else round(risk_steps_val, 4)
            out["risk_sedentary"] = None if risk_sedentary_val is None else round(risk_sedentary_val, 4)
            out["risk_sleep"] = None if risk_sleep_val is None else round(risk_sleep_val, 4)
            alerts.append(out)
            alerts_roll_rows.append(
                {
                    "window_end": row.get("window_end"),
                    "risk_30m": risk_30m if comp_count > 0 else None,
                    "alert_level_30m": alert_level,
                }
            )

        alerts_roll_df = pd.DataFrame(alerts_roll_rows)
        if not alerts_roll_df.empty:
            alerts_roll_df["window_end"] = pd.to_datetime(alerts_roll_df.get("window_end"), errors="coerce")
            alerts_roll_df["risk_30m"] = pd.to_numeric(alerts_roll_df.get("risk_30m"), errors="coerce")
            alerts_roll_df["alert_level_30m"] = pd.to_numeric(alerts_roll_df.get("alert_level_30m"), errors="coerce")
            alerts_roll_df = alerts_roll_df.dropna(subset=["window_end"])

        combined_df = pd.concat([history_df, alerts_roll_df], ignore_index=True)
        metrics_by_window: Dict[str, Any] = {}

        if not combined_df.empty:
            combined_df["window_end"] = pd.to_datetime(combined_df.get("window_end"), errors="coerce")
            combined_df = combined_df.dropna(subset=["window_end"])

            if not combined_df.empty:
                combined_df["risk_30m"] = pd.to_numeric(combined_df.get("risk_30m"), errors="coerce")
                combined_df["alert_level_30m"] = pd.to_numeric(combined_df.get("alert_level_30m"), errors="coerce")
                combined_df = combined_df.sort_values("window_end").set_index("window_end")

                combined_df["risk_n_3d"] = combined_df["risk_30m"].rolling("3D", min_periods=1).count()
                combined_df["risk_n_7d"] = combined_df["risk_30m"].rolling("7D", min_periods=1).count()
                combined_df["risk_3d"] = combined_df["risk_30m"].rolling("3D", min_periods=1).mean()
                combined_df["risk_7d"] = combined_df["risk_30m"].rolling("7D", min_periods=1).mean()

                combined_df["alert_level_3d"] = _alert_levels_from_risk(
                    combined_df["risk_3d"],
                    combined_df["risk_n_3d"],
                    min_cov=MIN_COV_3D,
                )
                combined_df["alert_level_7d"] = _alert_levels_from_risk(
                    combined_df["risk_7d"],
                    combined_df["risk_n_7d"],
                    min_cov=MIN_COV_7D,
                )

                avail = combined_df["risk_30m"].notna()
                any_alert = (combined_df["alert_level_30m"] > 0).astype(float)
                red_alert = (combined_df["alert_level_30m"] == 2).astype(float)
                any_alert.loc[~avail] = float("nan")
                red_alert.loc[~avail] = float("nan")

                for horizon, tag, base_cov in (("3D", "3d", MIN_COV_3D), ("7D", "7d", MIN_COV_7D)):
                    n_av = combined_df["risk_30m"].rolling(horizon, min_periods=1).count()
                    rate_any = any_alert.rolling(horizon, min_periods=1).mean()
                    rate_red = red_alert.rolling(horizon, min_periods=1).mean()

                    combined_df[f"persist_rate_any_{tag}"] = rate_any
                    ok = n_av.fillna(0) >= base_cov
                    persist_y = (rate_any >= PERSIST_YELLOW_RATE) & ok
                    persist_r = ((rate_red >= PERSIST_RED_RATE) | (rate_any >= (PERSIST_RED_RATE * 2))) & ok

                    levels = pd.Series(0, index=rate_any.index, dtype="int64")
                    levels.loc[persist_y] = 1
                    levels.loc[persist_r] = 2
                    combined_df[f"persist_level_{tag}"] = levels

                date_str = data_de_hoy.get("Fecha")
                target_date = None
                if isinstance(date_str, str) and date_str:
                    try:
                        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        target_date = None

                if target_date is not None:
                    combined_reset = combined_df.reset_index()
                    today_df = combined_reset.loc[combined_reset["window_end"].dt.date == target_date]
                    metrics_by_window = {
                        row["window_end"].strftime("%Y-%m-%dT%H:%M:%S"): row
                        for _, row in today_df.iterrows()
                    }

        def _as_number(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            return num if math.isfinite(num) else None

        defaults = {
            "risk_3d": None,
            "risk_7d": None,
            "alert_level_3d": None,
            "alert_level_7d": None,
            "persist_rate_any_3d": None,
            "persist_rate_any_7d": None,
            "persist_level_3d": None,
            "persist_level_7d": None,
        }

        for alert in alerts:
            alert.update(defaults)
            metrics = metrics_by_window.get(alert.get("window_end"))
            if metrics is None:
                continue
            risk_3d_val = _as_number(metrics.get("risk_3d"))
            risk_7d_val = _as_number(metrics.get("risk_7d"))
            rate_3d_val = _as_number(metrics.get("persist_rate_any_3d"))
            rate_7d_val = _as_number(metrics.get("persist_rate_any_7d"))
            level_3d_val = _as_number(metrics.get("alert_level_3d"))
            level_7d_val = _as_number(metrics.get("alert_level_7d"))
            persist_3d_val = _as_number(metrics.get("persist_level_3d"))
            persist_7d_val = _as_number(metrics.get("persist_level_7d"))

            alert["risk_3d"] = None if risk_3d_val is None else round(risk_3d_val, 4)
            alert["risk_7d"] = None if risk_7d_val is None else round(risk_7d_val, 4)
            alert["persist_rate_any_3d"] = None if rate_3d_val is None else round(rate_3d_val, 4)
            alert["persist_rate_any_7d"] = None if rate_7d_val is None else round(rate_7d_val, 4)

            alert["alert_level_3d"] = None if level_3d_val is None else int(level_3d_val)
            alert["alert_level_7d"] = None if level_7d_val is None else int(level_7d_val)
            alert["persist_level_3d"] = None if persist_3d_val is None else int(persist_3d_val)
            alert["persist_level_7d"] = None if persist_7d_val is None else int(persist_7d_val)

        data_de_hoy["Alertas_Intradia"] = alerts
        return data_de_hoy
    except Exception:
        logger.exception("Error evaluando alertas intradía para %s", client_id)
        data_de_hoy["Alertas_Intradia"] = []
        return data_de_hoy

__all__ = ["actualizar_baselines_historicos", "evaluar_hora_actual"]
