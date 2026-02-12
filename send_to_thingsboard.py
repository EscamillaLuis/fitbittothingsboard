from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import logging
from collections import defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import paho.mqtt.client as mqtt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPIC = "v1/devices/me/telemetry"
DEFAULT_TIMEZONE = os.getenv("FITBIT_DEFAULT_TIMEZONE", "America/Mexico_City")
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Envia datos a ThingsBoard via MQTT.")
    parser.add_argument("--mqtt-host", default=os.getenv("THINGSBOARD_MQTT_HOST", "thingsboard.cloud"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--window", type=int, default=int(os.getenv("AGG_WINDOW", "300")))
    return parser.parse_args()
def _to_number(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            num = float(value)
        except ValueError:
            return None
        return int(num) if num.is_integer() else num
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict) and "value" in first:
            return _to_number(first["value"])
    if isinstance(value, dict) and "value" in value:
        return _to_number(value["value"])
    return None
def _resolve_respiration(resp_entry: Dict) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    stacks = []
    if isinstance(resp_entry, dict):
        stacks.append(resp_entry)
        value = resp_entry.get("value")
        if isinstance(value, dict):
            stacks.append(value)
            full = value.get("fullSleepSummary")
            if isinstance(full, dict):
                stacks.append(full)
    for source in stacks:
        rate = source.get("breathingRate")
        if isinstance(rate, (int, float)):
            metrics["Frecuencia_Respiratoria"] = rate
        stages = source.get("breathingRateData")
        if isinstance(stages, list):
            for stage in stages:
                lvl = stage.get("level")
                br = stage.get("breathingRate")
                if lvl in {"light", "deep", "rem"} and isinstance(br, (int, float)):
                    metrics[f"Frecuencia_Respiratoria_{lvl}"] = br
    return metrics
def _resolve_sleep_details(sleep_entry: Dict) -> Dict[str, float]:
    values: Dict[str, float] = {}
    summary = sleep_entry.get("levels", {}).get("summary", {})
    for stage in ("deep", "light", "rem", "wake"):
        mins = summary.get(stage, {}).get("minutes")
        if isinstance(mins, (int, float)):
            values[f"sueño_{stage}_min"] = mins
    if isinstance(sleep_entry.get("minutesAsleep"), (int, float)):
        values["minutos_dormidos"] = sleep_entry["minutesAsleep"]
    if isinstance(sleep_entry.get("timeInBed"), (int, float)):
        values["minutos_en_cama"] = sleep_entry["timeInBed"]
    levels_data = sleep_entry.get("levels", {}).get("data", []) or []
    values["ciclos_sueño"] = sum(1 for seg in levels_data if seg.get("level") == "deep")
    return values
def _resolve_intraday_hrv(intraday: Dict) -> Dict[str, float]:
    minutes = intraday.get("minutes", []) if isinstance(intraday, dict) else []
    rmssd_vals = [m.get("rmssd") for m in minutes if isinstance(m.get("rmssd"), (int, float))]
    cov_vals = [m.get("coverage") for m in minutes if isinstance(m.get("coverage"), (int, float))]
    lf_vals = [m.get("lf") for m in minutes if isinstance(m.get("lf"), (int, float))]
    hf_vals = [m.get("hf") for m in minutes if isinstance(m.get("hf"), (int, float))]
    metrics: Dict[str, float] = {}
    if rmssd_vals:
        metrics["HRV_intraday_rmssd_mean"] = mean(rmssd_vals)
    if cov_vals:
        metrics["HRV_intraday_coverage_mean"] = mean(cov_vals)
    if lf_vals:
        metrics["HRV_intraday_lf_median"] = median(lf_vals)
    if hf_vals:
        metrics["HRV_intraday_hf_median"] = median(hf_vals)
    return metrics
def generate_static_payload(data: Dict, usuario: str, use_current_ts: bool = False) -> Optional[Dict]:
    base_metrics = {
        key: _to_number(data.get(key))
        for key in (
            "Edad",
            "Peso",
            "Grasa_Corporal",
            "IMC",
            "Frecuencia_Respiratoria",
            "Ritmo_Cardiaco_Reposo",
            "Pasos",
            "Calorias",
            "Distancia",
        )
    }
    values = {"Usuario": usuario}
    values.update({k: v for k, v in base_metrics.items() if isinstance(v, (int, float))})
    resp_list = data.get("Frecuencia_Respiratoria")
    if isinstance(resp_list, list) and resp_list:
        values.update(_resolve_respiration(resp_list[0]))
    hrv_list = data.get("HRV")
    if isinstance(hrv_list, list) and hrv_list:
        value = (hrv_list[0] or {}).get("value", {})
        for key in ("dailyRmssd", "deepRmssd"):
            metric = value.get(key)
            if isinstance(metric, (int, float)):
                values[f"HRV_{key}"] = metric
    values.update(_resolve_intraday_hrv(data.get("HRV_intraday") or {}))
    sleep_list = data.get("Resumen_Sueño")
    if not isinstance(sleep_list, list) or not sleep_list:
        sleep_list = data.get("Resumen_Sueño")
    if isinstance(sleep_list, list) and sleep_list:
        values.update(_resolve_sleep_details(sleep_list[0]))
    spo2_entry = data.get("SpO2")
    spo2_value = None
    if isinstance(spo2_entry, dict):
        candidate = spo2_entry.get("value", spo2_entry)
        if isinstance(candidate, dict):
            spo2_value = candidate
    elif isinstance(spo2_entry, list) and spo2_entry:
        first_entry = spo2_entry[0] or {}
        candidate = first_entry.get("value", first_entry)
        if isinstance(candidate, dict):
            spo2_value = candidate
    if isinstance(spo2_value, dict):
        for key, alias in (("avg", "SpO2_avg"), ("min", "SpO2_min"), ("max", "SpO2_max")):
            metric = _to_number(spo2_value.get(key))
            if isinstance(metric, (int, float)):
                values[alias] = metric
    actividad = data.get("Resumen_Actividades", {})
    if isinstance(actividad, dict):
        for key in (
            "sedentaryMinutes",
            "lightlyActiveMinutes",
            "fairlyActiveMinutes",
            "veryActiveMinutes",
        ):
            metric = actividad.get(key)
            if isinstance(metric, (int, float)):
                values[key] = metric
    if len(values) == 1:
        return None
    if use_current_ts:
        timestamp = int(dt.datetime.now().timestamp() * 1000)
    else:
        date_str = data.get("Fecha")
        if not date_str:
            return None
        day = dt.datetime.strptime(date_str, "%Y-%m-%d")
        timestamp = int(dt.datetime.combine(day.date(), dt.time()).timestamp() * 1000)
    return {"ts": timestamp, "values": values}
def _add_sample(buckets, seconds: float, metric: str, value: float, window_seconds: int) -> None:
    if not isinstance(value, (int, float)):
        return
    if window_seconds <= 0:
        return
    seconds_int = int(seconds)
    clamped_seconds = max(0, min(seconds_int, 86399))
    bucket_start = (clamped_seconds // window_seconds) * window_seconds
    buckets[bucket_start][metric].append(float(value))
def _emit_bucket_payloads(
    buckets: Dict[int, Dict[str, List[float]]],
    base: dt.datetime,
    window_seconds: int,
    usuario: str,
) -> List[Dict]:
    if window_seconds <= 0:
        return []
    payloads: List[Dict] = []
    for bucket_start_seconds in sorted(buckets):
        bucket_start_dt_local = base + dt.timedelta(seconds=int(bucket_start_seconds))
        ts = int(round(bucket_start_dt_local.astimezone(dt.timezone.utc).timestamp() * 1000.0))
        values = {"Usuario": usuario}
        for metric, samples in buckets[bucket_start_seconds].items():
            if not samples:
                continue
            if metric in {"calories", "distance", "elevation", "steps"}:
                continue
            avg = mean(samples)
            if metric in {"Ritmo_Cardiaco", "SpO2"}:
                values[metric] = int(round(avg))
            else:
                values[metric] = avg
        if len(values) > 1:
            payloads.append({"ts": ts, "values": values})
    return payloads
def _time_to_seconds(time_str: str) -> Optional[float]:
    if not isinstance(time_str, str):
        return None
    raw = time_str.strip()
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            t_obj = dt.datetime.strptime(raw, fmt).time()
            base_seconds = (t_obj.hour * 3600) + (t_obj.minute * 60) + t_obj.second
            return float(base_seconds) + (t_obj.microsecond / 1_000_000.0)
        except ValueError:
            continue
    return None
def _iso_to_seconds(value: str) -> Optional[float]:
    if not isinstance(value, str):
        return None
    seconds = _time_to_seconds(value)
    if seconds is not None:
        return seconds
    iso_value = value.rstrip()
    if iso_value.endswith('Z'):
        iso_value = iso_value[:-1] + '+00:00'
    try:
        dt_obj = dt.datetime.fromisoformat(iso_value)
    except ValueError:
        return None
    if dt_obj.tzinfo is not None:
        dt_obj = dt_obj.astimezone(dt.timezone.utc)
    return (
        dt_obj.hour * 3600
        + dt_obj.minute * 60
        + dt_obj.second
        + dt_obj.microsecond / 1_000_000
    )
def _iso_to_seconds_local(value: str, tzname: Optional[str]) -> Optional[float]:
    if not isinstance(value, str):
        return None
    seconds = _time_to_seconds(value)
    if seconds is not None:
        return seconds
    iso_value = value.rstrip()
    if iso_value.endswith('Z'):
        iso_value = iso_value[:-1] + '+00:00'
    try:
        dt_obj = dt.datetime.fromisoformat(iso_value)
    except ValueError:
        return None
    tzinfo = _resolve_zoneinfo(tzname)
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=tzinfo)
    else:
        dt_obj = dt_obj.astimezone(tzinfo)
    midnight = dt_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    return (dt_obj - midnight).total_seconds()
def _parse_iso_datetime(value: str) -> Optional[dt.datetime]:
    if not isinstance(value, str):
        return None
    iso_value = value.rstrip()
    if iso_value.endswith('Z'):
        iso_value = iso_value[:-1] + '+00:00'
    try:
        dt_obj = dt.datetime.fromisoformat(iso_value)
    except ValueError:
        return None
    if dt_obj.tzinfo is not None:
        return dt_obj.astimezone(dt.timezone.utc)
    return dt_obj
def _resolve_zoneinfo(tzname: Optional[str]) -> ZoneInfo:
    tz_candidate = tzname or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(tz_candidate)
    except ZoneInfoNotFoundError:
        if tz_candidate != DEFAULT_TIMEZONE:
            logger.warning("Zona horaria desconocida '%s'. Usando '%s'.", tz_candidate, DEFAULT_TIMEZONE)
        return ZoneInfo(DEFAULT_TIMEZONE)
def _aware_dt(date_obj: dt.date, tzname: Optional[str]) -> dt.datetime:
    tzinfo = _resolve_zoneinfo(tzname)
    return dt.datetime.combine(date_obj, dt.time(), tzinfo=tzinfo)
def _minute_to_ts(minute_str: str, tzname: Optional[str]) -> Optional[int]:
    dt_obj = _parse_iso_datetime(minute_str)
    if dt_obj is None:
        return None
    if dt_obj.tzinfo is None:
        tzinfo = _resolve_zoneinfo(tzname)
        dt_obj = dt_obj.replace(tzinfo=tzinfo)
    utc_dt = dt_obj.astimezone(dt.timezone.utc)
    return int(utc_dt.timestamp() * 1000)
def _format_ts(ts_ms: Optional[int]) -> str:
    if ts_ms is None:
        return '-'
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).isoformat()
def generate_time_series_payloads(data: Dict, window_seconds: int, usuario: str) -> List[Dict]:
    raw_mode = window_seconds <= 0
    aggregate_mode = window_seconds > 0
    bucket_window = window_seconds if aggregate_mode else 60
    buckets: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    date_str = data.get("Fecha")
    if not date_str:
        return []
    day = dt.datetime.strptime(date_str, "%Y-%m-%d")
    base = _aware_dt(day.date(), data.get("Timezone"))
    tzinfo = base.tzinfo or dt.timezone.utc
    tz_name = data.get("Timezone")
    n_hr_samples = 0
    n_spo2_samples = 0
    hr_raw_payloads: List[Dict] = []
    hrv_raw_payloads: List[Dict] = []
    spo2_raw_payloads: List[Dict] = []
    first_ts_ms: Optional[int] = None
    last_ts_ms: Optional[int] = None
    def _track_ts(ts_ms: Optional[int]) -> None:
        nonlocal first_ts_ms, last_ts_ms
        if ts_ms is None:
            return
        if first_ts_ms is None or ts_ms < first_ts_ms:
            first_ts_ms = ts_ms
        if last_ts_ms is None or ts_ms > last_ts_ms:
            last_ts_ms = ts_ms
    heart = data.get("Ritmo_Cardiaco", {})
    dataset = None
    if isinstance(heart, dict):
        dataset = heart.get("activities-heart-intraday", {}).get("dataset")
        if dataset is None:
            dataset = heart.get("dataset")
    if not isinstance(dataset, list):
        dataset = []
    for entry in dataset:
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if not isinstance(value, (int, float)):
            continue
        seconds = _time_to_seconds(entry.get("time"))
        if seconds is None:
            continue
        sample_dt_local = base + dt.timedelta(seconds=seconds)
        sample_ts_ms = int(round(sample_dt_local.astimezone(dt.timezone.utc).timestamp() * 1000.0))
        _track_ts(sample_ts_ms)
        n_hr_samples += 1
        if aggregate_mode:
            _add_sample(buckets, seconds, "Ritmo_Cardiaco", value, bucket_window)
        else:
            hr_raw_payloads.append({"ts": sample_ts_ms, "values": {"Usuario": usuario, "Ritmo_Cardiaco": value}})
    minutes: List[Dict] = []
    intraday = data.get("HRV_intraday")
    if isinstance(intraday, dict):
        intraday_minutes = intraday.get("minutes")
        if isinstance(intraday_minutes, list):
            minutes.extend(m for m in intraday_minutes if isinstance(m, dict))
    else:
        hrv_list = data.get("HRV")
        if isinstance(hrv_list, list):
            for entry in hrv_list:
                if not isinstance(entry, dict):
                    continue
                entry_minutes = entry.get("minutes")
                if not isinstance(entry_minutes, list):
                    continue
                for minute in entry_minutes:
                    if not isinstance(minute, dict):
                        continue
                    value = minute.get("value")
                    value_dict = value if isinstance(value, dict) else {}
                    minutes.append({"time": minute.get("minute"), "rmssd": value_dict.get("rmssd")})
    for minute in minutes:
        if not isinstance(minute, dict):
            continue
        minute_ts = minute.get("time")
        seconds = _iso_to_seconds_local(minute_ts, tz_name)
        if seconds is None:
            continue
        rmssd = minute.get("rmssd")
        if aggregate_mode:
            _add_sample(buckets, seconds, "HRV_RMSSD", rmssd, bucket_window)
        elif isinstance(rmssd, (int, float)):
            ts_ms = _minute_to_ts(minute_ts, tz_name)
            if ts_ms is None:
                sample_dt_local = base + dt.timedelta(seconds=seconds)
                ts_ms = int(round(sample_dt_local.astimezone(dt.timezone.utc).timestamp() * 1000.0))
            _track_ts(ts_ms)
            hrv_raw_payloads.append({"ts": ts_ms, "values": {"Usuario": usuario, "HRV_RMSSD": float(rmssd)}})
    actividades = data.get("Actividades", {})
    if isinstance(actividades, dict):
        pass
    spo2_entries = data.get("SpO2")
    spo2_minutes: List[Dict] = []
    if isinstance(spo2_entries, list):
        spo2_minutes = [entry for entry in spo2_entries if isinstance(entry, dict)]
    elif isinstance(spo2_entries, dict):
        minutes_block = spo2_entries.get("minutes")
        if isinstance(minutes_block, list):
            spo2_minutes = [entry for entry in minutes_block if isinstance(entry, dict)]
    elif isinstance(spo2_entries, str):
        logger.info("SpO2 %s: %s", date_str, spo2_entries)
    for entry in spo2_minutes:
        minute_str = entry.get("minute")
        value = entry.get("value")
        if not isinstance(minute_str, str) or not isinstance(value, (int, float)):
            continue
        minute_local = None
        minute_iso = minute_str.rstrip()
        if minute_iso.endswith('Z'):
            minute_iso = minute_iso[:-1] + '+00:00'
        try:
            minute_dt = dt.datetime.fromisoformat(minute_iso)
        except ValueError:
            continue
        if minute_dt.tzinfo is None:
            minute_local = minute_dt.replace(tzinfo=tzinfo)
        else:
            minute_local = minute_dt.astimezone(tzinfo)
        ts_ms = int(minute_local.astimezone(dt.timezone.utc).timestamp() * 1000)
        offset_seconds = int((minute_local - base).total_seconds())
        offset_seconds = max(0, min(offset_seconds, 86399))
        _track_ts(ts_ms)
        n_spo2_samples += 1
        if aggregate_mode:
            _add_sample(buckets, offset_seconds, "SpO2", value, bucket_window)
        else:
            spo2_raw_payloads.append({"ts": ts_ms, "values": {"Usuario": usuario, "SpO2": value}})
    agg_payloads = _emit_bucket_payloads(buckets, base, bucket_window, usuario)
    raw_payloads: List[Dict] = []
    raw_payloads.extend(hr_raw_payloads)
    raw_payloads.extend(hrv_raw_payloads)
    raw_payloads.extend(spo2_raw_payloads)
    if raw_mode:
        raw_payloads.sort(key=lambda item: item["ts"])
        payloads = sorted(agg_payloads + raw_payloads, key=lambda item: item["ts"])
    else:
        payloads = agg_payloads
    logger.debug(
        "Intradia %s -> n_hr_samples=%s, n_spo2_samples=%s, first_ts=%s, last_ts=%s, payloads=%s, window=%s",
        date_str,
        n_hr_samples,
        n_spo2_samples,
        _format_ts(first_ts_ms),
        _format_ts(last_ts_ms),
        len(payloads),
        window_seconds,
    )
    return payloads
def mqtt_publish(host: str, port: int, token: str, payloads: Iterable[Dict]) -> None:
    import time
    from collections import deque
    import json as _json
    import os as _os
    msgs_per_sec = int(_os.getenv("TB_MSGS_PER_SEC", "80"))
    batch_size = int(_os.getenv("TB_BATCH_SIZE", "1"))
    keepalive = int(_os.getenv("TB_KEEPALIVE", "60"))
    interval = 1.0 / max(1, msgs_per_sec)
    q = deque()
    try:
        iterator = iter(payloads)
    except TypeError:
        iterator = iter(list(payloads))
    if batch_size > 1:
        batch = []
        for p in iterator:
            batch.append(p)
            if len(batch) >= batch_size:
                q.append(list(batch))
                batch.clear()
        if batch:
            q.append(list(batch))
    else:
        for p in iterator:
            q.append(p)
    if not q:
        return
    client = mqtt.Client()
    client.username_pw_set(token)
    INFLIGHT_MAX = 20
    client.max_inflight_messages_set(INFLIGHT_MAX)
    client.max_queued_messages_set(0)
    pending: set[int] = set()
    pubacked = 0
    def on_publish(_c, _u, mid):
        nonlocal pubacked
        if mid in pending:
            pending.remove(mid)
        pubacked += 1
    client.on_publish = on_publish
    client.connect(host, port, keepalive=keepalive)
    client.loop_start()
    sent = 0
    try:
        next_at = time.perf_counter()
        total_msgs = len(q)
        logger.debug(
            "MQTT publish start: msgs=%s, batch_size=%s, rate=%s/s, inflight<=%s",
            total_msgs, batch_size, msgs_per_sec, INFLIGHT_MAX
        )
        while q:
            item = q.popleft()
            payload_str = _json.dumps(item)
            now = time.perf_counter()
            if now < next_at:
                time.sleep(next_at - now)
            next_at = max(next_at + interval, time.perf_counter())
            info = client.publish(TOPIC, payload_str, qos=1)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                time.sleep(0.25)
                info = client.publish(TOPIC, payload_str, qos=1)
            pending.add(info.mid)
            sent += 1
            while len(pending) >= INFLIGHT_MAX:
                time.sleep(0.01)
        deadline = time.time() + 60
        while pending and time.time() < deadline:
            time.sleep(0.05)
        logger.debug("MQTT publish done: sent=%s, acked=%s, pending=%s", sent, pubacked, len(pending))
    finally:
        client.loop_stop()
        client.disconnect()
