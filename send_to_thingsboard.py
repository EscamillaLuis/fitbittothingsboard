from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Optional

import paho.mqtt.client as mqtt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPIC = "v1/devices/me/telemetry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Envía datos a ThingsBoard vía MQTT.")
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
            return int(num) if num.is_integer() else num
        except ValueError:
            return None
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict) and "value" in first:
            return _to_number(first["value"])
    if isinstance(value, dict) and "value" in value:
        return _to_number(value["value"])
    return None


def _resolve_respiration(resp_entry: Dict) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    sources = []
    if isinstance(resp_entry, dict):
        sources.append(resp_entry)
        value = resp_entry.get("value")
        if isinstance(value, dict):
            sources.append(value)
            full = value.get("fullSleepSummary")
            if isinstance(full, dict):
                sources.append(full)
    for source in sources:
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
    metrics = {}
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

    proxy = data.get("HRV_Proxy", {})
    if isinstance(proxy, dict):
        for key, alias in (
            ("rmssd_proxy_ms", "HRV_proxy_rmssd_ms"),
            ("sdnn_proxy_ms", "HRV_proxy_sdnn_ms"),
            ("pnn50_proxy", "HRV_proxy_pnn50"),
        ):
            metric = proxy.get(key)
            if isinstance(metric, (int, float)):
                values[alias] = metric

    sleep_list = data.get("Resumen_Sueño")
    if isinstance(sleep_list, list) and sleep_list:
        values.update(_resolve_sleep_details(sleep_list[0]))

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


def _add_sample(buckets, seconds: int, metric: str, value: float, window_seconds: int) -> None:
    if not isinstance(value, (int, float)):
        return
    idx = seconds // window_seconds
    buckets[idx][metric].append(value)


def _time_to_seconds(time_str: str) -> Optional[int]:
    if not isinstance(time_str, str):
        return None
    for fmt in ("%H:%M:%S", "%H:%M:%S.%f"):
        try:
            t_obj = dt.datetime.strptime(time_str, fmt).time()
            return t_obj.hour * 3600 + t_obj.minute * 60 + t_obj.second
        except ValueError:
            continue
    return None

def _iso_to_seconds(value: str) -> Optional[int]:
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
    return dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second


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



def generate_time_series_payloads(data: Dict, window_seconds: int, usuario: str) -> List[Dict]:
    buckets: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    date_str = data.get("Fecha")
    if not date_str:
        return []
    day = dt.datetime.strptime(date_str, "%Y-%m-%d")

    heart = data.get("Ritmo_Cardiaco", {})
    dataset = None
    if isinstance(heart, dict):
        dataset = heart.get("dataset") or heart.get("activities-heart-intraday", {}).get("dataset")
    for entry in dataset or []:
        seconds = _time_to_seconds(entry.get("time"))
        if seconds is None:
            continue
        _add_sample(buckets, seconds, "Ritmo_Cardiaco", entry.get("value"), window_seconds)

    minutes = []
    intraday = data.get("HRV_intraday")
    if isinstance(intraday, dict):
        minutes = intraday.get("minutes", []) or []
    else:
        hrv_list = data.get("HRV")
        if isinstance(hrv_list, list):
            for entry in hrv_list:
                for minute in entry.get("minutes", []):
                    value = minute.get("value", {})
                    minutes.append({"time": minute.get("minute"), "rmssd": value.get("rmssd")})
    for minute in minutes:
        minute_ts = minute.get("time")
        seconds = _iso_to_seconds(minute_ts)
        if seconds is None:
            continue
        _add_sample(buckets, seconds, "HRV_RMSSD", minute.get("rmssd"), window_seconds)

    actividades = data.get("Actividades", {})
    if isinstance(actividades, dict):
        # Intraday activity metrics (steps, calories, distance, elevation) intentionally ignored.
        pass

    for entry in data.get("SpO2", []) or []:
        minute = entry.get("minute")
        seconds = _iso_to_seconds(minute)
        if seconds is None:
            continue
        _add_sample(buckets, seconds, "SpO2", entry.get("value"), window_seconds)

    payloads: List[Dict] = []
    start_of_day = dt.datetime.combine(day.date(), dt.time())
    for bucket_idx in sorted(buckets):
        ts = int((start_of_day + dt.timedelta(seconds=bucket_idx * window_seconds)).timestamp() * 1000)
        values = {"Usuario": usuario}
        for metric, samples in buckets[bucket_idx].items():
            if not samples:
                continue
            # Skip intraday metrics that must not be forwarded to ThingsBoard.
            if metric in {"calories", "distance", "elevation", "steps"}:
                continue
            values[metric] = mean(samples)
        payloads.append({"ts": ts, "values": values})
    return payloads


def generate_hrv_proxy_time_series_payloads(data: Dict, usuario: str) -> List[Dict]:
    series = data.get("HRV_Proxy_Series")
    if not isinstance(series, list):
        return []
    payloads: List[Dict] = []
    for entry in series:
        end_iso = entry.get("window_end")
        if not end_iso:
            continue
        parsed = _parse_iso_datetime(end_iso)
        if parsed is None:
            continue
        timestamp = int(parsed.timestamp() * 1000)
        values = {"Usuario": usuario}
        for src, dst in (
            ("rmssd_proxy_ms", "HRV_PROXY_RMSSD"),
            ("sdnn_proxy_ms", "HRV_PROXY_SDNN"),
            ("pnn50_proxy", "HRV_PROXY_PNN50"),
            ("n", "HRV_PROXY_N"),
        ):
            metric = entry.get(src)
            if isinstance(metric, (int, float)):
                values[dst] = metric
        payloads.append({"ts": timestamp, "values": values})
    return payloads


def mqtt_publish(host: str, port: int, token: str, payloads: Iterable[Dict]) -> None:
    payloads = list(payloads)
    if not payloads:
        return
    client = mqtt.Client()
    client.username_pw_set(token)
    client.connect(host, port)
    client.loop_start()
    try:
        for payload in payloads:
            client.publish(TOPIC, json.dumps(payload), qos=1)
    finally:
        client.loop_stop()
        client.disconnect()
