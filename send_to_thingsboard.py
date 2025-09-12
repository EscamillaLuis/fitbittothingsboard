# send_to_thingsboard.py
import os
import json
import time
import argparse
import datetime
import paho.mqtt.client as mqtt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Envía datos a ThingsBoard vía MQTT.")
    parser.add_argument("--mqtt-host", default=os.getenv("THINGSBOARD_MQTT_HOST", "thingsboard.cloud"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--window", type=int, default=int(os.getenv("AGG_WINDOW", "300")))
    return parser.parse_args()

def mqtt_publish(host, port, token, payloads):
    client = mqtt.Client()
    client.username_pw_set(token)
    client.connect(host, port)
    client.loop_start()
    topic = "v1/devices/me/telemetry"
    for p in payloads:
        client.publish(topic, json.dumps(p), qos=1)
        time.sleep(0.05)
    client.loop_stop()
    client.disconnect()

def generate_static_payload(data, usuario, use_current_ts: bool = False):
    """Build a telemetry payload with static daily metrics.

    Parameters
    ----------
    data: dict
        Fitbit data already retrieved for a particular day.
    usuario: str
        Identifier of the user, included in the telemetry values.
    use_current_ts: bool
        If True, the payload timestamp corresponds to the current
        moment. This is useful for monitor mode where the script runs
        several times during the day and we want each update to be
        stored as a new time-series entry. When False (default), the
        timestamp is set to midnight of ``data['Fecha']`` which is the
        desired behaviour when processing historical data.
    """
    
    static_keys = [
        "Edad", "Peso", "Grasa_Corporal", "IMC",
        "Frecuencia_Respiratoria", "Ritmo_Cardiaco_Reposo",
        "Pasos", "Calorias", "Distancia"
    ]
    values = {"Usuario": usuario}
    for key in static_keys:
        val = data.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict) and "value" in val[0]:
            raw = val[0]["value"]
        else:
            raw = val
        try:
            num = float(raw)
            values[key] = int(num) if num.is_integer() else num
        except Exception:
            continue
    
    resp_list = data.get("Frecuencia_Respiratoria", [])
    if isinstance(resp_list, list) and resp_list:
        full = resp_list[0].get("value", {}).get("fullSleepSummary", {})
        rate = full.get("breathingRate")
        if isinstance(rate, (int, float)):
            values["Frecuencia_Respiratoria"] = rate
        for stage in full.get("breathingRateData", []):
            lvl = stage.get("level")
            br = stage.get("breathingRate")
            if lvl in {"light", "deep", "rem"} and isinstance(br, (int, float)):
                values[f"Frecuencia_Respiratoria_{lvl}"] = br

    # HRV summary or intraday
    hrv_list = data.get("HRV", [])
    if isinstance(hrv_list, list) and hrv_list:
        entry = hrv_list[0]
        value = entry.get("value", {})
        daily = value.get("dailyRmssd")
        deep = value.get("deepRmssd")
        if isinstance(daily, (int, float)):
            values["HRV_dailyRmssd"] = daily
        if isinstance(deep, (int, float)):
            values["HRV_deepRmssd"] = deep
        rmssd_vals = []
        for minute in entry.get("minutes", []):
            rmssd = minute.get("value", {}).get("rmssd")
            if isinstance(rmssd, (int, float)):
                rmssd_vals.append(rmssd)
        if rmssd_vals and "HRV_dailyRmssd" not in values:
            values["HRV_RMSSD"] = sum(rmssd_vals) / len(rmssd_vals)
        if rmssd_vals:
            values["HRV_RMSSD"] = sum(rmssd_vals) / len(rmssd_vals)        
    sleep_list = data.get("Resumen_Sueño", [])
    if isinstance(sleep_list, list) and sleep_list:
        sleep = sleep_list[0]
        summary = sleep.get("levels", {}).get("summary", {})
        for stage in ("deep", "light", "rem", "wake"):
            mins = summary.get(stage, {}).get("minutes")
            if mins is not None:
                values[f"sueño_{stage}_min"] = mins
        if "minutesAsleep" in sleep:
            values["minutos_dormidos"] = sleep["minutesAsleep"]
        if "timeInBed" in sleep:
            values["minutos_en_cama"] = sleep["timeInBed"]
        
        levels_data = sleep.get("levels", {}).get("data", [])
        if isinstance(levels_data, list):
            ciclos = sum(1 for seg in levels_data if seg.get("level") == "deep")
            values["ciclos_sueño"] = ciclos

    act = data.get("Resumen_Actividades", {})
    if isinstance(act, dict):
        for k in (
            "sedentaryMinutes", "lightlyActiveMinutes",
            "fairlyActiveMinutes", "veryActiveMinutes"
        ):
            v = act.get(k)
            if isinstance(v, (int, float)):
                values[k] = v
                
    if len(values) <= 1:
        return None
    
    if use_current_ts:
        ts = int(datetime.datetime.now().timestamp() * 1000)
    else:
        date_obj = datetime.datetime.strptime(data.get("Fecha"), "%Y-%m-%d").date()
        ts = int(datetime.datetime.combine(date_obj, datetime.time()).timestamp() * 1000)

    return {"ts": ts, "values": values}

def generate_time_series_payloads(data, window_seconds, usuario):
    buckets = {}
    date_obj = datetime.datetime.strptime(data.get("Fecha"), "%Y-%m-%d").date()

    if isinstance(data.get("Ritmo_Cardiaco"), dict):
        for entry in data["Ritmo_Cardiaco"].get("dataset", []):
            t = entry.get("time")
            v = entry.get("value")
            if not t or not isinstance(v, (int, float)):
                continue
            tm = datetime.datetime.strptime(t, "%H:%M:%S").time()
            seconds = tm.hour * 3600 + tm.minute * 60 + tm.second
            idx = seconds // window_seconds
            buckets.setdefault(idx, {}).setdefault("Ritmo_Cardiaco", []).append(v)
            
    hrv_list = data.get("HRV", [])
    if isinstance(hrv_list, list):
        for entry in hrv_list:
            for minute in entry.get("minutes", []):
                minute_ts = minute.get("minute")
                rmssd = minute.get("value", {}).get("rmssd")
                if not minute_ts or not isinstance(rmssd, (int, float)):
                    continue
                dt = datetime.datetime.fromisoformat(minute_ts)
                seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
                idx = seconds // window_seconds
                buckets.setdefault(idx, {}).setdefault("HRV_RMSSD", []).append(rmssd)
                
    actividades = data.get("Actividades", {})
    if isinstance(actividades, dict) and isinstance(actividades.get("steps"), list):
        for entry in actividades["steps"]:
            t = entry.get("time")
            v = entry.get("value")
            if not t or not isinstance(v, (int, float)):
                continue
            tm = datetime.datetime.strptime(t, "%H:%M:%S").time()
            seconds = tm.hour * 3600 + tm.minute * 60 + tm.second
            idx = seconds // window_seconds
            buckets.setdefault(idx, {}).setdefault("steps", []).append(v)

    spo2_list = data.get("SpO2", [])
    if isinstance(spo2_list, list):
        for entry in spo2_list:
            minute = entry.get("minute")
            v = entry.get("value")
            if not minute or not isinstance(v, (int, float)):
                continue
            dt = datetime.datetime.fromisoformat(minute)
            seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
            idx = seconds // window_seconds
            buckets.setdefault(idx, {}).setdefault("SpO2", []).append(v)

    payloads = []
    for idx in sorted(buckets):
        window_start = datetime.datetime.combine(date_obj, datetime.time()) + datetime.timedelta(seconds=idx * window_seconds)
        ts = int(window_start.timestamp() * 1000)
        values = {"Usuario": usuario}
        for metric, vals in buckets[idx].items():
            values[metric] = round(sum(vals) / len(vals))
        payloads.append({"ts": ts, "values": values})

    return payloads



'''def main():
    args = parse_args()
    client_id = data.get("ID_Cliente") or data.get("ID_Usuario")
    with open("thingsboard_tokens.json", "r") as f:
        token_map = json.load(f)
    token = token_map.get(client_id)
    if not token:
        print(f"No se encontró token para el client_id: {client_id}")
        return
    path = args.json_file if os.path.isabs(args.json_file) else os.path.join(os.getcwd(), args.json_file)
    if not os.path.isfile(path):
        alt = os.path.join(SCRIPT_DIR, args.json_file)
        path = alt if os.path.isfile(alt) else path
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    payloads = []
    static = generate_static_payload(data)
    if static:
        payloads.append(static)
    payloads.extend(generate_time_series_payloads(data, args.window))
    if not payloads:
        print("No hay datos para enviar.")
        return
    mqtt_publish(args.mqtt_host, args.mqtt_port, token, payloads)
    print(f"Se enviaron {len(payloads)} mensajes.")
'''

#if __name__ == "__main__":
    #main()
