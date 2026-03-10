import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from analisis.motor_alertas_vivo import actualizar_baselines_historicos, evaluar_hora_actual
from send_to_thingsboard import generate_time_series_payloads, mqtt_publish
from fitbit_service import DATA_DIR


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TOKENS_FILE = SCRIPT_DIR / "thingsboard_tokens.json"
TB_HOST = "thingsboard.cloud"
TB_PORT = 1883
WINDOW_SECONDS = 60
DAYS_BACK = 14
PUBLISH_SLEEP_SECONDS = 0.5


def _load_thingsboard_tokens(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return payload
        print(f"Formato invalido en {path}: se esperaba un objeto JSON.")
        return {}
    except Exception as exc:
        print(f"No se pudo leer {path}: {exc}")
        return {}


def _resolve_json_path(data_dir: Path, client_id: str, date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")
    nested_path = (
        data_dir
        / client_id
        / date_obj.strftime("%Y")
        / date_obj.strftime("%m")
        / date_obj.strftime("%d")
        / f"{date_str}_{client_id}.json"
    )
    if nested_path.exists():
        return nested_path
    flat_path = data_dir / client_id / f"{date_str}_{client_id}.json"
    if flat_path.exists():
        return flat_path
    return None


def main() -> None:
    tokens = _load_thingsboard_tokens(TOKENS_FILE)
    if not tokens:
        print("No hay tokens de ThingsBoard para procesar.")
        return

    actualizar_baselines_historicos(DATA_DIR)

    today = datetime.now().date()
    start_date = today - timedelta(days=DAYS_BACK)

    data_dir = Path(DATA_DIR)

    for client_id, tb_info in tokens.items():
        if not isinstance(tb_info, dict):
            print(f"Tokens invalidos para {client_id}; se omite.")
            continue

        tb_token = tb_info.get("token")
        tb_usuario = tb_info.get("usuario")
        if not tb_token or tb_usuario is None:
            print(f"Faltan datos de token/usuario para {client_id}; se omite.")
            continue

        current_date = start_date
        while current_date <= today:
            json_path = _resolve_json_path(data_dir, client_id, current_date)
            if json_path is None:
                current_date += timedelta(days=1)
                continue

            try:
                with json_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:
                print(f"Error leyendo {json_path}: {exc}")
                current_date += timedelta(days=1)
                continue

            if not isinstance(data, dict):
                print(f"Datos invalidos en {json_path}; se omite.")
                current_date += timedelta(days=1)
                continue

            try:
                data = evaluar_hora_actual(client_id, data)
            except Exception as exc:
                print(f"Error evaluando alertas para {client_id} {current_date}: {exc}")
                current_date += timedelta(days=1)
                continue

            payloads = generate_time_series_payloads(
                data,
                window_seconds=WINDOW_SECONDS,
                usuario=tb_usuario,
            )

            if payloads:
                try:
                    mqtt_publish(TB_HOST, TB_PORT, tb_token, payloads)
                    date_str = current_date.strftime("%Y-%m-%d")
                    print(f"Enviados {len(payloads)} payloads para {client_id} en {date_str}")
                except Exception as exc:
                    date_str = current_date.strftime("%Y-%m-%d")
                    print(f"Error publicando {client_id} {date_str}: {exc}")
                finally:
                    time.sleep(PUBLISH_SLEEP_SECONDS)

            current_date += timedelta(days=1)


if __name__ == "__main__":
    main()
