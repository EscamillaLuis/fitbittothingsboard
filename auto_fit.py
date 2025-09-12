import os
import json
import argparse
import shutil
import time
from datetime import datetime, timedelta

from single_fit import (
    get_fitbit_data,
    export_json_to_excel_single,
    debug_print_token_scopes,
    fetch_hrv_data,
    fetch_br_data,
    fetch_hrv_intraday,
    fetch_hrv_intraday_range,
)
from send_to_thingsboard import (
    generate_static_payload,
    generate_time_series_payloads,
    mqtt_publish
)

CREDENTIALS_FILE = "credentials.json"
CLIENT_IDS_FILE = "client_ids.txt"
THINGSBOARD_TOKENS_FILE = "thingsboard_tokens.json"
DATA_DIR = "fitbit_data"
DEFAULT_WINDOW = 60       
DEFAULT_INTERVAL = 3000    
CHUNK_SIZE = 6            


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatiza descarga y envío de datos Fitbit para múltiples pulseras",
    )
    parser.add_argument(
        "--start-date", type=str,
        help="Fecha inicio (YYYY-MM-DD). Por defecto hoy",
    )
    parser.add_argument(
        "--end-date", type=str,
        help="Fecha fin (YYYY-MM-DD). Por defecto start-date",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help="Ventana de agregación (segundos) para series temporales",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Activar monitoreo horario continuo (solo fecha actual)",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help="Intervalo en segundos entre cada lote de descarga o monitoreo",
    )
    parser.add_argument("--check-scopes", type=str, help="Imprime los scopes del token para el usuario dado")
    parser.add_argument("--fetch-hrv", action="store_true", help="Descarga HRV para un usuario")
    parser.add_argument("--fetch-br", action="store_true", help="Descarga tasa respiratoria para un usuario")
    parser.add_argument("--date", type=str, help="Fecha para --fetch-hrv/--fetch-br")
    parser.add_argument("--dates", type=str, help="Rango YYYY-MM-DD:YYYY-MM-DD para backfill HRV intradía")
    parser.add_argument("--intraday", action="store_true", help="Usar endpoint intradía en pruebas")
    parser.add_argument("--user", type=str, help="Usuario para comandos de prueba")
    return parser.parse_args()

def process_single_date(client_id, secret, tb_token, usuario, date_str, window, monitor=False):
    print(f"📅 Procesando {client_id} - {date_str}")
    data = get_fitbit_data(client_id, secret, date_str)
    if not data:
        print(f"❌ No se obtuvieron datos para {client_id} en {date_str}")
        return
    
    year, month, day = date_str.split('-')
    nested_dir = os.path.join(DATA_DIR, client_id, year, month, day)
    os.makedirs(nested_dir, exist_ok=True)

    json_path = os.path.join(nested_dir, f"{date_str}_{client_id}.json")
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(data, jf, indent=4)

    client_root = os.path.join(DATA_DIR, client_id)
    os.makedirs(client_root, exist_ok=True)
    flat_json = os.path.join(client_root, f"{date_str}_{client_id}.json")
    try:
        shutil.copy(json_path, flat_json)
    except Exception as e:
        print(f"⚠️  Error copiando JSON para exportar: {e}")

    try:
        export_json_to_excel_single(client_id, date_str)
        src_excel = os.path.join(client_root, f"{date_str}_{client_id}_procesado.xlsx")
        dest_excel = os.path.join(nested_dir, f"{date_str}_{client_id}_procesado.xlsx")
        os.replace(src_excel, dest_excel)
        print(f"✅ JSON y Excel guardados en: {os.path.relpath(dest_excel)}")
    except Exception as e:
        print(f"⚠️  Error exportando o moviendo Excel ({client_id} - {date_str}): {e}")
    finally:
        if os.path.exists(flat_json):
            os.remove(flat_json)

    payloads = []
    if (static := generate_static_payload(data, usuario, use_current_ts=monitor)):
        payloads.append(static)
    payloads.extend(generate_time_series_payloads(data, window, usuario))

    if payloads:
        try:
            mqtt_publish(
                host="thingsboard.cloud",
                port=1883,
                token=tb_token,
                payloads=payloads
            )
            print(f"✅ Enviados {len(payloads)} payloads a ThingsBoard para {client_id}")
        except Exception as e:
            print(f"❌ Error enviando a ThingsBoard ({client_id}): {e}")
    else:
        print("⚠️ No hay datos para enviar a ThingsBoard")


def monitor_mode(client_ids, cred_map, tb_tokens, window, interval):
    print("🔃 Monitoreo horario activo. Presiona Ctrl+C para detener.")
    try:
        while True:
            today = datetime.now().date().strftime("%Y-%m-%d")
            for client_id in client_ids:
                secret = cred_map.get(client_id)
                rec = tb_tokens.get(client_id, {})
                token, usuario = rec.get('token'), rec.get('usuario')
                if not (secret and token and usuario):
                    print(f"⚠️ Credenciales faltantes para {client_id}")
                    continue
                process_single_date(client_id, secret, token, usuario, today, window, monitor=True)
            print(f"⏳ Esperando {interval} segundos para siguiente monitoreo...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("⏹ Monitoreo detenido por el usuario.")


def main():
    args = parse_args()
    
    if args.check_scopes:
        debug_print_token_scopes(args.check_scopes)
        return

    if args.fetch_hrv or args.fetch_br:
        if not args.user:
            raise ValueError("Debe indicar --user")
        creds = load_json(CREDENTIALS_FILE)
        cred_map = {c['client_id']: c['client_secret'] for c in creds}
        secret = cred_map.get(args.user)
        if not secret:
            print(f"No se encontraron credenciales para {args.user}")
            return
        if args.fetch_hrv:
            if args.intraday:
                if args.dates:
                    start_d, end_d = args.dates.split(":")
                    data = fetch_hrv_intraday_range(args.user, secret, start_d, end_d)
                else:
                    if not args.date:
                        raise ValueError("Debe indicar --date o --dates")
                    data = fetch_hrv_intraday(args.user, secret, args.date)
            else:
                if not args.date:
                    raise ValueError("Debe indicar --date")
                data = fetch_hrv_data(args.user, secret, args.date, intraday=False)
            print(json.dumps(data, indent=2) if data else "Sin datos")
        if args.fetch_br:
            if not args.date:
                raise ValueError("Debe indicar --date para --fetch-br")
            data = fetch_br_data(args.user, secret, args.date, intraday=args.intraday)
            print(json.dumps(data, indent=2) if data else "Sin datos")
        return

    today = datetime.now().date()
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else today
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else start
    if end < start:
        raise ValueError("La fecha fin no puede ser anterior a la fecha inicio")

    creds = load_json(CREDENTIALS_FILE)
    cred_map = {c['client_id']: c['client_secret'] for c in creds}
    tb_tokens = load_json(THINGSBOARD_TOKENS_FILE)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CLIENT_IDS_FILE, 'r', encoding='utf-8') as f:
        client_ids = [l.strip() for l in f if l.strip()]

    if args.monitor:
        monitor_mode(client_ids, cred_map, tb_tokens, args.window, args.interval)
    else:
        dates = []
        d = start
        while d <= end:
            dates.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        for i in range(0, len(dates), CHUNK_SIZE):
            batch = dates[i:i+CHUNK_SIZE]
            print(f"🔄 Procesando lote de días: {batch}")
            for date_str in batch:
                for client_id in client_ids:
                    secret = cred_map.get(client_id)
                    rec = tb_tokens.get(client_id, {})
                    token, usuario = rec.get('token'), rec.get('usuario')
                    if not (secret and token and usuario):
                        print(f"⚠️ Credenciales faltantes para {client_id} en {date_str}")
                        continue
                    process_single_date(client_id, secret, token, usuario, date_str, args.window)
            if i + CHUNK_SIZE < len(dates):
                print(f"⏳ Pausando {args.interval} segundos antes del siguiente lote...")
                time.sleep(args.interval)

if __name__ == "__main__":
    main()
