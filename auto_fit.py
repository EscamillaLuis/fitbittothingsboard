from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fitbit_service import (
    CREDENTIALS_FILE,
    CLIENT_IDS_FILE,
    DATA_DIR,
    FitbitAPIClient,
    export_json_to_excel_single,
    fetch_br_data,
    fetch_hrv_data,
    fetch_hrv_intraday,
    fetch_hrv_intraday_range,
    get_fitbit_data,
)
from send_to_thingsboard import (
    generate_hrv_proxy_time_series_payloads,
    generate_static_payload,
    generate_time_series_payloads,
    mqtt_publish,
)

THINGSBOARD_TOKENS_FILE = "thingsboard_tokens.json"
DEFAULT_WINDOW = 60
DEFAULT_INTERVAL = 3000
DEFAULT_WORKERS = int(os.getenv("AUTO_FIT_WORKERS", "4"))


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class AutoFitRunner:
    def __init__(
        self,
        credentials: Dict[str, str],
        tb_tokens: Dict[str, Dict[str, str]],
        *,
        window: int,
        monitor: bool = False,
    ) -> None:
        self.credentials = credentials
        self.tb_tokens = tb_tokens
        self.window = window
        self.monitor = monitor

    def process(self, client_id: str, date_str: str) -> Tuple[str, bool]:
        secret = self.credentials.get(client_id)
        tb_record = self.tb_tokens.get(client_id, {})
        token = tb_record.get("token")
        usuario = tb_record.get("usuario")
        if not secret:
            return f"⚠️ Credenciales faltantes para {client_id}", False
        if not (token and usuario):
            return f"⚠️ Token ThingsBoard faltante para {client_id}", False

        client = FitbitAPIClient(client_id, secret)
        data = get_fitbit_data(client_id, secret, date_str, client=client)
        if not data:
            return f"❌ Sin datos para {client_id} en {date_str}", False

        self._persist_files(client_id, date_str, data)
        payloads = self._build_payloads(data, usuario)
        if payloads:
            try:
                mqtt_publish("thingsboard.cloud", 1883, token, payloads)
            except Exception as exc:
                return f"❌ Error enviando a ThingsBoard ({client_id} {date_str}): {exc}", False
        return f"✅ Procesado {client_id} - {date_str} ({len(payloads)} payloads)", True

    def _persist_files(self, client_id: str, date_str: str, data: Dict) -> None:
        year, month, day = date_str.split("-")
        client_root = Path(DATA_DIR) / client_id
        nested_dir = client_root / year / month / day
        nested_dir.mkdir(parents=True, exist_ok=True)
        json_nested = nested_dir / f"{date_str}_{client_id}.json"
        with open(json_nested, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)

        client_root.mkdir(parents=True, exist_ok=True)
        flat_json = client_root / f"{date_str}_{client_id}.json"
        shutil.copy(json_nested, flat_json)
        export_json_to_excel_single(client_id, date_str)
        excel_flat = client_root / f"{date_str}_{client_id}_procesado.xlsx"
        if excel_flat.exists():
            shutil.move(excel_flat, nested_dir / excel_flat.name)
        if flat_json.exists():
            flat_json.unlink()

    def _build_payloads(self, data: Dict, usuario: str) -> List[Dict]:
        payloads: List[Dict] = []
        static = generate_static_payload(data, usuario, use_current_ts=self.monitor)
        if static:
            payloads.append(static)
        payloads.extend(generate_time_series_payloads(data, self.window, usuario))
        payloads.extend(generate_hrv_proxy_time_series_payloads(data, usuario))
        return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatiza descarga y envío de datos Fitbit para múltiples pulseras",
    )
    parser.add_argument("--start-date", type=str, help="Fecha inicio (YYYY-MM-DD). Por defecto hoy")
    parser.add_argument("--end-date", type=str, help="Fecha fin (YYYY-MM-DD). Por defecto start-date")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Ventana de agregación en segundos")
    parser.add_argument("--monitor", action="store_true", help="Activar monitoreo continuo")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Intervalo entre lotes")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Número máximo de hilos en paralelo")
    parser.add_argument("--check-scopes", type=str, help="Imprime los scopes del token para el usuario dado")
    parser.add_argument("--fetch-hrv", action="store_true", help="Descarga HRV para un usuario")
    parser.add_argument("--fetch-br", action="store_true", help="Descarga tasa respiratoria para un usuario")
    parser.add_argument("--date", type=str, help="Fecha para --fetch-hrv/--fetch-br")
    parser.add_argument("--dates", type=str, help="Rango YYYY-MM-DD:YYYY-MM-DD para backfill HRV intradía")
    parser.add_argument("--intraday", action="store_true", help="Usar endpoint intradía en pruebas")
    parser.add_argument("--user", type=str, help="Usuario para comandos de prueba")
    return parser.parse_args()


def daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def monitor_mode(
    client_ids: List[str],
    credentials: Dict[str, str],
    tb_tokens: Dict[str, Dict[str, str]],
    window: int,
    interval: int,
) -> None:
    runner = AutoFitRunner(credentials, tb_tokens, window=window, monitor=True)
    print("🔃 Monitoreo horario activo. Presiona Ctrl+C para detener.")
    try:
        while True:
            today = datetime.now().strftime("%Y-%m-%d")
            with ThreadPoolExecutor(max_workers=len(client_ids) or 1) as executor:
                futures = [executor.submit(runner.process, client_id, today) for client_id in client_ids]
                for future in as_completed(futures):
                    message, _ = future.result()
                    print(message)
            print(f"⏳ Esperando {interval} segundos para siguiente monitoreo...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("⏹ Monitoreo detenido por el usuario.")


def main() -> None:
    args = parse_args()

    if args.check_scopes:
        from fitbit_service import debug_print_token_scopes

        debug_print_token_scopes(args.check_scopes)
        return

    if args.fetch_hrv or args.fetch_br:
        if not args.user:
            raise ValueError("Debe indicar --user")
        creds = load_json(CREDENTIALS_FILE)
        cred_map = {c["client_id"]: c["client_secret"] for c in creds}
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
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else today
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else start_date
    if end_date < start_date:
        raise ValueError("La fecha fin no puede ser anterior a la fecha inicio")

    creds = load_json(CREDENTIALS_FILE)
    credentials = {c["client_id"]: c["client_secret"] for c in creds}
    tb_tokens = load_json(THINGSBOARD_TOKENS_FILE)
    with open(CLIENT_IDS_FILE, "r", encoding="utf-8") as fh:
        client_ids = [line.strip() for line in fh if line.strip()]

    if args.monitor:
        monitor_mode(client_ids, credentials, tb_tokens, args.window, args.interval)
        return

    runner = AutoFitRunner(credentials, tb_tokens, window=args.window)
    for chunk_start in daterange(start_date, end_date):
        date_str = chunk_start.strftime("%Y-%m-%d")
        print(f"📅 Procesando {date_str}")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(runner.process, client_id, date_str): client_id for client_id in client_ids}
            for future in as_completed(futures):
                message, success = future.result()
                print(message)
        if date_str != end_date.strftime("%Y-%m-%d"):
            print(f"⏳ Pausando {args.interval} segundos antes de la siguiente fecha...")
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
