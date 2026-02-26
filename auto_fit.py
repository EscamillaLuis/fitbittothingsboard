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
from analisis.motor_alertas_vivo import (
    actualizar_baselines_historicos,
    evaluar_hora_actual,
)
from fitbit_service import (
    CREDENTIALS_FILE,
    CLIENT_IDS_FILE,
    DATA_DIR,
    FitbitAPIClient,
    ensure_specific_scopes,
    export_json_to_excel_single,
    fetch_br_data,
    fetch_hrv_data,
    fetch_hrv_intraday,
    fetch_hrv_intraday_range,
    reauthenticate_scopes,
    get_fitbit_data,
)
from send_to_thingsboard import (
    generate_static_payload,
    generate_time_series_payloads,
    mqtt_publish,
)
import sys
import io
import threading
import atexit
_TEE_LOCK = threading.Lock()
_TEE_FILE_HANDLE = None
class TeeStream(io.TextIOBase):
    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle
        self._encoding = getattr(stream, "encoding", "utf-8")
    @property
    def encoding(self):
        return self._encoding
    def write(self, s):
        if not isinstance(s, str):
            try:
                s = s.decode(self.encoding, errors="replace")
            except Exception:
                s = str(s)
        with _TEE_LOCK:
            self._stream.write(s)
            self._stream.flush()
            self._file.write(s)
            self._file.flush()
        return len(s)
    def flush(self):
        with _TEE_LOCK:
            self._stream.flush()
            self._file.flush()
    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False
def enable_terminal_log_mirroring(log_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    global _TEE_FILE_HANDLE
    if _TEE_FILE_HANDLE is not None:
        return
    _TEE_FILE_HANDLE = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, _TEE_FILE_HANDLE)
    sys.stderr = TeeStream(sys.stderr, _TEE_FILE_HANDLE)
    atexit.register(lambda: (_TEE_FILE_HANDLE and _TEE_FILE_HANDLE.flush()))
    atexit.register(lambda: (_TEE_FILE_HANDLE and _TEE_FILE_HANDLE.close()))
THINGSBOARD_TOKENS_FILE = "thingsboard_tokens.json"
DEFAULT_WINDOW = 60
DEFAULT_INTERVAL = 3600
DEFAULT_WORKERS = int(os.getenv("AUTO_FIT_WORKERS", "4"))
DEFAULT_BATCH_SIZE = 5
BASELINE_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60
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
            return f"️ Credenciales faltantes para {client_id}", False
        if not (token and usuario):
            return f"️ Token ThingsBoard faltante para {client_id}", False
        client = FitbitAPIClient(client_id, secret)
        data = get_fitbit_data(client_id, secret, date_str, client=client)
        if not data:
            return f" Sin datos para {client_id} en {date_str}", False
        try:
            data = evaluar_hora_actual(client_id, data)
        except Exception as exc:
            print(f" Error en motor de alertas ({client_id} {date_str}): {exc}")
        self._persist_files(client_id, date_str, data)
        payloads = self._build_payloads(data, usuario)
        if payloads:
            try:
                mqtt_publish("thingsboard.cloud", 1883, token, payloads)
            except Exception as exc:
                return f" Error enviando a ThingsBoard ({client_id} {date_str}): {exc}", False
        return f" Procesado {client_id} - {date_str} ({len(payloads)} payloads)", True
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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Tamano de lote de fechas")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Número máximo de hilos en paralelo")
    parser.add_argument("--check-scopes", type=str, help="Imprime los scopes del token para el usuario dado")
    parser.add_argument("--fetch-hrv", action="store_true", help="Descarga HRV para un usuario")
    parser.add_argument("--fetch-br", action="store_true", help="Descarga tasa respiratoria para un usuario")
    parser.add_argument("--date", type=str, help="Fecha para --fetch-hrv/--fetch-br")
    parser.add_argument("--dates", type=str, help="Rango YYYY-MM-DD:YYYY-MM-DD para backfill HRV intradía")
    parser.add_argument("--intraday", action="store_true", help="Usar endpoint intradía en pruebas")
    parser.add_argument(
        "--reauth-missing-scopes",
        action="store_true",
        help="Reautoriza en navegador a todos los usuarios que no tengan oxygen_saturation/respiratory_rate/heartrate/sleep",
    )
    parser.add_argument("--user", type=str, help="Usuario para comandos de prueba")
    return parser.parse_args()
def daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
def _chunks(seq: List[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        raise ValueError("El tamano de lote debe ser mayor a cero")
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]
def _process_user_batch(
    runner: AutoFitRunner,
    client_id: str,
    dates: List[str],
) -> str:
    successes = 0
    total = len(dates)
    for date_str in dates:
        try:
            message, success = runner.process(client_id, date_str)
        except Exception as exc:
            success = False
            message = f" Error procesando {client_id} - {date_str}: {exc}"
        print(message)
        if success:
            successes += 1
    return f" {client_id}: {successes}/{total} fechas ok"


def _refresh_baselines_cache_if_needed() -> None:
    cache_path = Path(DATA_DIR) / "baselines_cache.json"
    should_refresh = True
    if cache_path.exists():
        age_seconds = time.time() - cache_path.stat().st_mtime
        should_refresh = age_seconds > BASELINE_CACHE_MAX_AGE_SECONDS
    if not should_refresh:
        return
    try:
        ok = actualizar_baselines_historicos(DATA_DIR)
        if ok:
            print(" Baselines intradia actualizados.")
        else:
            print(" No se pudieron actualizar baselines intradia.")
    except Exception as exc:
        print(f" Error actualizando baselines intradia: {exc}")


def monitor_mode(
    client_ids: List[str],
    credentials: Dict[str, str],
    tb_tokens: Dict[str, Dict[str, str]],
    window: int,
    interval: int,
) -> None:
    _refresh_baselines_cache_if_needed()
    runner = AutoFitRunner(credentials, tb_tokens, window=window, monitor=True)
    print(" Monitoreo horario activo. Presiona Ctrl+C para detener.")
    try:
        while True:
            today = datetime.now().strftime("%Y-%m-%d")
            with ThreadPoolExecutor(max_workers=len(client_ids) or 1) as executor:
                futures = [executor.submit(runner.process, client_id, today) for client_id in client_ids]
                for future in as_completed(futures):
                    message, _ = future.result()
                    print(message)
            pause_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f" Esperando {interval} segundos para siguiente monitoreo... ({pause_time})")
            time.sleep(interval)
    except KeyboardInterrupt:
        print(" Monitoreo detenido por el usuario.")
def main() -> None:
    args = parse_args()
    if args.check_scopes:
        from fitbit_service import debug_print_token_scopes
        debug_print_token_scopes(args.check_scopes)
        return
    if args.reauth_missing_scopes:
        creds = load_json(CREDENTIALS_FILE)
        credentials = {c["client_id"]: c["client_secret"] for c in creds}
        try:
            with open(CLIENT_IDS_FILE, "r", encoding="utf-8") as fh:
                file_ids = {line.strip() for line in fh if line.strip()}
        except FileNotFoundError:
            file_ids = set()
        all_ids = sorted(set(credentials.keys()) | file_ids)
        if not all_ids:
            print("No se encontraron usuarios para reautorizar.")
            return
        required_scopes = ["oxygen_saturation", "respiratory_rate", "heartrate", "sleep"]
        for client_id in all_ids:
            secret = credentials.get(client_id)
            if not secret:
                print(f"Saltando {client_id}: falta client_secret en credentials.json")
                continue
            if ensure_specific_scopes(client_id, secret, required_scopes):
                continue
            reauthenticate_scopes(client_id, secret)
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
    if args.batch_size <= 0:
        raise ValueError("El tamano de lote debe ser mayor a cero")
    runner = AutoFitRunner(credentials, tb_tokens, window=args.window)
    all_dates = [d.strftime("%Y-%m-%d") for d in daterange(start_date, end_date)]
    total_batches = (len(all_dates) + args.batch_size - 1) // args.batch_size
    for batch_idx, batch_dates in enumerate(_chunks(all_dates, args.batch_size), start=1):
        print(f" Lote {batch_idx}: {batch_dates[0]}..{batch_dates[-1]}")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_user_batch, runner, client_id, batch_dates): client_id
                for client_id in client_ids
            }
            for future in as_completed(futures):
                summary = future.result()
                if summary:
                    print(summary)
        if batch_idx < total_batches:
            pause_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f" Pausando {args.interval} segundos antes del siguiente lote... ({pause_time})")
            time.sleep(args.interval)
    pause_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(" Backfill completado. Entrando a modo monitor...")
    print(f" Esperando {args.interval} segundos antes de iniciar monitoreo... ({pause_time})")
    time.sleep(args.interval)
    monitor_mode(client_ids, credentials, tb_tokens, args.window, args.interval)
if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    log_path = os.getenv("AUTO_FIT_LOG_FILE", os.path.join("logs", "auto_fit.log"))
    enable_terminal_log_mirroring(log_path)
    main()

