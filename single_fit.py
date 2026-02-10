from __future__ import annotations
import json
import os
import threading
import webbrowser
from datetime import datetime, timedelta
from typing import Dict, Optional
import tkinter as tk
from flask import Flask, redirect, request, session as flask_session
from requests_oauthlib import OAuth2Session
from tkcalendar import DateEntry
from tkinter import messagebox, ttk
from fitbit_service import (
    AGG_WINDOW_SECONDS,
    CREDENTIALS_FILE,
    CLIENT_IDS_FILE,
    DATA_DIR,
    OAUTH_SCOPES,
    REDIRECT_URI,
    THINGSBOARD_MQTT_HOST,
    THINGSBOARD_MQTT_PORT,
    TOKEN_URL,
    FitbitAPIClient,
    debug_print_token_scopes,
    export_json_to_excel_single,
    fetch_br_data,
    fetch_hrv_data,
    fetch_hrv_intraday,
    fetch_hrv_intraday_range,
    get_fitbit_data,
    refresh_access_token,
    save_credentials,
    save_daily_data,
    save_token,
)
from send_to_thingsboard import (
    generate_hrv_proxy_time_series_payloads,
    generate_static_payload,
    generate_time_series_payloads,
    mqtt_publish,
)
THINGSBOARD_TOKENS_FILE = "thingsboard_tokens.json"
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.makedirs(DATA_DIR, exist_ok=True)
def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return {}
class ThingsBoardPublisher:
    def __init__(self, tokens_path: str = THINGSBOARD_TOKENS_FILE) -> None:
        self.tokens_path = tokens_path
        self._tokens = _load_json(tokens_path)
    def reload(self) -> None:
        self._tokens = _load_json(self.tokens_path)
    def publish(self, client_id: str, data: Dict, monitor: bool = False) -> None:
        record = self._tokens.get(client_id) or {}
        token = record.get("token")
        usuario = record.get("usuario")
        if not (token and usuario):
            raise ValueError(f"No se encontró token ThingsBoard para {client_id}")
        payloads = []
        static = generate_static_payload(data, usuario, use_current_ts=monitor)
        if static:
            payloads.append(static)
        payloads.extend(generate_time_series_payloads(data, 60, usuario))
        payloads.extend(generate_hrv_proxy_time_series_payloads(data, usuario))
        if payloads:
            mqtt_publish(THINGSBOARD_MQTT_HOST, THINGSBOARD_MQTT_PORT, token, payloads)
def store_daily_artifacts(client_id: str, date_str: str, data: Dict) -> str:
    save_daily_data(client_id, data)
    export_json_to_excel_single(client_id, date_str)
    return os.path.join(DATA_DIR, client_id, f"{date_str}_{client_id}.json")
def collect_and_process(client_id: str, client_secret: str, date_str: str) -> Optional[Dict]:
    client = FitbitAPIClient(client_id, client_secret)
    data = get_fitbit_data(client_id, client_secret, date_str, client=client)
    if not data:
        return None
    store_daily_artifacts(client_id, date_str, data)
    return data
class FitbitApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fitbit Data Collector")
        self.status = tk.StringVar(value="Estado: Esperando credenciales")
        self.client_id = tk.StringVar()
        self.client_secret = tk.StringVar()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.publisher = ThingsBoardPublisher()
        self._build_ui()
        self._load_client_ids()
        self._start_flask()
    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        cred_box = ttk.LabelFrame(frame, text="Credenciales Fitbit", padding=10)
        cred_box.pack(fill=tk.X, pady=5)
        ttk.Label(cred_box, text="Client ID:").grid(row=0, column=0, sticky=tk.W)
        self.client_combo = ttk.Combobox(cred_box, textvariable=self.client_id, width=36)
        self.client_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.client_combo.bind("<<ComboboxSelected>>", lambda _: self._fill_secret())
        ttk.Button(cred_box, text="", width=3, command=self._load_client_ids).grid(row=0, column=2, padx=5)
        ttk.Label(cred_box, text="Client Secret:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(cred_box, textvariable=self.client_secret, width=40, show="*").grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Button(cred_box, text="Autenticar", command=self.authenticate).grid(row=2, column=1, sticky=tk.E, pady=10)
        date_box = ttk.LabelFrame(frame, text="Rango de fechas", padding=10)
        date_box.pack(fill=tk.X, pady=5)
        ttk.Label(date_box, text="Inicio:").grid(row=0, column=0, sticky=tk.W)
        self.start_date = DateEntry(date_box, date_pattern="yyyy-mm-dd")
        self.start_date.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(date_box, text="Fin:").grid(row=1, column=0, sticky=tk.W)
        self.end_date = DateEntry(date_box, date_pattern="yyyy-mm-dd")
        self.end_date.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(date_box, text="Descargar", command=self.download_range).grid(row=2, column=1, sticky=tk.E, pady=10)
        monitor_box = ttk.LabelFrame(frame, text="Monitoreo", padding=10)
        monitor_box.pack(fill=tk.X, pady=5)
        ttk.Button(monitor_box, text="Iniciar", command=self.start_monitoring).grid(row=0, column=0, padx=5)
        ttk.Button(monitor_box, text="Detener", command=self.stop_monitoring).grid(row=0, column=1, padx=5)
        self.monitor_status = ttk.Label(monitor_box, text="Estado: Inactivo")
        self.monitor_status.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        status_box = ttk.Frame(frame)
        status_box.pack(fill=tk.X, pady=5)
        ttk.Label(status_box, textvariable=self.status).pack(side=tk.LEFT)
        console_box = ttk.LabelFrame(frame, text="Registro", padding=10)
        console_box.pack(fill=tk.BOTH, expand=True)
        self.console = tk.Text(console_box, height=10, state=tk.DISABLED)
        self.console.pack(fill=tk.BOTH, expand=True)
    def _load_client_ids(self) -> None:
        values = []
        if os.path.exists(CLIENT_IDS_FILE):
            with open(CLIENT_IDS_FILE, "r", encoding="utf-8") as fh:
                values = [line.strip() for line in fh if line.strip()]
        self.client_combo["values"] = values
    def _fill_secret(self) -> None:
        cid = self.client_id.get()
        if not cid or not os.path.exists(CREDENTIALS_FILE):
            return
        try:
            with open(CREDENTIALS_FILE, "r", encoding="utf-8") as fh:
                creds = json.load(fh)
        except json.JSONDecodeError:
            return
        for record in creds:
            if record.get("client_id") == cid:
                self.client_secret.set(record.get("client_secret", ""))
                break
    def _start_flask(self) -> None:
        self.flask_app = Flask(__name__)
        self.flask_app.secret_key = "fitbit-local"
        @self.flask_app.route("/")
        def index():
            return redirect("/auth")
        @self.flask_app.route("/auth")
        def auth():
            cid = self.client_id.get()
            secret = self.client_secret.get()
            if not cid or not secret:
                return "Faltan credenciales"
            save_credentials(cid, secret)
            self._append_client_id(cid)
            fitbit_session = OAuth2Session(cid, redirect_uri=REDIRECT_URI, scope=OAUTH_SCOPES)
            authorization_url, _ = fitbit_session.authorization_url("https://www.fitbit.com/oauth2/authorize")
            webbrowser.open(authorization_url)
            return "Redirigiendo a Fitbit..."
        @self.flask_app.route("/callback")
        def callback():
            cid = self.client_id.get()
            secret = self.client_secret.get()
            fitbit_session = OAuth2Session(cid, redirect_uri=REDIRECT_URI, scope=OAUTH_SCOPES)
            token = fitbit_session.fetch_token(TOKEN_URL, client_secret=secret, authorization_response=request.url)
            save_token(cid, token)
            flask_session["client_id"] = cid
            self.log(f"Token guardado para {cid}")
            return "Autenticación completada. Puedes cerrar esta pestaña."
        def run():
            self.flask_app.run(port=5000)
        threading.Thread(target=run, daemon=True).start()
    def authenticate(self) -> None:
        cid = self.client_id.get()
        secret = self.client_secret.get()
        if not cid or not secret:
            messagebox.showerror("Error", "Debe proporcionar Client ID y Secret")
            return
        save_credentials(cid, secret)
        self.status.set(f"Estado: autenticando {cid}")
        webbrowser.open("http://localhost:5000/auth")
    def _append_client_id(self, cid: str) -> None:
        if not cid:
            return
        existing = set(self.client_combo["values"] or [])
        if cid not in existing:
            with open(CLIENT_IDS_FILE, "a", encoding="utf-8") as fh:
                fh.write(cid + "\n")
            self._load_client_ids()
    def download_range(self) -> None:
        cid = self.client_id.get()
        secret = self.client_secret.get()
        if not cid or not secret:
            messagebox.showerror("Error", "Debe proporcionar credenciales")
            return
        start_date = self.start_date.get_date()
        end_date = self.end_date.get_date()
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            self.log(f"Descargando datos para {cid} - {date_str}")
            data = collect_and_process(cid, secret, date_str)
            if data:
                self.log("Exportando a ThingsBoard...")
                try:
                    self.publisher.publish(cid, data)
                    self.log("Envío completado")
                except Exception as exc:
                    self.log(f"Error enviando a ThingsBoard: {exc}")
            else:
                self.log("No se obtuvieron datos")
            current += timedelta(days=1)
        self.status.set("Estado: Descarga finalizada")
    def start_monitoring(self) -> None:
        if self.monitoring:
            return
        cid = self.client_id.get()
        secret = self.client_secret.get()
        if not cid or not secret:
            messagebox.showerror("Error", "Debe proporcionar credenciales")
            return
        self.monitoring = True
        self.stop_event.clear()
        self.monitor_status.config(text="Estado: Monitoreo activo")
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(cid, secret), daemon=True)
        self.monitor_thread.start()
    def stop_monitoring(self) -> None:
        self.monitoring = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join()
        self.monitor_status.config(text="Estado: Inactivo")
    def _monitor_loop(self, client_id: str, client_secret: str) -> None:
        while not self.stop_event.is_set():
            today = datetime.utcnow().strftime("%Y-%m-%d")
            self.log(f"Monitoreo {client_id} - {today}")
            data = collect_and_process(client_id, client_secret, today)
            if data:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                client_dir = os.path.join(DATA_DIR, client_id)
                json_path = os.path.join(client_dir, f"{today}_{client_id}_monitor_{timestamp}.json")
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=4)
                export_json_to_excel_single(client_id, today)
                try:
                    self.publisher.publish(client_id, data, monitor=True)
                except Exception as exc:
                    self.log(f"Error enviando monitoreo: {exc}")
            self.stop_event.wait(3600)
    def log(self, message: str) -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)
__all__ = [
    "get_fitbit_data",
    "fetch_hrv_data",
    "fetch_br_data",
    "fetch_hrv_intraday",
    "fetch_hrv_intraday_range",
    "export_json_to_excel_single",
    "debug_print_token_scopes",
    "refresh_access_token",
]
def main() -> None:
    root = tk.Tk()
    FitbitApp(root)
    root.mainloop()
if __name__ == "__main__":
     main()