import os
import json
import time
import webbrowser
import requests
import socket
import ssl
import logging
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.util import connection  as urllib3_connection
from urllib3.util import ssl_ as urllib3_ssl
import certifi
from base64 import b64encode
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from flask import Flask, request, redirect, session
from requests_oauthlib import OAuth2Session
import threading

TOKEN_FILE = "fitbit_tokens.json"
CREDENTIALS_FILE = "credentials.json"
CLIENT_IDS_FILE = "client_ids.txt" 
DATA_DIR = "fitbit_data"
API_DELAY = 1  
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
REDIRECT_URI = "http://localhost:5000/callback"

from send_to_thingsboard import (
    generate_static_payload,
    generate_time_series_payloads,
    mqtt_publish
)
THINGSBOARD_MQTT_HOST = 'thingsboard.cloud'
THINGSBOARD_MQTT_PORT = 1883
AGG_WINDOW_SECONDS = 60

DEFAULT_TIMEOUT = (5, 30)

REQUIRED_SCOPES = {
    "activity",
    "heartrate",
    "sleep",
    "profile",
    "settings",
    "weight",
    "respiratory_rate",
    "electrocardiogram",
    "oxygen_saturation",
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('urllib3').setLevel(logging.WARNING)

_conn_timings = threading.local()


def _force_ipv4() -> int:
    return socket.AF_INET


urllib3_connection.allowed_gai_family = _force_ipv4

_original_create_connection = urllib3_connection.create_connection
_original_ssl_wrap_socket = urllib3_ssl.ssl_wrap_socket


def _create_connection_timed(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
                             source_address=None, socket_options=None, **kwargs):
    host, port = address
    dns_start = time.perf_counter()
    addrinfos = socket.getaddrinfo(host, port, family=_force_ipv4(), type=socket.SOCK_STREAM)
    dns_end = time.perf_counter()
    _conn_timings.dns = dns_end - dns_start
    _conn_timings.addrs = [ai[4][0] for ai in addrinfos]
    for af, socktype, proto, canonname, sa in addrinfos:
        sock = socket.socket(af, socktype, proto)
        _conn_timings.family = af
        if socket_options:
            for opt in socket_options:
                sock.setsockopt(*opt)
        if source_address:
            sock.bind(source_address)
        if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            sock.settimeout(timeout)
        connect_start = time.perf_counter()
        sock.connect(sa)
        connect_end = time.perf_counter()
        _conn_timings.connect = connect_end - connect_start
        return sock
    raise OSError("Unable to connect")


def _ssl_wrap_socket_timed(sock, *args, **kwargs):
    tls_start = time.perf_counter()
    ssock = _original_ssl_wrap_socket(sock, *args, **kwargs)
    tls_end = time.perf_counter()
    _conn_timings.tls = tls_end - tls_start
    try:
        _conn_timings.cert = ssock.getpeercert()
    except Exception:
        _conn_timings.cert = {}
    return ssock


urllib3_connection.create_connection = _create_connection_timed
urllib3_ssl.ssl_wrap_socket = _ssl_wrap_socket_timed


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = certifi.where()
    return session


def timed_request(session: requests.Session, method: str, url: str, headers: Dict = None,
                  timeout: tuple = DEFAULT_TIMEOUT, **kwargs) -> requests.Response:
    _conn_timings.dns = _conn_timings.connect = _conn_timings.tls = None
    start = time.perf_counter()
    response = session.request(method, url, headers=headers, timeout=timeout, **kwargs)
    ttfb = response.elapsed.total_seconds()
    total = time.perf_counter() - start
    host = urlparse(url).hostname
    family = "IPv6" if getattr(_conn_timings, 'family', socket.AF_INET) == socket.AF_INET6 else "IPv4"
    ips = getattr(_conn_timings, 'addrs', [])
    logger.debug(
        "[%s] DNS %.3fms connect %.3fms TLS %.3fms TTFB %.3fms total %.3fms IPs %s family %s",
        host,
        (_conn_timings.dns or 0) * 1000,
        (_conn_timings.connect or 0) * 1000,
        (_conn_timings.tls or 0) * 1000,
        ttfb * 1000,
        total * 1000,
        ips,
        family,
    )
    return response 

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

os.makedirs(DATA_DIR, exist_ok=True)

class FitbitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fitbit Data Collector")
        self.root.geometry("650x600")  # Aumentado para acomodar nueva sección
        
        # Variables
        self.client_id = tk.StringVar()
        self.client_secret = tk.StringVar()
        self.status = tk.StringVar(value="Estado: Esperando credenciales")
        self.running_flask = False
        self.existing_client_ids = []  # Lista para almacenar client_ids existentes
        
        # Variables para monitoreo horario
        self.monitoring_active = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Crear interfaz
        self.create_widgets()
        
        # Cargar client_ids existentes
        self.load_client_ids()
        
        # Iniciar Flask en segundo plano
        self.start_flask_server()
    
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sección de credenciales
        cred_frame = ttk.LabelFrame(main_frame, text="Credenciales Fitbit", padding="10")
        cred_frame.pack(fill=tk.X, pady=5)
        
        # Combobox para seleccionar client_id existente
        ttk.Label(cred_frame, text="Client ID existente:").grid(row=0, column=0, sticky=tk.W)
        self.client_id_combobox = ttk.Combobox(cred_frame, textvariable=self.client_id, width=37)
        self.client_id_combobox.grid(row=0, column=1, padx=5, sticky=tk.W)
        self.client_id_combobox.bind("<<ComboboxSelected>>", self.on_client_id_selected)
        
        # Botón para actualizar lista de client_ids
        ttk.Button(cred_frame, text="↻", width=2, command=self.load_client_ids).grid(row=0, column=2, padx=5)
        
        # Campos para nuevo client_id y client_secret
        ttk.Label(cred_frame, text="Nuevo Client ID:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(cred_frame, textvariable=self.client_id, width=40).grid(row=1, column=1, columnspan=2, padx=5)
        
        ttk.Label(cred_frame, text="Client Secret:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(cred_frame, textvariable=self.client_secret, width=40, show="").grid(row=2, column=1, columnspan=2, padx=5)
        
        ttk.Button(cred_frame, text="Autenticar", command=self.authenticate).grid(row=3, column=1, pady=10, sticky=tk.E)
        
        # Sección de fechas
        date_frame = ttk.LabelFrame(main_frame, text="Rango de fechas", padding="10")
        date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(date_frame, text="Fecha inicio:").grid(row=0, column=0, sticky=tk.W)
        self.start_date = DateEntry(date_frame, date_pattern='yyyy-mm-dd')
        self.start_date.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(date_frame, text="Fecha fin:").grid(row=1, column=0, sticky=tk.W)
        self.end_date = DateEntry(date_frame, date_pattern='yyyy-mm-dd')
        self.end_date.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        ttk.Button(date_frame, text="Descargar datos", command=self.download_data).grid(row=2, column=1, pady=10, sticky=tk.E)
        
        # Nueva sección de monitoreo horario
        monitor_frame = ttk.LabelFrame(main_frame, text="Monitoreo", padding="10")
        monitor_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(monitor_frame, text="Iniciar Monitoreo", command=self.start_hourly_monitoring).grid(row=0, column=0, padx=5, sticky=tk.W)
        ttk.Button(monitor_frame, text="Detener Monitoreo", command=self.stop_hourly_monitoring).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        self.monitor_status = ttk.Label(monitor_frame, text="Estado: Inactivo")
        self.monitor_status.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # Sección de estado
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(status_frame, textvariable=self.status).pack(side=tk.LEFT)
        
        # Consola de salida
        console_frame = ttk.LabelFrame(main_frame, text="Registro", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True)
        
        self.console = tk.Text(console_frame, height=10, state=tk.DISABLED)
        self.console.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.console)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.console.yview)
    
    def send_to_thingsboard(self, data: dict):
        client_key = data.get("ID_Cliente") or self.client_id.get()
        try:
            with open("thingsboard_tokens.json", "r") as f:
                token_map = json.load(f)
            record = token_map.get(client_key, {})
            token = record.get("token")
            usuario = record.get("usuario")
        except Exception as e:
            self.log_message(f"Error cargando tokens: {e}")
            return

        if not token or usuario is None:
            self.log_message(f"No se encontró token o usuario para el client_id {client_key}")
            return
        payloads = []
        static = generate_static_payload(data, usuario)
        if static:
            payloads.append(static)
        payloads.extend(generate_time_series_payloads(data, AGG_WINDOW_SECONDS, usuario))

        if not payloads:
            self.log_message("No hay datos para enviar a ThingsBoard")
            return
        try:
            mqtt_publish(
                THINGSBOARD_MQTT_HOST,
                THINGSBOARD_MQTT_PORT,
                token,
                payloads
            )
            self.log_message(f"Enviados {len(payloads)} mensajes a ThingsBoard para usuario '{usuario}'")
        except Exception as e:
            self.log_message(f"Error enviando a ThingsBoard: {e}")

    def start_hourly_monitoring(self):
        if not self.client_id.get():
            messagebox.showerror("Error", "Por favor ingrese Client ID")
            return
            
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.stop_event.clear()
        self.monitor_status.config(text="Estado: Monitoreo activo ")
        self.log_message("\nIniciando monitoreo horario...")
        
        # Crear hilo para el monitoreo
        self.monitoring_thread = threading.Thread(
            target=self.hourly_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()

    def stop_hourly_monitoring(self):
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitoring_thread is not None:
            self.monitoring_thread.join()
            self.monitoring_thread = None
            
        self.monitor_status.config(text="Estado: Inactivo")
        self.log_message("\nMonitoreo detenido")

    def hourly_monitoring_loop(self):
        while not self.stop_event.is_set() and self.monitoring_active:
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M:%S")
            
            self.log_message(f"\nMonitoreo iniciado a las {current_time}")
            
            try:
                data = get_fitbit_data(self.client_id.get(), self.client_secret.get(), current_date)
                self.send_to_thingsboard(data)
                
                if data:
                    # Guardar datos con marca de tiempo
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    filename = f"{current_date}_{self.client_id.get()}_monitor_{timestamp}.json"
                    client_dir = os.path.join(DATA_DIR, self.client_id.get())
                    os.makedirs(client_dir, exist_ok=True)
                    file_path = os.path.join(client_dir, filename)
                    
                    with open(file_path, "w") as f:
                        json.dump(data, f, indent=4)
                    
                    self.log_message(f"Datos guardados: {filename}")
                    
                    # Exportar a Excel
                    excel_filename = f"{current_date}_{self.client_id.get()}_monitor_{timestamp}.xlsx"
                    excel_path = os.path.join(client_dir, excel_filename)
                    self.export_monitoring_data(file_path, excel_path)
                    self.log_message(f"Exportado a Excel: {excel_filename}")
                else:
                    self.log_message("No se pudieron obtener datos")
            except Exception as e:
                self.log_message(f"Error en monitoreo: {str(e)}")
            self.stop_event.wait(3600)

    def export_monitoring_data(self, json_path, excel_path):
        try:
            with open(json_path, "r", encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                self.log_message("Error: Los datos no están en formato de diccionario")
                return False
            
            resumen_actividades = data.get('Resumen_Actividades', {}) or {}
            resumen_sueño = data.get('Resumen_Sueño', [{}])[0] if isinstance(data.get('Resumen_Sueño', []), list) else {}
            spo2_data = data.get('SpO2', []) if isinstance(data.get('SpO2', []), list) else []
    
            df_data = {
            'Fecha': [data.get('Fecha', '')],
            'Hora': [datetime.now().strftime("%H:%M:%S")],
            'Pasos': [resumen_actividades.get('steps', 0)],
            'Calorias': [resumen_actividades.get('caloriesOut', 0)],
            'Ritmo Cardiaco Reposo': [resumen_actividades.get('restingHeartRate', 'N/A')],
            'Minutos Activos': [resumen_actividades.get('veryActiveMinutes', 0)],
            'Sueño (min)': [resumen_sueño.get('minutesAsleep', 0)],
            'SpO2 Promedio': [pd.DataFrame(spo2_data).get('value', pd.Series()).mean() if spo2_data else 'N/A']
            }
        
            df = pd.DataFrame(df_data)
        
        # Guardar en Excel
            df.to_excel(excel_path, index=False)
            self.log_message(f"Datos exportados correctamente a {excel_path}")
            return True
        
        except json.JSONDecodeError:
            self.log_message("Error: El archivo JSON no tiene formato válido")
            return False
        except Exception as e:
            self.log_message(f"Error inesperado al exportar datos: {str(e)}")
            return False
    
    def load_client_ids(self):
        try:
            if os.path.exists(CLIENT_IDS_FILE):
                with open(CLIENT_IDS_FILE, 'r') as f:
                    self.existing_client_ids = [line.strip() for line in f.readlines() if line.strip()]
            else:
                self.existing_client_ids = []
            
            # Actualizar el combobox
            self.client_id_combobox['values'] = self.existing_client_ids
            self.log_message("\nLista de client_ids actualizada")
            return True
        except Exception as e:
            self.log_message(f"\nError al cargar client_ids: {str(e)}")
            return False
    
    def save_client_id(self, client_id):
        """Guarda un nuevo client_id en el archivo si no existe"""
        try:
            # Verificar si ya existe
            if client_id in self.existing_client_ids:
                return True
                
            # Agregar a la lista y guardar
            self.existing_client_ids.append(client_id)
            with open(CLIENT_IDS_FILE, 'a') as f:
                f.write(client_id + "\n")
            
            # Actualizar el combobox
            self.client_id_combobox['values'] = self.existing_client_ids
            self.log_message(f"\n📝 Client_ID {client_id} guardado para futuras sesiones")
            return True
        except Exception as e:
            self.log_message(f"\nError al guardar client_id: {str(e)}")
            return False
    
    def on_client_id_selected(self, event):
        """Cuando se selecciona un client_id existente, cargar su client_secret"""
        selected_id = self.client_id.get()
        if not selected_id:
            return
        
        # Buscar en credentials.json el client_secret correspondiente
        try:
            if os.path.exists(CREDENTIALS_FILE):
                with open(CREDENTIALS_FILE, 'r') as f:
                    credentials = json.load(f)
                    for cred in credentials:
                        if cred['client_id'] == selected_id:
                            self.client_secret.set(cred['client_secret'])
                            self.log_message(f"\n🔑 Credenciales cargadas para {selected_id}")
                            self.status.set(f"Estado: Credenciales cargadas para {selected_id}")
                            break
        except Exception as e:
            self.log_message(f"\nError al cargar credenciales: {str(e)}")
    
    def log_message(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
    
    def start_flask_server(self):
        if not self.running_flask:
            self.flask_app = Flask(__name__)
            self.flask_app.secret_key = "clave_secreta_segura"
            
            @self.flask_app.route("/")
            def index():
                return redirect("http://localhost:5000/auth")
            
            @self.flask_app.route("/auth")
            def auth():
                save_credentials(self.client_id.get(), self.client_secret.get())
                self.save_client_id(self.client_id.get())  # Guardar el client_id
                fitbit_session = OAuth2Session(
                    self.client_id.get(), 
                    redirect_uri=REDIRECT_URI, 
                    scope=["activity", "heartrate", "sleep", "profile", "oxygen_saturation",
                          "respiratory_rate", "electrocardiogram", "settings", "weight", "nutrition", "location"]
                )
                authorization_url, _ = fitbit_session.authorization_url("https://www.fitbit.com/oauth2/authorize")
                webbrowser.open(authorization_url)
                return "Redirigiendo a Fitbit para autenticación..."
            
            @self.flask_app.route("/callback")
            def callback():
                try:
                    fitbit_session = OAuth2Session(
                        self.client_id.get(), 
                        redirect_uri=REDIRECT_URI,
                        scope=["activity", "heartrate", "sleep", "profile", "oxygen_saturation",
                              "respiratory_rate", "electrocardiogram", "settings", "weight", "nutrition", "location"]
                    )
                    token = fitbit_session.fetch_token(
                        TOKEN_URL,
                        client_secret=self.client_secret.get(),
                        authorization_response=request.url
                    )
                    
                    save_token(self.client_id.get(), token)
                    session["client_id"] = self.client_id.get()
                    
                    self.log_message("\nAutenticación exitosa!")
                    self.status.set("Estado: Autenticado - Puede descargar datos")
                    
                    return """
                    <h1>Autenticación exitosa</h1>
                    <p>Puedes cerrar esta ventana y volver a la aplicación.</p>
                    """
                except Exception as e:
                    self.log_message(f"\nError en autenticación: {str(e)}")
                    return f"Error en la autenticación: {str(e)}"
            
            flask_thread = threading.Thread(
                target=self.flask_app.run,
                kwargs={'host': '0.0.0.0', 'port': 5000},
                daemon=True
            )
            flask_thread.start()
            self.running_flask = True
    
    def authenticate(self):
        if not self.client_id.get() or not self.client_secret.get():
            messagebox.showerror("Error", "Por favor ingrese Client ID y Client Secret")
            return
        
        self.log_message("\n🔗 Abriendo navegador para autenticación...")
        self.status.set("Estado: Esperando autenticación en navegador...")
        webbrowser.open("http://localhost:5000/auth")
    
    def download_data(self):
        if not self.client_id.get():
            messagebox.showerror("Error", "Por favor ingrese Client ID")
            return
        
        start_date = self.start_date.get_date()
        end_date = self.end_date.get_date()
        
        if end_date < start_date:
            messagebox.showerror("Error", "La fecha fin no puede ser anterior a la fecha inicio")
            return
        
        self.status.set("Estado: Descargando datos...")
        self.log_message(f"\n⏳ Descargando datos del {start_date} al {end_date}...")
        
        # Ejecutar en segundo plano para no bloquear la GUI
        download_thread = threading.Thread(
            target=self._download_data_thread,
            args=(start_date, end_date),
            daemon=True
        )
        download_thread.start()
    
    def _download_data_thread(self, start_date, end_date):
        current_date = start_date
        delta = timedelta(days=1)
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            self.log_message(f"\n📅 Procesando fecha: {date_str}")
            
            data = get_fitbit_data(self.client_id.get(), self.client_secret.get(), date_str)
            if data:
                self.send_to_thingsboard(data)
                if save_daily_data(self.client_id.get(), data):
                    self.log_message(f"Datos guardados para {date_str}")
                    export_json_to_excel_single(self.client_id.get(), date_str)
                else:
                    self.log_message(f"Error al guardar datos para {date_str}")
            else:
                self.log_message(f"No se pudieron obtener datos para {date_str}")
            
            current_date += delta
        
        self.status.set("Estado: Descarga completada")
        self.log_message("\n🎉 ¡Proceso de descarga completado!")

# Funciones de manejo de datos
def save_credentials(client_id, client_secret):
    try:
        credentials = []
        if os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, 'r') as f:
                credentials = json.load(f)
        
        if not any(c['client_id'] == client_id for c in credentials):
            credentials.append({
                "client_id": client_id,
                "client_secret": client_secret
            })
            with open(CREDENTIALS_FILE, 'w') as f:
                json.dump(credentials, f, indent=4)
    
    except Exception as e:
        print(f"Error al guardar credenciales: {e}")

def load_tokens():
    try:
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_token(client_id, new_token):
    tokens = load_tokens()
    tokens[client_id] = new_token
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=4)

def check_scopes(client_id: str):
    tokens = load_tokens()
    token_data = tokens.get(client_id)
    if not token_data:
        return False
        
    current_scopes = set(token_data.get("scope", "").split())
    return REQUIRED_SCOPES.issubset(current_scopes)
    

def reauthenticate_scopes(client_id: str, client_secret: str) -> Optional[str]:
    print(f"Iniciando reautenticación para {client_id} con scopes completos...")
    fitbit_session = OAuth2Session(
        client_id,
        redirect_uri=REDIRECT_URI,
        scope=list(REQUIRED_SCOPES),
    )
    authorization_url, _ = fitbit_session.authorization_url("https://www.fitbit.com/oauth2/authorize")
    print("Abre la siguiente URL en tu navegador y autoriza el acceso:")
    print(authorization_url)
    webbrowser.open(authorization_url)
    redirect_response = input(
        "Después de autorizar, pega aquí la URL completa de redirección:\n"
    ).strip()
    try:
        token = fitbit_session.fetch_token(
            TOKEN_URL,
            client_secret=client_secret,
            authorization_response=redirect_response,
        )
    except Exception as e:
        print(f"Error durante la reautenticación: {e}")
        return None
    save_token(client_id, token)
    print(f"Nuevo token guardado para {client_id}")
    return token.get("access_token")
    
def refresh_access_token(client_id: str, client_secret: str) -> Optional[str]:
    tokens = load_tokens()
    token_data = tokens.get(client_id)
    
    if not token_data:
        print(f"No hay token para {client_id}. Necesita autenticación inicial.")
        return None

    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        print(f"No hay refresh_token para {client_id}.")
        return None

    scope_value = token_data.get("scope", "")
    if isinstance(scope_value, list):
        current_scopes = set(scope_value)
        scope_str = " ".join(scope_value)
    else:
        current_scopes = set(scope_value.split())
        scope_str = scope_value

    if not REQUIRED_SCOPES.issubset(current_scopes):
        missing_scopes = REQUIRED_SCOPES - current_scopes
        print(f"Error: Faltan scopes esenciales para {client_id}: {missing_scopes}")
        return reauthenticate_scopes(client_id, client_secret)

    auth_header = b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "scope": scope_str or "activity heartrate oxygen_saturation respiratory_rate electrocardiogram sleep settings weight"
    }

    try:
        time.sleep(API_DELAY)
        session = create_session()
        response = timed_request(session, "POST", TOKEN_URL, headers=headers, data=data)
        response.raise_for_status()
        new_token = response.json()

        if "access_token" in new_token:
            if "scope" not in new_token:
                new_token["scope"] = scope_str
                
            tokens[client_id] = new_token
            save_token(client_id, new_token)
            print(f"Token actualizado para {client_id}")
            return new_token["access_token"]
        else:
            print(f"Error al refrescar token: {new_token}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return None

def get_fitbit_data(client_id: str, client_secret: str, date: str) -> Optional[Dict]:
    token = refresh_access_token(client_id, client_secret)
    if not token:
        return None

    headers = {"Authorization": f"Bearer {token}"}
    session = create_session()
    result = {"Fecha": date, "ID_Cliente": client_id}

    def api_request(url: str, data_key: str = None) -> Optional[Dict]:
        time.sleep(API_DELAY)
        try:
            response = timed_request(session, "GET", url, headers=headers)
            if response.status_code == 200:
                return response.json().get(data_key) if data_key else response.json()
            elif response.status_code == 404:
                print(f"Datos no encontrados en {url}")
                return None
            elif response.status_code == 403:
                print(f"Permisos insuficientes para {url}")
                return None
            else:
                print(f"Error en {url}: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error en petición a {url}: {e}")
            return None

    # 1. Perfil de usuario
    if profile := api_request("https://api.fitbit.com/1/user/-/profile.json", "user"):
        result.update({
            "ID_Usuario": profile.get("encodedId"),
            "Nombre": profile.get("fullName"),
            "Edad": profile.get("age")
        })

    # 2. Endpoints principales existentes
    existing_endpoints = [
        ("Frecuencia_Respiratoria", f"https://api.fitbit.com/1/user/-/br/date/{date}/all.json", "br"),
        ("Ritmo_Cardiaco", f"https://api.fitbit.com/1/user/-/activities/heart/date/{date}/1d/1sec.json", "activities-heart-intraday"),
        ("Resumen_Actividades", f"https://api.fitbit.com/1/user/-/activities/date/{date}.json", "summary"),
        ("HRV", f"https://api.fitbit.com/1/user/-/hrv/date/{date}/all.json", "hrv"),
        ("SpO2", f"https://api.fitbit.com/1/user/-/spo2/date/{date}/all.json", "minutes")
    ]

    # 3. Endpoints de actividad
    activity_endpoints = [
        ("Pasos", f"https://api.fitbit.com/1/user/-/activities/steps/date/{date}/1d.json", "activities-steps"),
        ("Calorias", f"https://api.fitbit.com/1/user/-/activities/calories/date/{date}/1d.json", "activities-calories"),
        ("Distancia", f"https://api.fitbit.com/1/user/-/activities/distance/date/{date}/1d.json", "activities-distance"),
        ("Pisos", f"https://api.fitbit.com/1/user/-/activities/floors/date/{date}/1d.json", "activities-floors"),
        ("Zonas_Actividad", f"https://api.fitbit.com/1/user/-/activities/active-zone-minutes/date/{date}/1d.json", "activities-active-zone-minutes"),
        ("Metas_Actividad", f"https://api.fitbit.com/1/user/-/activities/goals/daily.json", "goals")
    ]

    # 4. Sueño
    sleep_endpoints = [
        ("Resumen_Sueño", f"https://api.fitbit.com/1.2/user/-/sleep/date/{date}.json", "sleep"),
        ("Metas_Sueño", f"https://api.fitbit.com/1/user/-/sleep/goal.json", "goal")
    ]

    # 5. Frecuencia cardíaca
    heart_endpoints = [
        ("Resumen_Ritmo_Cardiaco", f"https://api.fitbit.com/1/user/-/activities/heart/date/{date}/1d.json", "activities-heart"),
        ("Ritmo_Cardiaco_Reposo", f"https://api.fitbit.com/1/user/-/activities/heart/date/{date}/1d.json", "restingHeartRate")
    ]

    # 6. Peso y composición corporal (actualizado)
    body_endpoints = [
        ("Peso", f"https://api.fitbit.com/1/user/-/body/log/weight/date/{date}.json", "weight"),
        ("Grasa_Corporal", f"https://api.fitbit.com/1/user/-/body/log/fat/date/{date}.json", "fat"),
        ("IMC", f"https://api.fitbit.com/1/user/-/body/log/bmi/date/{date}.json", "bmi")
    ]

    # 7. Dispositivos (solo si tenemos permisos)
    device_endpoints = []
    if check_scopes(client_id):
        device_endpoints = [
            ("Dispositivos", f"https://api.fitbit.com/1/user/-/devices.json", None)
        ]

    # Combinar todos los endpoints
    all_endpoints = (
        existing_endpoints + 
        activity_endpoints +
        sleep_endpoints +
        heart_endpoints +
        body_endpoints +
        device_endpoints
    )

    # Procesar todos los endpoints
    for name, url, key in all_endpoints:
        result[name] = api_request(url, key) or "No disponible"

    # Mantener el procesamiento de actividades intraday existente
    activities = {}
    for resource in ["steps", "calories", "distance", "elevation"]:
        url = f"https://api.fitbit.com/1/user/-/activities/{resource}/date/{date}/1d/1min.json"
        if data := api_request(url, f"activities-{resource}-intraday"):
            activities[resource] = data.get("dataset", "No disponible")
        else:
            activities[resource] = "No disponible"
    
    result["Actividades"] = activities

    return result

def save_daily_data(client_id: str, data: Dict) -> bool:
    """Guarda datos en una subcarpeta por client_id."""
    client_dir = os.path.join(DATA_DIR, client_id)
    os.makedirs(client_dir, exist_ok=True)
    
    filename = f"{data['Fecha']}_{client_id}.json"
    file_path = os.path.join(client_dir, filename)

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error al guardar datos: {e}")
        return False

def export_json_to_excel_single(client_id: str, date: str, data_dir: str = "fitbit_data"):
    """
    Exporta un archivo JSON completo a un Excel con múltiples hojas organizadas.
    
    Args:
        client_id (str): ID del cliente
        date (str): Fecha en formato YYYY-MM-DD
        data_dir (str): Directorio base donde se almacenan los datos
    """
    client_path = os.path.join(data_dir, client_id)
    json_path = os.path.join(client_path, f"{date}_{client_id}.json")
    output_excel = os.path.join(client_path, f"{date}_{client_id}_procesado.xlsx")
    
    if not os.path.exists(json_path):
        print(f"Archivo no encontrado: {json_path}")
        return

    try:
        with open(json_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            print(f"Datos no válidos en {json_path}")
            return
        
        # Crear un escritor de Excel
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            
            # 1. Hoja de Información Básica del Cliente
            create_client_info_sheet(writer, data, date)
            
            # 2. Hoja de Ritmo Cardíaco
            create_heart_rate_sheet(writer, data, date)
            
            # 3. Hoja de Frecuencia Respiratoria
            create_respiratory_rate_sheet(writer, data, date)
            
            # 4. Hoja de Resumen de Actividades
            create_activity_summary_sheet(writer, data, date)
            
            # 5. Hoja de Sueño
            create_sleep_sheet(writer, data, date)
            
            # 6. Hoja de HRV (Variabilidad del Ritmo Cardíaco)
            create_hrv_sheet(writer, data, date)
            
            # 7. Hoja de SpO2 (Oxigenación)
            create_spo2_sheet(writer, data, date)
            
            # 8. Hoja de Zonas de Actividad
            create_activity_zones_sheet(writer, data, date)
            
            # 9. Hoja de Metas
            create_goals_sheet(writer, data, date)
            
            # 10. Hoja de Dispositivos
            create_devices_sheet(writer, data, date)
            
            # 11. Hoja de Datos de Actividad por minuto
            create_minute_activity_sheet(writer, data, date)
            
            # 12. Hoja de Resumen de Ritmo Cardíaco
            create_heart_rate_summary_sheet(writer, data, date)
            
            # 13. Hoja de Datos Biométricos
            create_biometrics_sheet(writer, data, date)
            
        print(f"Datos exportados exitosamente a {output_excel}")
        
    except json.JSONDecodeError:
        print(f"Error: El archivo {json_path} no es un JSON válido")
    except Exception as e:
        print(f"Error inesperado procesando {json_path}: {str(e)}")

def create_client_info_sheet(writer, data, date):
    """Crea hoja con información básica del cliente"""
    client_info = {
        "Campo": ["Fecha", "ID Cliente", "ID Usuario", "Nombre", "Edad"],
        "Valor": [
            data.get("Fecha", date),
            data.get("ID_Cliente", ""),
            data.get("ID_Usuario", ""),
            data.get("Nombre", ""),
            data.get("Edad", "")
        ]
    }
    df_info = pd.DataFrame(client_info)
    df_info.to_excel(writer, sheet_name="Informacion_Cliente", index=False)

def create_heart_rate_sheet(writer, data, date):
    """Crea hoja con datos de ritmo cardíaco"""
    if "Ritmo_Cardiaco" in data and isinstance(data["Ritmo_Cardiaco"], dict):
        heart_data = data["Ritmo_Cardiaco"].get("dataset", [])
        if heart_data and isinstance(heart_data, list):
            try:
                df_heart = pd.DataFrame(heart_data)
                df_heart = df_heart.rename(columns={"value": "Ritmo_Cardiaco", "time": "Tiempo"})
                df_heart["Tiempo"] = pd.to_datetime(df_heart["Tiempo"], format='%H:%M:%S').dt.time
                df_heart.insert(0, "Fecha", date)
                
                # Calcular estadísticas
                stats = {
                    "Estadística": ["Mínimo", "Máximo", "Promedio", "Desviación Estándar", "Frecuencia Cardiaca en Reposo"],
                    "Valor": [
                        df_heart["Ritmo_Cardiaco"].min(),
                        df_heart["Ritmo_Cardiaco"].max(),
                        df_heart["Ritmo_Cardiaco"].mean(),
                        df_heart["Ritmo_Cardiaco"].std(),
                        data.get("Resumen_Actividades", {}).get("restingHeartRate", "N/A")
                    ]
                }
                df_heart_stats = pd.DataFrame(stats)
                
                # Escribir ambas tablas en la misma hoja
                df_heart.to_excel(writer, sheet_name="Ritmo_Cardiaco", index=False, startrow=0)
                df_heart_stats.to_excel(writer, sheet_name="Ritmo_Cardiaco", index=False, startrow=len(df_heart)+3)
            except Exception as e:
                print(f"Error procesando ritmo cardíaco: {e}")

def create_respiratory_rate_sheet(writer, data, date):
    """Crea hoja con datos de frecuencia respiratoria"""
    if "Frecuencia_Respiratoria" in data and isinstance(data["Frecuencia_Respiratoria"], list):
        resp_data = data["Frecuencia_Respiratoria"]
        if resp_data and isinstance(resp_data[0], dict):
            try:
                resp_values = resp_data[0].get("value", {})
                if isinstance(resp_values, dict):
                    df_resp = pd.DataFrame({
                        "Tipo": ["Sueño profundo", "Sueño REM", "Sueño completo", "Sueño ligero"],
                        "Frecuencia_Resp/min": [
                            resp_values.get("deepSleepSummary", {}).get("breathingRate", "N/A"),
                            resp_values.get("remSleepSummary", {}).get("breathingRate", "N/A"),
                            resp_values.get("fullSleepSummary", {}).get("breathingRate", "N/A"),
                            resp_values.get("lightSleepSummary", {}).get("breathingRate", "N/A")
                        ]
                    })
                    df_resp.insert(0, "Fecha", date)
                    df_resp.to_excel(writer, sheet_name="Frecuencia_Respiratoria", index=False)
            except Exception as e:
                print(f"Error procesando frecuencia respiratoria: {e}")

def create_activity_summary_sheet(writer, data, date):
    """Crea hoja con resumen de actividades"""
    if "Resumen_Actividades" in data and isinstance(data["Resumen_Actividades"], dict):
        try:
            actividad_data = data["Resumen_Actividades"]
            
            # Resumen principal
            df_actividad = pd.DataFrame({
                "Metrica": ["Pasos", "Calorías quemadas", "Calorías actividad", "Calorías BMR", 
                          "Puntuación actividad", "Minutos sedentarios", "Minutos actividad ligera",
                          "Minutos actividad moderada", "Minutos actividad intensa", "Frecuencia cardiaca en reposo"],
                "Valor": [ 
                    actividad_data.get("steps", 0),
                    actividad_data.get("caloriesOut", 0),
                    actividad_data.get("activityCalories", 0),
                    actividad_data.get("caloriesBMR", 0),
                    actividad_data.get("activeScore", -1),
                    actividad_data.get("sedentaryMinutes", 0),
                    actividad_data.get("lightlyActiveMinutes", 0),
                    actividad_data.get("fairlyActiveMinutes", 0),
                    actividad_data.get("veryActiveMinutes", 0),
                    actividad_data.get("restingHeartRate", "N/A")
                ]
            })
            df_actividad.insert(0, "Fecha", date)
            
            # Distancias
            distances = actividad_data.get("distances", [])
            df_distances = pd.DataFrame(distances) if distances else pd.DataFrame()
            
            # Escribir hojas
            df_actividad.to_excel(writer, sheet_name="Resumen_Actividades", index=False)
            if not df_distances.empty:
                df_distances.to_excel(writer, sheet_name="Distancias_Actividad", index=False)
        except Exception as e:
            print(f"Error procesando resumen de actividades: {e}")

def create_sleep_sheet(writer, data, date):
    """Crea hoja con datos de sueño"""
    if "Resumen_Sueño" in data and isinstance(data["Resumen_Sueño"], list) and len(data["Resumen_Sueño"]) > 0:
        try:
            sueño_data = data["Resumen_Sueño"][0]
            
            # Resumen de sueño
            sueño_summary = sueño_data.get("levels", {}).get("summary", {})
            df_sueño = pd.DataFrame({
                "Tipo": ["Profundo", "Ligero", "REM", "Despierto"],
                "Minutos": [
                    sueño_summary.get("deep", {}).get("minutes", 0),
                    sueño_summary.get("light", {}).get("minutes", 0),
                    sueño_summary.get("rem", {}).get("minutes", 0),
                    sueño_summary.get("wake", {}).get("minutes", 0)
                ],
                "Porcentaje": [
                    sueño_summary.get("deep", {}).get("percent", 0),
                    sueño_summary.get("light", {}).get("percent", 0),
                    sueño_summary.get("rem", {}).get("percent", 0),
                    sueño_summary.get("wake", {}).get("percent", 0)
                ]
            })
            df_sueño.insert(0, "Fecha", date)
            
            # Información adicional del sueño
            sleep_info = {
                "Campo": ["Hora de inicio", "Hora de fin", "Duración (min)", "Minutos dormido", 
                         "Minutos despierto", "Minutos para dormirse", "Eficiencia", "Tiempo en cama (min)"],
                "Valor": [
                    sueño_data.get("startTime", ""),
                    sueño_data.get("endTime", ""),
                    round(sueño_data.get("duration", 0)/60000, 1),
                    sueño_data.get("minutesAsleep", 0),
                    sueño_data.get("minutesAwake", 0),
                    sueño_data.get("minutesToFallAsleep", 0),
                    sueño_data.get("efficiency", 0),
                    sueño_data.get("timeInBed", 0)
                ]
            }
            df_sleep_info = pd.DataFrame(sleep_info)
            
            # Datos detallados de niveles de sueño
            levels_data = sueño_data.get("levels", {}).get("data", [])
            df_levels = pd.DataFrame(levels_data) if levels_data else pd.DataFrame()
            
            # Escribir hojas
            df_sleep_info.to_excel(writer, sheet_name="Sueño", index=False, startrow=0)
            df_sueño.to_excel(writer, sheet_name="Sueño", index=False, startrow=len(df_sleep_info)+3)
            if not df_levels.empty:
                df_levels.to_excel(writer, sheet_name="Niveles_Sueño", index=False)
        except Exception as e:
            print(f"Error procesando datos de sueño: {e}")

def create_hrv_sheet(writer, data, date):
    """Crea hoja con datos de HRV (Variabilidad del Ritmo Cardíaco)"""
    if "HRV" in data and isinstance(data["HRV"], list):
        try:
            hrv_data = []
            for entry in data["HRV"]:
                minutes = entry.get("minutes", [])
                if minutes and isinstance(minutes, list):
                    for minute in minutes:
                        value = minute.get("value", {})
                        hrv_data.append({
                            "Fecha_Hora": minute.get("minute", ""),
                            "RMSSD": value.get("rmssd", ""),
                            "Cobertura": value.get("coverage", ""),
                            "HF": value.get("hf", ""),
                            "LF": value.get("lf", "")
                        })
            
            if hrv_data:
                df_hrv = pd.DataFrame(hrv_data)
                df_hrv.insert(0, "Fecha", date)
                
                # Calcular estadísticas
                stats = {
                    "Estadística": ["RMSSD Promedio", "RMSSD Máximo", "RMSSD Mínimo", "HF Promedio", "LF Promedio"],
                    "Valor": [
                        df_hrv["RMSSD"].mean(),
                        df_hrv["RMSSD"].max(),
                        df_hrv["RMSSD"].min(),
                        df_hrv["HF"].mean(),
                        df_hrv["LF"].mean()
                    ]
                }
                df_hrv_stats = pd.DataFrame(stats)
                
                # Escribir ambas tablas en la misma hoja
                df_hrv.to_excel(writer, sheet_name="HRV", index=False, startrow=0)
                df_hrv_stats.to_excel(writer, sheet_name="HRV", index=False, startrow=len(df_hrv)+3)
        except Exception as e:
            print(f"Error procesando datos HRV: {e}")

def create_spo2_sheet(writer, data, date):
    """Crea hoja con datos de SpO2 (Oxigenación)"""
    if "SpO2" in data and isinstance(data["SpO2"], list):
        try:
            spo2_data = []
            for entry in data["SpO2"]:
                spo2_data.append({
                    "Fecha_Hora": entry.get("minute", ""),
                    "SpO2": entry.get("value", "")
                })
            
            if spo2_data:
                df_spo2 = pd.DataFrame(spo2_data)
                df_spo2.insert(0, "Fecha", date)
                
                # Calcular estadísticas
                stats = {
                    "Estadística": ["SpO2 Promedio", "SpO2 Máximo", "SpO2 Mínimo"],
                    "Valor": [
                        df_spo2["SpO2"].mean(),
                        df_spo2["SpO2"].max(),
                        df_spo2["SpO2"].min()
                    ]
                }
                df_spo2_stats = pd.DataFrame(stats)
                
                # Escribir ambas tablas en la misma hoja
                df_spo2.to_excel(writer, sheet_name="SpO2", index=False, startrow=0)
                df_spo2_stats.to_excel(writer, sheet_name="SpO2", index=False, startrow=len(df_spo2)+3)
        except Exception as e:
            print(f"Error procesando datos SpO2: {e}")

def create_activity_zones_sheet(writer, data, date):
    """Crea hoja con zonas de actividad"""
    if "Zonas_Actividad" in data and isinstance(data["Zonas_Actividad"], list):
        try:
            zones_data = data["Zonas_Actividad"]
            if zones_data and isinstance(zones_data[0], dict):
                value = zones_data[0].get("value", {})
                df_zones = pd.DataFrame({
                    "Zona": ["Minutos Zona Activa", "Minutos Quema Grasa"],
                    "Minutos": [
                        value.get("activeZoneMinutes", 0),
                        value.get("fatBurnActiveZoneMinutes", 0)
                    ]
                })
                df_zones.insert(0, "Fecha", date)
                df_zones.to_excel(writer, sheet_name="Zonas_Actividad", index=False)
        except Exception as e:
            print(f"Error procesando zonas de actividad: {e}")

def create_goals_sheet(writer, data, date):
    """Crea hoja con metas de actividad y sueño"""
    # Metas de actividad
    if "Metas_Actividad" in data and isinstance(data["Metas_Actividad"], dict):
        try:
            activity_goals = data["Metas_Actividad"]
            df_activity_goals = pd.DataFrame({
                "Meta": ["Minutos activos", "Minutos zona activa", "Calorías", 
                        "Distancia (km)", "Pisos", "Pasos"],
                "Objetivo": [
                    activity_goals.get("activeMinutes", 0),
                    activity_goals.get("activeZoneMinutes", 0),
                    activity_goals.get("caloriesOut", 0),
                    activity_goals.get("distance", 0),
                    activity_goals.get("floors", 0),
                    activity_goals.get("steps", 0)
                ]
            })
            df_activity_goals.insert(0, "Fecha", date)
            df_activity_goals.to_excel(writer, sheet_name="Metas_Actividad", index=False)
        except Exception as e:
            print(f"Error procesando metas de actividad: {e}")
    
    # Metas de sueño
    if "Metas_Sueño" in data and isinstance(data["Metas_Sueño"], dict):
        try:
            sleep_goals = data["Metas_Sueño"]
            df_sleep_goals = pd.DataFrame({
                "Meta": ["Duración mínima sueño (min)"],
                "Objetivo": [sleep_goals.get("minDuration", 0)]
            })
            df_sleep_goals.insert(0, "Fecha", date)
            df_sleep_goals.to_excel(writer, sheet_name="Metas_Sueño", index=False)
        except Exception as e:
            print(f"Error procesando metas de sueño: {e}")

def create_devices_sheet(writer, data, date):
    """Crea hoja con información de dispositivos"""
    if "Dispositivos" in data and isinstance(data["Dispositivos"], list):
        try:
            devices = []
            for device in data["Dispositivos"]:
                devices.append({
                    "Tipo": device.get("type", ""),
                    "Versión": device.get("deviceVersion", ""),
                    "Batería": device.get("battery", ""),
                    "Nivel Batería": device.get("batteryLevel", ""),
                    "Última Sincronización": device.get("lastSyncTime", "")
                })
            
            if devices:
                df_devices = pd.DataFrame(devices)
                df_devices.insert(0, "Fecha", date)
                df_devices.to_excel(writer, sheet_name="Dispositivos", index=False)
        except Exception as e:
            print(f"Error procesando información de dispositivos: {e}")

def create_minute_activity_sheet(writer, data, date):
    """Crea hoja con datos de actividad por minuto"""
    if "Actividades" in data and isinstance(data["Actividades"], dict):
        try:
            activities = data["Actividades"]
            
            # Pasos por minuto
            if "steps" in activities and isinstance(activities["steps"], list):
                df_steps = pd.DataFrame(activities["steps"])
                if not df_steps.empty:
                    df_steps = df_steps.rename(columns={"value": "Pasos", "time": "Hora"})
                    df_steps.insert(0, "Fecha", date)
                    df_steps.to_excel(writer, sheet_name="Pasos_Minuto", index=False)
            
            # Calorías por minuto
            if "calories" in activities and isinstance(activities["calories"], list):
                df_calories = pd.DataFrame(activities["calories"])
                if not df_calories.empty:
                    df_calories = df_calories.rename(columns={"value": "Calorías", "time": "Hora"})
                    df_calories.insert(0, "Fecha", date)
                    df_calories.to_excel(writer, sheet_name="Calorias_Minuto", index=False)
            
            # Distancia por minuto
            if "distance" in activities and isinstance(activities["distance"], list):
                df_distance = pd.DataFrame(activities["distance"])
                if not df_distance.empty:
                    df_distance = df_distance.rename(columns={"value": "Distancia", "time": "Hora"})
                    df_distance.insert(0, "Fecha", date)
                    df_distance.to_excel(writer, sheet_name="Distancia_Minuto", index=False)
            
            # Elevación por minuto
            if "elevation" in activities and isinstance(activities["elevation"], list):
                df_elevation = pd.DataFrame(activities["elevation"])
                if not df_elevation.empty:
                    df_elevation = df_elevation.rename(columns={"value": "Elevación", "time": "Hora"})
                    df_elevation.insert(0, "Fecha", date)
                    df_elevation.to_excel(writer, sheet_name="Elevacion_Minuto", index=False)
        except Exception as e:
            print(f"Error procesando actividad por minuto: {e}")

def create_heart_rate_summary_sheet(writer, data, date):
    """Crea hoja con resumen de zonas de ritmo cardíaco"""
    if "Resumen_Ritmo_Cardiaco" in data and isinstance(data["Resumen_Ritmo_Cardiaco"], list):
        try:
            hr_summary = data["Resumen_Ritmo_Cardiaco"][0].get("value", {})
            zones = hr_summary.get("heartRateZones", [])
            
            if zones:
                df_zones = pd.DataFrame(zones)
                df_zones.insert(0, "Fecha", date)
                df_zones.to_excel(writer, sheet_name="Zonas_Ritmo_Cardiaco", index=False)
        except Exception as e:
            print(f"Error procesando resumen de ritmo cardíaco: {e}")

def create_biometrics_sheet(writer, data, date):
    """Crea hoja con datos biométricos"""
    resp_rate = None
    resp_list = data.get("Frecuencia_Respiratoria", [])
    if isinstance(resp_list, list) and resp_list:
        resp_rate = resp_list[0].get("value", {}).get("fullSleepSummary", {}).get("breathingRate")

    hrv_avg = None
    hrv_list = data.get("HRV", [])
    if isinstance(hrv_list, list):
        rmssd_vals = []
        for entry in hrv_list:
            for minute in entry.get("minutes", []):
                rmssd = minute.get("value", {}).get("rmssd")
                if isinstance(rmssd, (int, float)):
                    rmssd_vals.append(rmssd)
        if rmssd_vals:
            hrv_avg = sum(rmssd_vals) / len(rmssd_vals)
    biometrics = {
        "Campo": [
            "Peso", "Grasa Corporal", "IMC",
            "Ritmo Cardíaco en Reposo", "Frecuencia Respiratoria",
            "HRV RMSSD Promedio"
        ],
        "Valor": [
            data.get("Peso", "No disponible"),
            data.get("Grasa_Corporal", "No disponible"),
            data.get("IMC", "No disponible"),
            data.get("Ritmo_Cardiaco_Reposo", "No disponible"),
            resp_rate if resp_rate is not None else "No disponible",
            hrv_avg if hrv_avg is not None else "No disponible"
        ]
    }
    df_biometrics = pd.DataFrame(biometrics)
    df_biometrics.insert(0, "Fecha", date)
    df_biometrics.to_excel(writer, sheet_name="Biometricos", index=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = FitbitApp(root)
    root.mainloop()
