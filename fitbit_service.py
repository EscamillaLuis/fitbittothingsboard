from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import certifi
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests import Response
from requests.sessions import Session
from urllib.parse import urlparse
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration constants shared across the project
# ---------------------------------------------------------------------------

TOKEN_FILE = "fitbit_tokens.json"
CREDENTIALS_FILE = "credentials.json"
CLIENT_IDS_FILE = "client_ids.txt"
DATA_DIR = "fitbit_data"
API_DELAY = float(os.getenv("FITBIT_API_DELAY", "1"))
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
REDIRECT_URI = "http://localhost:5000/callback"
THINGSBOARD_MQTT_HOST = os.getenv("THINGSBOARD_MQTT_HOST", "thingsboard.cloud")
THINGSBOARD_MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
AGG_WINDOW_SECONDS = int(os.getenv("AGG_WINDOW", "60"))
DEFAULT_TIMEOUT: Tuple[int, int] = (5, 30)
MAX_WORKERS = int(os.getenv("FITBIT_MAX_WORKERS", "4"))
OAUTH_SCOPES: List[str] = [
    "activity",
    "heartrate",
    "sleep",
    "profile",
    "respiratory_rate",
    "oxygen_saturation",
    "weight",
    "settings",
]
REQUIRED_SCOPES = {scope.lower() for scope in OAUTH_SCOPES}
USE_INTRADAY = os.getenv("USE_INTRADAY", "true").lower() == "true"
HRV_PROXY_ENABLED = os.getenv("HRV_PROXY_ENABLED", "true").lower() == "true"
HRV_PROXY_WINDOW_SECONDS = int(os.getenv("HRV_PROXY_WINDOW_SECONDS", "300"))
HRV_PROXY_STEP_SECONDS = int(os.getenv("HRV_PROXY_STEP_SECONDS", "60"))
HRV_PROXY_MIN_SAMPLES = int(os.getenv("HRV_PROXY_MIN_SAMPLES", "60"))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Session and rate limit management
# ---------------------------------------------------------------------------

def create_session() -> Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = certifi.where()
    return session

def timed_request(
    session: Session,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: Tuple[int, int] = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> Response:
    start = time.perf_counter()
    response = session.request(method, url, headers=headers, timeout=timeout, **kwargs)
    total = time.perf_counter() - start
    host = urlparse(url).hostname or "?"
    logger.debug("%s %s -> %s (%.0fms)", method, host, response.status_code, total * 1000)
    return response

class RateLimiter:
    def __init__(self, interval: float) -> None:
        self.interval = max(interval, 0.0)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = now - self._last_call
            if delta < self.interval:
                time.sleep(self.interval - delta)
                now = time.monotonic()
            self._last_call = now

class SessionManager:

    def __init__(self) -> None:
        self._local = threading.local()

    def get(self) -> Session:
        if not hasattr(self._local, "session"):
            self._local.session = create_session()
        return self._local.session
SESSION_MANAGER = SessionManager()
RATE_LIMITER = RateLimiter(API_DELAY)

# ---------------------------------------------------------------------------
# Token handling helpers
# ---------------------------------------------------------------------------

class TokenStore:
    def __init__(self, path: str = TOKEN_FILE) -> None:
        self.path = path
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self._tokens = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            self._tokens = {}

    def get(self, client_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._tokens.get(client_id, {}))

    def set(self, client_id: str, token: Dict[str, Any]) -> None:
        with self._lock:
            self._tokens[client_id] = token
            self._save()

    def all(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._tokens.items()}

    def _save(self) -> None:
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._tokens, fh, indent=4)
        os.replace(tmp_path, self.path)

TOKEN_STORE = TokenStore()

def load_tokens() -> Dict[str, Dict[str, Any]]:
    return TOKEN_STORE.all()

def save_token(client_id: str, new_token: Dict[str, Any]) -> None:
    TOKEN_STORE.set(client_id, new_token)

def _normalize_scopes_set(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, dict):
        return {str(k).strip().lower() for k in raw.keys()}
    if isinstance(raw, list):
        return {str(v).strip().lower() for v in raw}
    scope_str = str(raw).strip()
    if "=" in scope_str and "," in scope_str:
        import re
        names = [m.group(1) for m in re.finditer(r"([A-Za-z_]+)\s*=", scope_str)]
        return {n.strip().lower() for n in names if n}
    scope_str = scope_str.replace(",", " ")
    return {token.strip().lower() for token in scope_str.split() if token.strip()}

def token_has_scope(client_id: str, scope: str) -> bool:
    token_data = TOKEN_STORE.get(client_id)
    scopes = _normalize_scopes_set(token_data.get("scope", ""))
    return scope.strip().lower() in scopes

def check_scopes(client_id: str) -> bool:
    token_data = TOKEN_STORE.get(client_id)
    if not token_data:
        return False
    scopes = _normalize_scopes_set(token_data.get("scope", ""))
    return REQUIRED_SCOPES.issubset(scopes)

def save_credentials(client_id: str, client_secret: str) -> None:
    creds: List[Dict[str, str]] = []
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, "r", encoding="utf-8") as fh:
                creds = json.load(fh)
        except json.JSONDecodeError:
            creds = []
    if any(c.get("client_id") == client_id for c in creds):
        return
    creds.append({"client_id": client_id, "client_secret": client_secret})
    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as fh:
        json.dump(creds, fh, indent=4)

def refresh_access_token(
    client_id: str,
    client_secret: str,
    *,
    token_store: TokenStore | None = None,
    session: Session | None = None,
    rate_limiter: RateLimiter | None = None,
) -> Optional[str]:
    store = token_store or TOKEN_STORE
    token_data = store.get(client_id)
    if not token_data:
        print(f"No hay token para {client_id}. Necesita autenticación inicial.")
        return None
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        print(f"No hay refresh_token para {client_id}.")
        return None
    raw_scope = token_data.get("scope", "")
    scope_set = _normalize_scopes_set(raw_scope)
    if not REQUIRED_SCOPES.issubset(scope_set):
        missing = REQUIRED_SCOPES - scope_set
        strict = os.getenv("FITBIT_STRICT_SCOPES", "false").lower() == "true"
        if strict:
            print(f"Error: faltan scopes para {client_id}: {', '.join(sorted(missing))}")
            return None
        logger.warning(
            "Continuando con scopes parciales para %s. Faltan: %s",
            client_id,
            ", ".join(sorted(missing)),
        )
    scope_str = " ".join(sorted(scope_set)) if scope_set else " ".join(OAUTH_SCOPES)
    auth_header = b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "scope": scope_str,
    }
    session = session or create_session()
    limiter = rate_limiter or RATE_LIMITER
    try:
        limiter.wait()
        response = timed_request(session, "POST", TOKEN_URL, headers=headers, data=data)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error de conexión refrescando token: {exc}")
        return None
    new_token = response.json()
    if "access_token" not in new_token:
        print(f"Respuesta inesperada refrescando token: {new_token}")
        return None
    if "scope" not in new_token:
        new_token["scope"] = scope_str
    store.set(client_id, new_token)
    return new_token["access_token"]

def reauthenticate_scopes(client_id: str, client_secret: str) -> Optional[str]:
    from requests_oauthlib import OAuth2Session
    print(f"Iniciando reautenticación para {client_id} con scopes completos...")
    session = OAuth2Session(client_id, redirect_uri=REDIRECT_URI, scope=OAUTH_SCOPES)
    authorization_url, _ = session.authorization_url("https://www.fitbit.com/oauth2/authorize")
    print("Abre la siguiente URL en tu navegador y autoriza el acceso:")
    print(authorization_url)
    import webbrowser
    webbrowser.open(authorization_url)
    redirect_response = input("Después de autorizar, pega aquí la URL completa de redirección:\n").strip()
    try:
        token = session.fetch_token(
            TOKEN_URL,
            client_secret=client_secret,
            authorization_response=redirect_response,
        )
    except Exception as exc:
        print(f"Error durante la reautenticación: {exc}")
        return None
    save_token(client_id, token)
    print(f"Nuevo token guardado para {client_id}")
    return token.get("access_token")

def debug_print_token_scopes(user_id: str) -> None:
    token_data = TOKEN_STORE.get(user_id)
    if not token_data:
        print(f"No se encontró token para {user_id}")
        return
    try:
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as fh:
            creds = {c["client_id"]: c["client_secret"] for c in json.load(fh)}
    except Exception as exc:
        print(f"Error cargando credenciales: {exc}")
        return
    client_secret = creds.get(user_id)
    if not client_secret:
        print(f"No se encontró client_secret para {user_id}")
        return
    session = create_session()
    auth_header = b64encode(f"{user_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"token": token_data.get("access_token", "")}
    try:
        resp = timed_request(
            session,
            "POST",
            "https://api.fitbit.com/1.1/oauth2/introspect",
            headers=headers,
            data=data,
        )
        if resp.status_code == 200:
            info = resp.json()
            scopes = _normalize_scopes_set(info.get("scope", ""))
            print(f"Scopes actuales para {user_id}: {', '.join(sorted(scopes))}")
        else:
            print(f"Error introspectando token ({resp.status_code})")
    except Exception as exc:
        print(f"Error introspectando token: {exc}")

# ---------------------------------------------------------------------------
# Fitbit API client with concurrency helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EndpointSpec:
    name: str
    url: str
    key: Optional[str] = None
    transform: Optional[Callable[[Any], Any]] = None

class FitbitAPIClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        token_store: TokenStore | None = None,
        session_manager: SessionManager | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_store = token_store or TOKEN_STORE
        self.session_manager = session_manager or SESSION_MANAGER
        self.rate_limiter = rate_limiter or RateLimiter(API_DELAY)
        self._headers: Optional[Dict[str, str]] = None
        self._token: Optional[str] = None
        self._token_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Token handling
    # ------------------------------------------------------------------

    def ensure_token(self) -> Optional[str]:
        with self._token_lock:
            if self._token:
                return self._token
            token = refresh_access_token(
                self.client_id,
                self.client_secret,
                token_store=self.token_store,
                session=self.session_manager.get(),
                rate_limiter=self.rate_limiter,
            )
            if token:
                self._token = token
                self._headers = {"Authorization": f"Bearer {token}"}
            return self._token

    def refresh_token(self) -> Optional[str]:
        with self._token_lock:
            token = refresh_access_token(
                self.client_id,
                self.client_secret,
                token_store=self.token_store,
                session=self.session_manager.get(),
                rate_limiter=self.rate_limiter,
            )
            if token:
                self._token = token
                self._headers = {"Authorization": f"Bearer {token}"}
            return self._token
        
    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def request(self, url: str, key: Optional[str] = None) -> Tuple[Optional[Any], int]:
        token = self.ensure_token()
        if not token:
            return None, 401
        headers = dict(self._headers or {})
        attempt = 0
        backoff = 1.0
        while attempt < 3:
            attempt += 1
            self.rate_limiter.wait()
            session = self.session_manager.get()
            try:
                response = timed_request(session, "GET", url, headers=headers)
            except Exception as exc:
                if attempt >= 3:
                    logger.error("Error en petición a %s: %s", url, exc)
                    return None, 0
                time.sleep(backoff)
                backoff *= 2
                continue
            status = response.status_code
            if status == 200:
                payload = response.json()
                data = payload.get(key) if key else payload
                return data, status
            if status == 401 and attempt == 1:
                new_token = self.refresh_token()
                if not new_token:
                    return None, status
                headers = dict(self._headers or {})
                continue
            if status == 429:
                retry_after = int(response.headers.get("Retry-After", "1"))
                wait_time = max(retry_after, math.ceil(backoff))
                logger.warning("Rate limit alcanzado. Esperando %ss", wait_time)
                time.sleep(wait_time)
                backoff *= 2
                continue
            if status >= 500:
                logger.warning("Error %s en %s. Reintentando en %.1fs", status, url, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            if status in (403, 404):
                return None, status
            logger.error("Error en %s: %s %s", url, status, response.text[:200])
            return None, status
        return None, status  # type: ignore[UnboundLocalError]
    
    # ------------------------------------------------------------------
    # High level helpers
    # ------------------------------------------------------------------

    def fetch_endpoints(self, specs: List[EndpointSpec]) -> Dict[str, Any]:
        if not specs:
            return {}
        results: Dict[str, Any] = {}
        specs = list(specs)
        chunk_size = max(1, min(MAX_WORKERS, len(specs)))
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=chunk_size) as executor:
            for start in range(0, len(specs), chunk_size):
                chunk = specs[start:start + chunk_size]
                future_map = {
                    executor.submit(self._fetch_single, spec): spec for spec in chunk
                }
                for future in as_completed(future_map):
                    spec = future_map[future]
                    try:
                        data, status = future.result()
                    except Exception as exc:
                        logger.error("Fallo obteniendo %s: %s", spec.name, exc)
                        data, status = None, 0
                    if status == 200 and spec.transform:
                        try:
                            data = spec.transform(data)
                        except Exception as exc:
                            logger.error("Error transformando %s: %s", spec.name, exc)
                            data = None
                    results[spec.name] = data if data is not None else "No disponible"
                if start + chunk_size < len(specs):
                    self.rate_limiter.wait()
        return results

    def _fetch_single(self, spec: EndpointSpec) -> Tuple[Optional[Any], int]:
        return self.request(spec.url, spec.key)

# ---------------------------------------------------------------------------
# HRV proxy helpers (self-contained implementation)
# ---------------------------------------------------------------------------

def parse_times_and_hr_from_data(data: Dict[str, Any]) -> Tuple[List[datetime], List[int]]:
    heart = data.get("Ritmo_Cardiaco", {})
    dataset = heart.get("activities-heart-intraday", {}).get("dataset") if isinstance(heart, dict) else None
    if not dataset and isinstance(heart, dict):
        dataset = heart.get("dataset")
    times: List[datetime] = []
    values: List[int] = []
    if not isinstance(dataset, list):
        return times, values
    for entry in dataset:
        try:
            t = entry.get("time")
            v = entry.get("value")
            if not t or not isinstance(v, (int, float)):
                continue
            dt = datetime.strptime(t, "%H:%M:%S")
            times.append(dt)
            values.append(int(v))
        except Exception:
            continue
    return times, values

@dataclass
class HRVProxyAggregate:
    rmssd_proxy_ms: Optional[float]
    sdnn_proxy_ms: Optional[float]
    pnn50_proxy: Optional[float]
    valid_samples: int

def aggregate_proxies_from_hr(hr_values: List[int]) -> HRVProxyAggregate:
    if not hr_values:
        return HRVProxyAggregate(None, None, None, 0)
    diffs = [abs(hr_values[i] - hr_values[i - 1]) for i in range(1, len(hr_values))]
    if not diffs:
        return HRVProxyAggregate(None, None, None, len(hr_values))
    squared = [d ** 2 for d in diffs]
    rmssd = math.sqrt(sum(squared) / len(squared))
    mean_diff = mean(diffs)
    sdnn = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    pnn50 = sum(1 for d in diffs if d > 50) / len(diffs)
    return HRVProxyAggregate(rmssd, sdnn, pnn50, len(hr_values))

def windowed_proxies(
    times: List[datetime],
    hr_values: List[int],
    *,
    window_seconds: int,
    step_seconds: int,
    min_samples: int,
) -> List[Dict[str, Any]]:
    if not times or not hr_values or len(times) != len(hr_values):
        return []
    series: List[Dict[str, Any]] = []
    base = datetime.combine(datetime.today(), datetime.min.time())
    zipped = sorted(zip(times, hr_values), key=lambda item: item[0])
    time_list = [t for t, _ in zipped]
    value_list = [v for _, v in zipped]
    idx = 0
    while idx < len(zipped):
        window_start = time_list[idx]
        window_end = window_start + timedelta(seconds=window_seconds)
        bucket: List[int] = []
        j = idx
        while j < len(zipped) and time_list[j] < window_end:
            bucket.append(value_list[j])
            j += 1
        if len(bucket) >= min_samples:
            agg = aggregate_proxies_from_hr(bucket)
            series.append(
                {
                    "window_start": (
                        base
                        + timedelta(
                            hours=window_start.hour,
                            minutes=window_start.minute,
                            seconds=window_start.second,
                        )
                    ).isoformat(),
                    "window_end": (
                        base
                        + timedelta(
                            hours=window_end.hour,
                            minutes=window_end.minute,
                            seconds=window_end.second,
                        )
                    ).isoformat(),
                    "rmssd_proxy_ms": agg.rmssd_proxy_ms,
                    "sdnn_proxy_ms": agg.sdnn_proxy_ms,
                    "pnn50_proxy": agg.pnn50_proxy,
                    "n": agg.valid_samples,
                }
            )
        next_threshold = window_start + timedelta(seconds=step_seconds)
        while idx < len(zipped) and time_list[idx] < next_threshold:
            idx += 1
    return series

def compute_and_attach_hrv_proxy(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        times, hr_values = parse_times_and_hr_from_data(data)
        if not times:
            return data
        agg = aggregate_proxies_from_hr(hr_values)
        series = windowed_proxies(
            times,
            hr_values,
            window_seconds=HRV_PROXY_WINDOW_SECONDS,
            step_seconds=HRV_PROXY_STEP_SECONDS,
            min_samples=HRV_PROXY_MIN_SAMPLES,
        )
        data["HRV_Proxy"] = {
            "rmssd_proxy_ms": agg.rmssd_proxy_ms,
            "sdnn_proxy_ms": agg.sdnn_proxy_ms,
            "pnn50_proxy": agg.pnn50_proxy,
            "valid_samples": agg.valid_samples,
        }
        data["HRV_Proxy_Series"] = series
    except Exception as exc:
        logger.error("Error calculando HRV Proxy: %s", exc)
    return data
# ---------------------------------------------------------------------------
# HRV/Respiratory endpoints
# ---------------------------------------------------------------------------
def _parse_hrv_minutes(raw_minutes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    minutes: List[Dict[str, Any]] = []
    for entry in raw_minutes or []:
        minute_ts = entry.get("minute")
        val = entry.get("value", {})
        if not minute_ts:
            continue
        try:
            dt = datetime.fromisoformat(minute_ts)
            minutes.append(
                {
                    "time": dt.isoformat(),
                    "rmssd": val.get("rmssd"),
                    "coverage": val.get("coverage"),
                    "lf": val.get("lf"),
                    "hf": val.get("hf"),
                }
            )
        except Exception:
            continue
    return minutes

def fetch_hrv_data(
    client_id: str,
    client_secret: str,
    date: str,
    *,
    intraday: Optional[bool] = None,
    client: FitbitAPIClient | None = None,
) -> Optional[Dict[str, Any]]:
    if intraday is None:
        intraday = USE_INTRADAY
    fitbit = client or FitbitAPIClient(client_id, client_secret)
    if intraday:
        url = f"https://api.fitbit.com/1/user/-/hrv/date/{date}/all.json"
        data, status = fitbit.request(url, "hrv")
        if status == 403:
            print("Falta acceso intradía aprobado por Fitbit. Usando resumen diario.")
            intraday = False
        elif status == 200:
            return data
    if not intraday:
        url = f"https://api.fitbit.com/1/user/-/hrv/date/{date}.json"
        data, status = fitbit.request(url, "hrv")
        if status == 200:
            return data
    today = datetime.now().strftime("%Y-%m-%d")
    if date == today:
        print("HRV sin datos todavía")
    return None

def fetch_hrv_intraday(
    client_id: str,
    client_secret: str,
    date: str,
    *,
    client: FitbitAPIClient | None = None,
) -> Optional[Dict[str, Any]]:
    fitbit = client or FitbitAPIClient(client_id, client_secret)
    url = f"https://api.fitbit.com/1/user/-/hrv/date/{date}/all.json"
    data, status = fitbit.request(url, "hrv")
    if status != 200 or not data:
        return None
    minutes = []
    for day in data:
        minutes.extend(_parse_hrv_minutes(day.get("minutes", [])))
    return {"minutes": minutes}

def fetch_hrv_intraday_range(
    client_id: str,
    client_secret: str,
    start_date: str,
    end_date: str,
    *,
    client: FitbitAPIClient | None = None,
) -> Optional[List[Dict[str, Any]]]:
    fitbit = client or FitbitAPIClient(client_id, client_secret)
    url = f"https://api.fitbit.com/1/user/-/hrv/date/{start_date}/{end_date}/all.json"
    data, status = fitbit.request(url, "hrv")
    if status != 200 or not data:
        return None
    result: List[Dict[str, Any]] = []
    for day in data:
        minutes = _parse_hrv_minutes(day.get("minutes", []))
        result.append({"date": day.get("dateTime"), "minutes": minutes})
    return result

def fetch_br_data(
    client_id: str,
    client_secret: str,
    date: str,
    *,
    intraday: Optional[bool] = None,
    client: FitbitAPIClient | None = None,

) -> Optional[Dict[str, Any]]:
    if intraday is None:
        intraday = USE_INTRADAY
    fitbit = client or FitbitAPIClient(client_id, client_secret)
    if intraday:
        url = f"https://api.fitbit.com/1/user/-/br/date/{date}/all.json"
        data, status = fitbit.request(url, "br")
        if status == 403:
            print("Falta acceso intradía aprobado por Fitbit para respiración. Usando resumen.")
            intraday = False
        elif status == 200:
            return data
    if not intraday:
        url = f"https://api.fitbit.com/1/user/-/br/date/{date}.json"
        data, status = fitbit.request(url, "br")
        if status == 200:
            return data
    today = datetime.now().strftime("%Y-%m-%d")
    if date == today:
        print("Frecuencia respiratoria sin datos todavía")
    return None
# ---------------------------------------------------------------------------
# High level orchestration for a full data download
# ---------------------------------------------------------------------------
def _chunked(sequence: List[Any], size: int) -> Iterator[List[Any]]:
    for start in range(0, len(sequence), size):
        yield sequence[start:start + size]

def get_fitbit_data(
    client_id: str,
    client_secret: str,
    date: str,
    *,
    client: FitbitAPIClient | None = None,
) -> Optional[Dict[str, Any]]:
    fitbit = client or FitbitAPIClient(client_id, client_secret)
    if not fitbit.ensure_token():
        return None
    result: Dict[str, Any] = {"Fecha": date, "ID_Cliente": client_id}
    profile, status = fitbit.request("https://api.fitbit.com/1/user/-/profile.json", "user")
    if status == 200 and isinstance(profile, dict):
        result.update(
            {
                "ID_Usuario": profile.get("encodedId"),
                "Nombre": profile.get("fullName"),
                "Edad": profile.get("age"),
            }
        )
    has_oxygen_scope = token_has_scope(client_id, "oxygen_saturation")
    has_weight_scope = token_has_scope(client_id, "weight")
    endpoint_specs: List[EndpointSpec] = [
        EndpointSpec(
            "Ritmo_Cardiaco",
            f"https://api.fitbit.com/1/user/-/activities/heart/date/{date}/1d/1sec.json",
        ),
        EndpointSpec(
            "Resumen_Actividades",
            f"https://api.fitbit.com/1/user/-/activities/date/{date}.json",
            "summary",
        ),
    ]
    if has_oxygen_scope:
        endpoint_specs.append(
            EndpointSpec(
                "SpO2",
                f"https://api.fitbit.com/1/user/-/spo2/date/{date}/all.json",
                "minutes",
            )
        )
    endpoint_specs.extend(
        [
            EndpointSpec("Pasos", f"https://api.fitbit.com/1/user/-/activities/steps/date/{date}/1d.json", "activities-steps"),
            EndpointSpec(
                "Calorias",
                f"https://api.fitbit.com/1/user/-/activities/calories/date/{date}/1d.json",
                "activities-calories",
            ),
            EndpointSpec(
                "Distancia",
                f"https://api.fitbit.com/1/user/-/activities/distance/date/{date}/1d.json",
                "activities-distance",
            ),
            EndpointSpec(
                "Pisos",
                f"https://api.fitbit.com/1/user/-/activities/floors/date/{date}/1d.json",
                "activities-floors",
            ),
            EndpointSpec(
                "Zonas_Actividad",
                f"https://api.fitbit.com/1/user/-/activities/active-zone-minutes/date/{date}/1d.json",
                "activities-active-zone-minutes",
            ),
            EndpointSpec(
                "Metas_Actividad",
                "https://api.fitbit.com/1/user/-/activities/goals/daily.json",
                "goals",
            ),
            EndpointSpec(
                "Resumen_Sueño",
                f"https://api.fitbit.com/1.2/user/-/sleep/date/{date}.json",
                "sleep",
            ),
            EndpointSpec(
                "Metas_Sueño",
                "https://api.fitbit.com/1/user/-/sleep/goal.json",
                "goal",
            ),
            EndpointSpec(
                "Resumen_Ritmo_Cardiaco",
                f"https://api.fitbit.com/1/user/-/activities/heart/date/{date}/1d.json",
                "activities-heart",
            ),
            EndpointSpec(
                "Ritmo_Cardiaco_Reposo",
                f"https://api.fitbit.com/1/user/-/activities/heart/date/{date}/1d.json",
                "restingHeartRate",
            ),
        ]
    )
    if has_weight_scope:
        endpoint_specs.extend(
            [
                EndpointSpec(
                    "Peso",
                    f"https://api.fitbit.com/1/user/-/body/log/weight/date/{date}.json",
                    "weight",
                ),
                EndpointSpec(
                    "Grasa_Corporal",
                    f"https://api.fitbit.com/1/user/-/body/log/fat/date/{date}.json",
                    "fat",
                ),
                EndpointSpec(
                    "IMC",
                    f"https://api.fitbit.com/1/user/-/body/log/bmi/date/{date}.json",
                    "bmi",
                ),
            ]
        )
    if check_scopes(client_id):
        endpoint_specs.append(
            EndpointSpec("Dispositivos", "https://api.fitbit.com/1/user/-/devices.json")
        )
    results = fitbit.fetch_endpoints(endpoint_specs)
    result.update(results)
    # Actividades intradía específicas
    intraday = {}
    intraday_specs = [
        EndpointSpec(
            resource,
            f"https://api.fitbit.com/1/user/-/activities/{resource}/date/{date}/1d/1min.json",
            f"activities-{resource}-intraday",
        )
        for resource in ("steps", "calories", "distance", "elevation")
    ]
    intraday_results = fitbit.fetch_endpoints(intraday_specs)
    for key, value in intraday_results.items():
        if isinstance(value, dict):
            intraday[key] = value.get("dataset", "No disponible")
        else:
            intraday[key] = value
    result["Actividades"] = intraday
    result["HRV"] = fetch_hrv_data(client_id, client_secret, date, intraday=False, client=fitbit)
    hrv_intraday = fetch_hrv_intraday(client_id, client_secret, date, client=fitbit)
    if hrv_intraday:
        result["HRV_intraday"] = hrv_intraday
    result["Frecuencia_Respiratoria"] = fetch_br_data(client_id, client_secret, date, client=fitbit)
    if HRV_PROXY_ENABLED:
        result = compute_and_attach_hrv_proxy(result)
    return result
# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_daily_data(client_id: str, data: Dict[str, Any], *, data_dir: str = DATA_DIR) -> bool:
    client_dir = os.path.join(data_dir, client_id)
    os.makedirs(client_dir, exist_ok=True)
    filename = f"{data['Fecha']}_{client_id}.json"
    path = os.path.join(client_dir, filename)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)
        return True
    except Exception as exc:
        print(f"Error al guardar datos: {exc}")
        return False
# ---------------------------------------------------------------------------
# Excel export helpers
# ---------------------------------------------------------------------------
def _safe_dataframe(data: Any) -> pd.DataFrame:
    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

def _write_sheet(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame, startrow: int = 0) -> None:
    if df.empty:
        return
    df.to_excel(writer, sheet_name=sheet, index=False, startrow=startrow)

def create_client_info_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    info = {
        "Campo": ["Usuario", "Cliente", "Fecha"],
        "Valor": [data.get("Nombre"), data.get("ID_Cliente"), date],
    }
    df = pd.DataFrame(info)
    _write_sheet(writer, "Cliente", df)

def create_heart_rate_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    heart = data.get("Ritmo_Cardiaco", {}) or {}
    dataset = None
    if isinstance(heart, dict):
        dataset = heart.get("activities-heart-intraday", {}).get("dataset") or heart.get("dataset")
    df_dataset = _safe_dataframe(dataset)
    if not df_dataset.empty:
        df_dataset.insert(0, "Fecha", date)
        _write_sheet(writer, "Ritmo_Cardiaco", df_dataset)
    summary = data.get("Resumen_Ritmo_Cardiaco")
    activities_summary = None
    if isinstance(summary, dict):
        activities_summary = summary.get("activities-heart")
        if activities_summary is None:
            activities_summary = summary
    elif isinstance(summary, list):
        activities_summary = summary
    df_summary = _safe_dataframe(activities_summary)
    if not df_summary.empty:
        _write_sheet(writer, "Resumen_RC", df_summary)

def create_respiratory_rate_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    resp = data.get("Frecuencia_Respiratoria") or {}
    df_resp = _safe_dataframe(resp)
    if not df_resp.empty:
        _write_sheet(writer, "Respiracion", df_resp)

def create_activity_summary_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    summary = data.get("Resumen_Actividades") or {}
    df_summary = _safe_dataframe(summary)
    if not df_summary.empty:
        _write_sheet(writer, "Actividades", df_summary)

def create_sleep_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    sleep = data.get("Resumen_Sueño") or []
    sleep_entry = sleep[0] if isinstance(sleep, list) and sleep else {}
    if sleep_entry:
        info = {
            "Campo": [
                "Hora de inicio",
                "Hora de fin",
                "Duración (min)",
                "Minutos dormido",
                "Minutos despierto",
                "Minutos para dormirse",
                "Eficiencia",
                "Tiempo en cama (min)",
            ],
            "Valor": [
                sleep_entry.get("startTime"),
                sleep_entry.get("endTime"),
                round(sleep_entry.get("duration", 0) / 60000, 1),
                sleep_entry.get("minutesAsleep"),
                sleep_entry.get("minutesAwake"),
                sleep_entry.get("minutesToFallAsleep"),
                sleep_entry.get("efficiency"),
                sleep_entry.get("timeInBed"),
            ],
        }
        df_info = pd.DataFrame(info)
        _write_sheet(writer, "Sueño", df_info, startrow=0)
        levels = sleep_entry.get("levels", {}).get("data", [])
        df_levels = _safe_dataframe(levels)
        if not df_levels.empty:
            _write_sheet(writer, "Niveles_Sueño", df_levels)

def create_hrv_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    daily_rows: List[Dict[str, Any]] = []
    hrv_summary = data.get("HRV")
    if isinstance(hrv_summary, list) and hrv_summary:
        v = (hrv_summary[0] or {}).get("value", {})
        daily_rows.append({
            "Fecha": date,
            "dailyRmssd": v.get("dailyRmssd"),
            "deepRmssd": v.get("deepRmssd"),
        })
    df_daily = pd.DataFrame(daily_rows)
    _write_sheet(writer, "HRV", df_daily)
    intraday = data.get("HRV_intraday") or {}
    minutes = intraday.get("minutes") if isinstance(intraday, dict) else None
    if isinstance(minutes, list) and minutes:
        df_minutes = pd.DataFrame(minutes)
        _write_sheet(writer, "HRV", df_minutes, startrow=len(df_daily) + 3 if not df_daily.empty else 3)

def create_hrv_proxy_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    agg = data.get("HRV_Proxy") or {}
    df_agg = pd.DataFrame(
        [
            {
                "Fecha": date,
                "rmssd_proxy_ms": agg.get("rmssd_proxy_ms"),
                "sdnn_proxy_ms": agg.get("sdnn_proxy_ms"),
                "pnn50_proxy": agg.get("pnn50_proxy"),
                "valid_samples": agg.get("valid_samples"),
                "ventana_s": HRV_PROXY_WINDOW_SECONDS,
                "paso_s": HRV_PROXY_STEP_SECONDS,
                "min_samples": HRV_PROXY_MIN_SAMPLES,
            }
        ]
    )
    _write_sheet(writer, "HRV_PROXY", df_agg)
    series = data.get("HRV_Proxy_Series") or []
    df_series = _safe_dataframe(series)
    if not df_series.empty:
        _write_sheet(writer, "HRV_PROXY", df_series, startrow=len(df_agg) + 3)

def create_spo2_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    spo2 = data.get("SpO2")
    if isinstance(spo2, list) and spo2:
        df = pd.DataFrame(
            {
                "Fecha_Hora": [entry.get("minute") for entry in spo2],
                "SpO2": [entry.get("value") for entry in spo2],
            }
        )
        df.insert(0, "Fecha", date)
        _write_sheet(writer, "SpO2", df)
        stats = {
            "Estadística": ["SpO2 Promedio", "SpO2 Máximo", "SpO2 Mínimo"],
            "Valor": [df["SpO2"].mean(), df["SpO2"].max(), df["SpO2"].min()],
        }
        _write_sheet(writer, "SpO2", pd.DataFrame(stats), startrow=len(df) + 3)

def create_activity_zones_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    zones = data.get("Zonas_Actividad") or {}
    df_zones = _safe_dataframe(zones)
    if not df_zones.empty:
        _write_sheet(writer, "Zonas_Actividad", df_zones)

def create_goals_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    goals = data.get("Metas_Actividad") or {}
    df_goals = _safe_dataframe(goals)
    if not df_goals.empty:
        _write_sheet(writer, "Metas", df_goals)

def create_devices_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    devices = data.get("Dispositivos") or []
    df_devices = _safe_dataframe(devices)
    if not df_devices.empty:
        _write_sheet(writer, "Dispositivos", df_devices)

def create_minute_activity_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    activities = data.get("Actividades") or {}
    for resource, dataset in activities.items():
        df_dataset = _safe_dataframe(dataset)
        if df_dataset.empty:
            continue
        sheet_name = f"{resource}_min"
        _write_sheet(writer, sheet_name[:31], df_dataset)

def create_heart_rate_summary_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    summary = data.get("Resumen_Ritmo_Cardiaco") or {}
    df_summary = _safe_dataframe(summary)
    if not df_summary.empty:
        _write_sheet(writer, "Resumen_RC", df_summary)

def create_biometrics_sheet(writer: pd.ExcelWriter, data: Dict[str, Any], date: str) -> None:
    entries = []
    for key in ("Peso", "Grasa_Corporal", "IMC"):
        value = data.get(key)
        if value and isinstance(value, list):
            entries.extend(value)
    df_entries = _safe_dataframe(entries)
    if not df_entries.empty:
        _write_sheet(writer, "Biometria", df_entries)

SHEET_BUILDERS: List[Tuple[str, Callable[[pd.ExcelWriter, Dict[str, Any], str], None]]] = [
    ("Cliente", create_client_info_sheet),
    ("Ritmo_Cardiaco", create_heart_rate_sheet),
    ("Respiracion", create_respiratory_rate_sheet),
    ("Actividades", create_activity_summary_sheet),
    ("Sueño", create_sleep_sheet),
    ("HRV", create_hrv_sheet),
    ("HRV_PROXY", create_hrv_proxy_sheet),
    ("SpO2", create_spo2_sheet),
    ("Zonas_Actividad", create_activity_zones_sheet),
    ("Metas", create_goals_sheet),
    ("Dispositivos", create_devices_sheet),
    ("Minutales", create_minute_activity_sheet),
    ("Resumen_RC", create_heart_rate_summary_sheet),
    ("Biometria", create_biometrics_sheet),
]

def export_json_to_excel_single(client_id: str, date: str, data_dir: str = DATA_DIR) -> None:
    client_path = os.path.join(data_dir, client_id)
    json_path = os.path.join(client_path, f"{date}_{client_id}.json")
    output_excel = os.path.join(client_path, f"{date}_{client_id}_procesado.xlsx")
    if not os.path.exists(json_path):
        print(f"Archivo no encontrado: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        print(f"Datos no válidos en {json_path}")
        return
    os.makedirs(client_path, exist_ok=True)
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for _, builder in SHEET_BUILDERS:
            builder(writer, data, date)

__all__ = [
    "TOKEN_FILE",
    "CREDENTIALS_FILE",
    "CLIENT_IDS_FILE",
    "DATA_DIR",
    "API_DELAY",
    "TOKEN_URL",
    "REDIRECT_URI",
    "THINGSBOARD_MQTT_HOST",
    "THINGSBOARD_MQTT_PORT",
    "AGG_WINDOW_SECONDS",
    "DEFAULT_TIMEOUT",
    "OAUTH_SCOPES",
    "REQUIRED_SCOPES",
    "USE_INTRADAY",
    "HRV_PROXY_ENABLED",
    "HRV_PROXY_WINDOW_SECONDS",
    "HRV_PROXY_STEP_SECONDS",
    "HRV_PROXY_MIN_SAMPLES",
    "load_tokens",
    "save_token",
    "check_scopes",
    "token_has_scope",
    "refresh_access_token",
    "reauthenticate_scopes",
    "debug_print_token_scopes",
    "get_fitbit_data",
    "fetch_hrv_data",
    "fetch_hrv_intraday",
    "fetch_hrv_intraday_range",
    "fetch_br_data",
    "save_daily_data",
    "export_json_to_excel_single",
    "compute_and_attach_hrv_proxy",
    "FitbitAPIClient",
    "TokenStore",
    "SHEET_BUILDERS",
]
