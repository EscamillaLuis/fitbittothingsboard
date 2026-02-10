# Fitbit to ThingsBoard

Colector y publicador de datos Fitbit. Permite obtener ritmos cardiacos, HRV, SpO2, respiracion, actividad, sueno y metricas asociadas, guardarlos en JSON/Excel y enviarlos a ThingsBoard por MQTT.

## Flujo en breve
- Preparas credenciales Fitbit (`credentials.json`) y listas los IDs a procesar (`client_ids.txt`).
- Autenticas cada usuario via OAuth; los tokens quedan en `fitbit_tokens.json`.
- Descargas datos con `single_fit.py` (GUI) o `auto_fit.py` (CLI) y se guardan en `fitbit_data/`.
- Si existe `thingsboard_tokens.json`, se publican payloads a ThingsBoard con `send_to_thingsboard.py`.

## Modos de ejecucion
- Modo intervalo (backfill): usa `auto_fit.py` con `--start-date/--end-date`. Recorre todas las fechas (por lotes `--batch-size`) y pausa entre lotes con `--interval` segundos. Al terminar el rango imprime "Backfill completado" y entra automaticamente a monitoreo continuo sin relanzar el script.
- Modo monitoreo continuo: `auto_fit.py --monitor ...` ejecuta consultas sobre el dia actual para todos los usuarios cada `--interval` segundos; la GUI `single_fit.py` tambien tiene botones Iniciar/Detener que hacen un ciclo horario similar.
- Modo manual por usuario: la GUI `single_fit.py` permite bajar un rango puntual y publicar solo para el usuario seleccionado, sin afectar a los demas.

## Archivos clave (quien los crea y para que sirven)
- `credentials.json` (generado): lo escribe `single_fit.py` al guardar un Client ID/Secret nuevos. Formato lista de objetos:
  ```json
  [
    {"client_id": "ABC123", "client_secret": "xxxxx"}
  ]
  ```
  Puedes editarlo a mano si prefieres no usar la GUI.
- `client_ids.txt` (generado/manual): `single_fit.py` agrega cada Client ID nuevo, uno por linea. `auto_fit.py` lee esta lista para procesar varios usuarios; tambien puedes pegar IDs manualmente.
- `fitbit_tokens.json` (generado): se crea/actualiza en el callback OAuth (navegador abierto por `single_fit.py`) o al reautenticar con `auto_fit.py --reauth-missing-scopes`. Contiene access/refresh tokens por Client ID. No necesitas editarlo.
- `thingsboard_tokens.json` (manual): mapa de Client ID a token de dispositivo ThingsBoard y el campo `usuario` que se incluye en los payloads. Ejemplo:
  ```json
  {
    "ABC123": {"token": "TB_DEVICE_TOKEN", "usuario": 201}
  }
  ```
  Este archivo no se genera solo; sin el no se publicara nada a ThingsBoard.
- `fitbit_data/` (generado): salidas por usuario. `single_fit.py` guarda `YYYY-MM-DD_<client>.json` y `<client>_procesado.xlsx` en `fitbit_data/<client>/`. `auto_fit.py` los anida en `fitbit_data/<client>/YYYY/MM/DD/`.
- `logs/auto_fit.log` (generado): bitacora que `auto_fit.py` escribe automaticamente (ruta personalizable con `AUTO_FIT_LOG_FILE`).

## Instalacion rapida
```bash
python -m venv .venv
.venv\\Scripts\\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```
Requisitos: Python 3.8+ y acceso a un navegador para la primera autenticacion.

## Autenticacion inicial (una vez por usuario Fitbit)
1) Ejecuta `python single_fit.py`.
2) Ingresa `Client ID` y `Client Secret` (los obtienes en https://dev.fitbit.com) y pulsa **Autenticar**.
3) Se abre el navegador, aceptas permisos Fitbit y vuelves; el token se guarda en `fitbit_tokens.json`, el ID se anade a `client_ids.txt` y las credenciales a `credentials.json`.
4) Repite por cada pulsera/usuario.

### Configuracion de la app Fitbit (requerida antes del paso 1)
- En https://dev.fitbit.com crea/edita tu aplicacion y define el Callback URL exactamente como `http://localhost:5000/callback` (coincide con `REDIRECT_URI` del proyecto).
- Activa los scopes: activity, heartrate, sleep, profile, respiratory_rate, oxygen_saturation, weight, settings. Sin oxygen_saturation o respiratory_rate no hay SpO2 ni respiracion; sin heartrate/sleep se bloquea HRV.
- Usa el Client ID y Client Secret que te muestra Fitbit; copialos en la GUI o en `credentials.json`.
- Los tokens se refrescan solos via `refresh_token`; no hace falta reautenticar salvo que Fitbit revoque permisos o falten scopes (usa `auto_fit.py --reauth-missing-scopes` para rehacerlos).

## Uso: `single_fit.py` (GUI por usuario)
Comando: `python single_fit.py`
- **Rango de fechas**: selecciona inicio/fin y pulsa **Descargar**. Por cada dia:
  - Llama a la API Fitbit, guarda `fitbit_data/<client>/YYYY-MM-DD_<client>.json`.
  - Genera `<client>_procesado.xlsx` con pestanas (ritmo cardiaco, sueno, HRV, SpO2, etc).
  - Si `thingsboard_tokens.json` tiene ese Client ID, publica a ThingsBoard.
- **Monitoreo**: botones Iniciar/Detener lanzan un bucle horario que toma el dia actual, guarda un JSON con sufijo `_monitor_<timestamp>.json` y publica en modo monitor.
Notas: levanta un servidor Flask local en `http://localhost:5000` para el callback OAuth; requiere entorno con interfaz grafica.

## Uso: `auto_fit.py` (CLI multipulsera)
Previo: asegurate de tener `client_ids.txt`, `credentials.json`, `fitbit_tokens.json` y `thingsboard_tokens.json` completos.

### Backfill por rango
```bash
python auto_fit.py --start-date 2026-02-01 --end-date 2026-02-07 --window 60 --batch-size 5 --workers 4 --interval 3000
```
- Procesa todas las fechas del rango para cada ID en `client_ids.txt`.
- Almacena JSON y Excel en `fitbit_data/<client>/YYYY/MM/DD/`.
- Publica a ThingsBoard con el host/puerto configurados.
- Al completar todos los lotes entra automaticamente en modo monitoreo continuo usando el mismo `--interval` (no es necesario relanzar).

### Monitoreo continuo
```bash
python auto_fit.py --monitor --window 60 --interval 3000
```
- Cada `interval` segundos (default 3600) reconsulta el dia actual para todos los usuarios y publica.
- Para detenerlo presiona Ctrl+C en la terminal.

### Opciones utiles
- `--window`: segundos de agregacion para series (0 o negativo envia muestras crudas).
- `--batch-size`: cuantas fechas por lote antes de pausar.
- `--workers`: hilos en paralelo (default 4 o `AUTO_FIT_WORKERS`).
- `--check-scopes <client>`: imprime scopes del token.
- `--reauth-missing-scopes`: reabre navegador para todos los usuarios sin scopes clave (oxygen_saturation, respiratory_rate, heartrate, sleep).
- `--fetch-hrv/--fetch-br --user <id> --date YYYY-MM-DD [--intraday]`: pruebas puntuales de HRV o respiracion.

## Publicacion a ThingsBoard
`send_to_thingsboard.py` arma payloads con valores estaticos (edad, peso, sueno, etc) y series temporales agregadas por `--window`. Los scripts usan:
- Host: `THINGSBOARD_MQTT_HOST` (default `thingsboard.cloud`)
- Puerto: `MQTT_PORT` (default `1883`)
- Topico: `v1/devices/me/telemetry`
El campo `usuario` de `thingsboard_tokens.json` se inserta en cada payload para identificar al usuario en el tablero.

## Variables de entorno
- `THINGSBOARD_MQTT_HOST` / `MQTT_PORT`: destino MQTT.
- `AGG_WINDOW` / `HRV_PROXY_WINDOW_SECONDS`: ventanas de agregacion (segundos).
- `AUTO_FIT_WORKERS`, `AUTO_FIT_LOG_FILE`: concurrencia y ruta de log.
- `FITBIT_API_DELAY`: pausa entre llamadas (s).
- `FITBIT_DEFAULT_TIMEZONE`: tz usada al construir timestamps.
- `HRV_PROXY_ENABLED`: habilita HRV derivado de HR.

## Recetas rapidas
- **Agregar nuevo dispositivo**: corre `single_fit.py`, ingresa ID/secret, autentica; verifica que `fitbit_tokens.json` tenga el nuevo registro y anade su token de ThingsBoard al `thingsboard_tokens.json`.
- **Faltan scopes**: `python auto_fit.py --reauth-missing-scopes` y vuelve a autorizar en el navegador.
- **Solo probar HRV intradia**: `python auto_fit.py --fetch-hrv --user ABC123 --date 2026-02-10 --intraday`.

## Estructura de salidas
- `fitbit_data/<client>/YYYY-MM-DD_<client>.json` y `_procesado.xlsx` (GUI).
- `fitbit_data/<client>/YYYY/MM/DD/` con JSON y Excel anidados (CLI).
- `logs/auto_fit.log` para diagnosticos de ejecucion automatica.
