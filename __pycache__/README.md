# Fitbit to ThingsBoard

Python application for downloading data from multiple Fitbit trackers and publishing it to [ThingsBoard](https://thingsboard.io/) via MQTT.

## Requirements

- Python 3.8 or later
- Dependencies listed in `requirements.txt`

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration files

The application relies on the following files for credentials and tokens:

- `credentials.json`: contains Fitbit `client_id` and `client_secret` pairs.
- `client_ids.txt`: Fitbit device identifiers to process.
- `fitbit_tokens.json`: OAuth tokens generated during Fitbit authorization.
- `thingsboard_tokens.json`: ThingsBoard access tokens and usernames.

## Usage

### Interactive mode

Launch a graphical interface to authorize a Fitbit account and export its data:

```bash
python single_fit.py
```

The retrieved data is saved in `fitbit_data/` and converted to Excel automatically.

### Automated mode

Process multiple accounts and send the resulting metrics to ThingsBoard:

```bash
python auto_fit.py --start-date 2024-01-01 --end-date 2024-01-07
```

Key options:

- `--start-date` and `--end-date`: date range in `YYYY-MM-DD` format.
- `--window`: aggregation window size in seconds (default `60`).
- `--monitor`: enable continuous monitoring for the current day.
- `--interval`: wait time in seconds between batches or monitoring checks (default `3000`).

