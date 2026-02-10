from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import statistics
def _is_valid_hr(bpm: Optional[float]) -> bool:
    if bpm is None:
        return False
    try:
        b = float(bpm)
    except (TypeError, ValueError):
        return False
    return 30.0 <= b <= 220.0
def hr_to_ibi_ms(hr_series: Sequence[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for bpm in hr_series:
        if not _is_valid_hr(bpm):
            out.append(None)
        else:
            out.append(60000.0 / float(bpm))
    return out
def _clean_ibi(ibi_ms: Sequence[Optional[float]], lo: float = 300.0, hi: float = 2000.0) -> List[float]:
    return [x for x in ibi_ms if x is not None and lo <= x <= hi]
def rmssd_proxy(ibi_ms: Sequence[Optional[float]]) -> Optional[float]:
    clean = _clean_ibi(ibi_ms)
    if len(clean) < 3:
        return None
    diffs: List[float] = []
    prev = clean[0]
    for x in clean[1:]:
        diffs.append(x - prev)
        prev = x
    if not diffs:
        return None
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))
def sdnn_proxy(ibi_ms: Sequence[Optional[float]]) -> Optional[float]:
    clean = _clean_ibi(ibi_ms)
    if len(clean) < 3:
        return None
    try:
        return statistics.pstdev(clean)
    except statistics.StatisticsError:
        return None
def pnn50_proxy(ibi_ms: Sequence[Optional[float]], threshold_ms: float = 50.0) -> Optional[float]:
    clean = _clean_ibi(ibi_ms)
    if len(clean) < 3:
        return None
    prev = clean[0]
    cnt = 0
    total = 0
    for x in clean[1:]:
        total += 1
        if abs(x - prev) > threshold_ms:
            cnt += 1
        prev = x
    if total == 0:
        return None
    return cnt / total
def parse_times_and_hr_from_data(data: Dict) -> Tuple[List[datetime], List[Optional[float]]]:
    date_str = data.get("Fecha") or data.get("date") or "1970-01-01"
    ds = data.get("Ritmo_Cardiaco", {}).get("dataset", [])
    base = datetime.fromisoformat(str(date_str))
    times: List[datetime] = []
    hr: List[Optional[float]] = []
    for row in ds:
        t = row.get("time")
        v = row.get("value")
        try:
            hh, mm, ss = [int(x) for x in str(t).split(":")]
            times.append(base.replace(hour=hh, minute=mm, second=ss, microsecond=0))
            hr.append(float(v) if v is not None else None)
        except Exception:
            continue
    return times, hr
@dataclass
class HRVProxyAgg:
    rmssd_proxy_ms: Optional[float]
    sdnn_proxy_ms: Optional[float]
    pnn50_proxy: Optional[float]
    valid_samples: int
def aggregate_proxies_from_hr(hr_values: Sequence[Optional[float]]) -> HRVProxyAgg:
    ibi = hr_to_ibi_ms(hr_values)
    return HRVProxyAgg(
        rmssd_proxy_ms=rmssd_proxy(ibi),
        sdnn_proxy_ms=sdnn_proxy(ibi),
        pnn50_proxy=pnn50_proxy(ibi),
        valid_samples=sum(1 for v in hr_values if _is_valid_hr(v)),
    )
def windowed_proxies(
    times: Sequence[datetime],
    hr_values: Sequence[Optional[float]],
    window_seconds: int = 300,
    step_seconds: int = 60,
    min_samples: int = 60,
) -> List[Dict]:
    if len(times) != len(hr_values):
        raise ValueError("times y hr_values deben tener la misma longitud")
    n = len(times)
    if n == 0:
        return []
    start = times[0]
    end = times[-1]
    win = timedelta(seconds=window_seconds)
    step = timedelta(seconds=step_seconds)
    out: List[Dict] = []
    t = start
    i0 = 0
    while t <= end:
        t1 = t + win
        while i0 < n and times[i0] < t:
            i0 += 1
        i1 = i0
        while i1 < n and times[i1] < t1:
            i1 += 1
        if i1 - i0 > 1:
            seg = hr_values[i0:i1]
            if sum(1 for v in seg if _is_valid_hr(v)) >= min_samples:
                ibi = hr_to_ibi_ms(seg)
                out.append({
                    "window_start": t.isoformat(),
                    "window_end": t1.isoformat(),
                    "rmssd_proxy_ms": rmssd_proxy(ibi),
                    "sdnn_proxy_ms": sdnn_proxy(ibi),
                    "pnn50_proxy": pnn50_proxy(ibi),
                    "n": sum(1 for v in seg if _is_valid_hr(v)),
                })
        t += step
    return out
def attach_proxies_to_data(data: Dict, agg: HRVProxyAgg, series: List[Dict]) -> Dict:
    out = dict(data)
    out["HRV_Proxy"] = asdict(agg)
    out["HRV_Proxy_Series"] = series
    return out