"""
healthcheck.py — the Detection Layer for the MRP daily monitor.

Read-only. Runs every app-level check (HTTP health endpoints, SQL Server DWH
connectivity, SQLite integrity, disk/memory/CPU, log error scanning) and
prints ONE JSON object to stdout — nothing else touches stdout, so
Invoke-DailyHealthCheck.ps1 can parse it directly.

Windows-native checks (services, scheduled tasks, IIS, process/port state,
and every remediation action) deliberately live in the PowerShell script
instead — this file only reports, it never restarts anything. Keeping "what
is true" (Python) separate from "what to do about it" (PowerShell) is what
makes each half independently testable: run this file by hand any time to
see exactly what the monitor sees, with no side effects.

Usage:
    python healthcheck.py                  # run every check
    python healthcheck.py --only db,disk   # run only the named checks
                                            # (used by the Validation Layer
                                            # to re-check just what it fixed)
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
from urllib.parse import quote_plus

import psutil
import requests

# ── config ──────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.json"), "r", encoding="utf-8") as f:
    CFG = json.load(f)

APP_ROOT = CFG["app_root"]
THRESH   = CFG["thresholds"]
STATE_DIR = os.path.join(HERE, "state")
os.makedirs(STATE_DIR, exist_ok=True)
CURSOR_FILE = os.path.join(STATE_DIR, "log_cursors.json")


def _ok(check, detail="", **data):
    return {"check": check, "status": "OK", "detail": detail, "data": data}


def _warn(check, detail="", **data):
    return {"check": check, "status": "WARN", "detail": detail, "data": data}


def _crit(check, detail="", **data):
    return {"check": check, "status": "CRITICAL", "detail": detail, "data": data}


# ── HTTP health endpoints ──────────────────────────────────────────

def check_http(name, url):
    t0 = time.time()
    try:
        r = requests.get(url, timeout=THRESH["http_timeout_sec"])
        ms = round((time.time() - t0) * 1000)
        if r.status_code == 200:
            return _ok(name, f"HTTP 200 in {ms}ms", url=url, ms=ms)
        return _crit(name, f"HTTP {r.status_code} from {url}", url=url, status_code=r.status_code)
    except requests.exceptions.Timeout:
        return _crit(name, f"Timed out after {THRESH['http_timeout_sec']}s: {url}", url=url)
    except requests.exceptions.RequestException as e:
        return _crit(name, f"Unreachable: {url} — {e.__class__.__name__}: {e}", url=url)


# ── SQL Server DWH connectivity ────────────────────────────────────

def check_db():
    """Mirrors api/config.py's own connection method on purpose (same driver,
    same odbc_connect quoting) — if that file's approach ever changes, this
    should change with it. Kept as a standalone copy rather than importing
    api.config so this script has no dependency on the app's package layout
    and keeps working even if the app is mid-restart."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(APP_ROOT, ".env"))
        server = os.environ["DB_SERVER"]
        database = os.environ["DB_DWH_DATABASE"]
        uid = os.environ["DB_DWH_UID"]
        pwd = os.environ["DB_DWH_PWD"]
    except KeyError as e:
        return _crit("db_connectivity", f"Missing env var {e} — check {APP_ROOT}\\.env")

    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};DATABASE={database};UID={uid};PWD={pwd};"
        "TrustServerCertificate=yes;Connection Timeout=8;"
    )
    t0 = time.time()
    try:
        import pyodbc
        conn = pyodbc.connect(conn_str, timeout=8)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        ms = round((time.time() - t0) * 1000)
        return _ok("db_connectivity", f"SELECT 1 OK in {ms}ms", server=server, database=database, ms=ms)
    except Exception as e:
        return _crit("db_connectivity", f"{e.__class__.__name__}: {e}", server=server, database=database)


# ── SQLite tracking DB ─────────────────────────────────────────────

def check_sqlite():
    path = os.path.join(APP_ROOT, CFG["sqlite_db"])
    if not os.path.exists(path):
        return _crit("sqlite_tracking_db", f"File not found: {path}")
    try:
        conn = sqlite3.connect(path, timeout=8)
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM sku_tracking").fetchone()[0]
        conn.close()
        if result != "ok":
            return _crit("sqlite_tracking_db", f"PRAGMA integrity_check failed: {result}")
        return _ok("sqlite_tracking_db", f"integrity ok, {count} rows", rows=count)
    except Exception as e:
        return _crit("sqlite_tracking_db", f"{e.__class__.__name__}: {e}")


# ── system resources ────────────────────────────────────────────────

def check_disk():
    path = os.path.splitdrive(APP_ROOT)[0] + "\\"
    u = psutil.disk_usage(path)
    free_pct = 100 - u.percent
    detail = f"{path} — {free_pct:.1f}% free ({u.free // (1024**3)} GB of {u.total // (1024**3)} GB)"
    data = dict(drive=path, free_pct=round(free_pct, 1), free_gb=round(u.free / 1024**3, 1))
    if free_pct <= THRESH["disk_free_pct_critical"]:
        return _crit("disk", detail, **data)
    if free_pct <= THRESH["disk_free_pct_warn"]:
        return _warn("disk", detail, **data)
    return _ok("disk", detail, **data)


def check_memory():
    m = psutil.virtual_memory()
    detail = f"{m.percent:.1f}% used ({(m.total - m.available) // (1024**2)} MB of {m.total // (1024**2)} MB)"
    data = dict(used_pct=m.percent)
    if m.percent >= THRESH["mem_pct_critical"]:
        return _crit("memory", detail, **data)
    if m.percent >= THRESH["mem_pct_warn"]:
        return _warn("memory", detail, **data)
    return _ok("memory", detail, **data)


def check_cpu():
    # psutil needs two samples per process to compute a %, so prime everything,
    # take the one blocking system-wide sample (1s — deliberate, once per run,
    # not in a hot loop), then read back the per-process deltas over that window.
    procs = list(psutil.process_iter(["pid", "name"]))
    for p in procs:
        try:
            p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    pct = psutil.cpu_percent(interval=1.0)
    top = []
    for p in procs:
        try:
            # System Idle Process (pid 0) reports a meaningless/inflated number
            # by design (it's the "no work happening" counter, not real load).
            if p.pid == 0:
                continue
            cpu = min(p.cpu_percent(interval=None), 100.0)
            if cpu > 1.0:
                top.append({"pid": p.pid, "name": p.info["name"], "cpu_pct": round(cpu, 1)})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    top = sorted(top, key=lambda x: -x["cpu_pct"])[:5]
    detail = f"{pct:.1f}% system-wide"
    data = dict(cpu_pct=pct, top_processes=top)
    if pct >= THRESH["cpu_pct_critical"]:
        return _crit("cpu", detail, **data)
    if pct >= THRESH["cpu_pct_warn"]:
        return _warn("cpu", detail, **data)
    return _ok("cpu", detail, **data)


# ── log error scanning ──────────────────────────────────────────────
# Cursor-tracked: each run only looks at bytes appended since the last run,
# so a historical error found once doesn't re-fire forever. A log file that
# shrinks (rotated) resets its cursor to 0 automatically.

def _load_cursors():
    if os.path.exists(CURSOR_FILE):
        with open(CURSOR_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cursors(cursors):
    with open(CURSOR_FILE, "w", encoding="utf-8") as f:
        json.dump(cursors, f, indent=2)


def check_log(name, relative_path):
    path = os.path.join(APP_ROOT, relative_path)
    cursors = _load_cursors()
    if not os.path.exists(path):
        return _ok(name, f"{relative_path} does not exist yet — nothing to scan")

    size = os.path.getsize(path)
    last = cursors.get(relative_path, size)  # first-ever run: don't replay history
    if size < last:
        last = 0  # file rotated/truncated

    new_lines = []
    if size > last:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(last)
            new_lines = f.readlines()
    cursors[relative_path] = size
    _save_cursors(cursors)

    critical_markers = ("Traceback", "CRITICAL", "WinError")
    warn_markers = ("ERROR", "Exception")
    crit_hits = [l.strip() for l in new_lines if any(m in l for m in critical_markers)]
    warn_hits = [l.strip() for l in new_lines if l.strip() not in crit_hits and any(m in l for m in warn_markers)]

    if crit_hits:
        return _crit(name, f"{len(crit_hits)} critical line(s) since last check",
                     sample=crit_hits[-5:], new_bytes=size - last)
    if warn_hits:
        return _warn(name, f"{len(warn_hits)} warning line(s) since last check",
                     sample=warn_hits[-5:], new_bytes=size - last)
    return _ok(name, f"no new errors ({size - last} new bytes scanned)")


# ── registry ─────────────────────────────────────────────────────────

def all_checks():
    eps = CFG["endpoints"]
    # Keys here MUST equal the "check" name each function emits (_ok/_warn/_crit's
    # first arg) — Invoke-DailyHealthCheck.ps1's validation layer re-invokes a
    # single failed check by this same name (`--only <name>`), so a mismatch
    # here would make re-validation silently no-op instead of re-checking.
    checks = {
        "service_streamlit": lambda: check_http("service_streamlit", eps["streamlit_health"]),
        "service_api":       lambda: check_http("service_api", eps["api_health"]),
        "service_api_proxy": lambda: check_http("service_api_proxy", eps["api_proxy_health"]),
        "db_connectivity":   check_db,
        "sqlite_tracking_db": check_sqlite,
        "disk":              check_disk,
        "memory":            check_memory,
        "cpu":               check_cpu,
    }
    for name, relpath in CFG.get("log_files", {}).items():
        checks[f"log_{name}"] = (lambda rp=relpath, n=name: check_log(f"log_{n}", rp))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated check names to run (default: all)")
    args = ap.parse_args()

    registry = all_checks()
    names = [n.strip() for n in args.only.split(",")] if args.only else list(registry.keys())

    results = []
    for name in names:
        fn = registry.get(name)
        if fn is None:
            results.append(_warn(name, f"unknown check name (not in registry)"))
            continue
        try:
            results.append(fn())
        except Exception as e:
            results.append(_crit(name, f"check itself raised {e.__class__.__name__}: {e}",
                                  traceback=traceback.format_exc()[-1500:]))

    print(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
