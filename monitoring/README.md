# MRP Daily Health Check & Self-Healing Monitor

Runs once a day (Task Scheduler), checks the whole MRP stack, tries to fix
what it safely can, and emails you when it can't. Full design rationale —
architecture, decision/escalation logic, risk assessment — is in the
published design doc; this file is the "how do I actually run/operate it"
quick reference.

## What it watches

| Layer | Checked by | What |
|---|---|---|
| App HTTP health | `healthcheck.py` | Streamlit `:8501`, API `:8502`, IIS proxy `:8503` |
| Database | `healthcheck.py` | SQL Server DWH (`ORACLE_DWH`) — `SELECT 1` |
| Tracking data | `healthcheck.py` | `outputs\sku_tracking.db` integrity check |
| System resources | `healthcheck.py` | Disk free %, memory %, CPU % + top processes |
| Logs | `healthcheck.py` | `streamlit.err.log`, `api.log` — new `ERROR`/`Traceback`/`CRITICAL` lines since last run |
| Process/port state | `Invoke-DailyHealthCheck.ps1` | Duplicate binds, stuck sockets — the exact 2026-09-10 outage pattern |
| Scheduled Tasks | `Invoke-DailyHealthCheck.ps1` | "Data API" / "Streamlit App" boot-task status (informational) |
| IIS | `Invoke-DailyHealthCheck.ps1` | `MrpApiProxy` site + app pool state |

**Not covered** — there's no evidence in this codebase of a batch ETL /
interface layer beyond the two web services and the SQL Server DWH itself.
If real scheduled jobs or interfaces exist elsewhere (e.g. inside the DWH,
or a separate integration platform), they aren't visible to this monitor —
add a check for them in `healthcheck.py`'s registry (see the file's
docstring) rather than assuming they're covered.

## First-time setup

```powershell
# 1. Fill in alerting credentials (never committed — see .gitignore)
copy monitoring\.env.example monitoring\.env
notepad monitoring\.env

# 2. Test the whole pipeline with zero side effects
powershell -File monitoring\Invoke-DailyHealthCheck.ps1 -DryRun
# review monitoring\reports\latest.html

# 3. (Optional but recommended) during a maintenance window, test a REAL
#    remediation once — e.g. stop the Streamlit process by hand and run
#    without -DryRun, confirm it comes back and you get no alert.

# 4. Register the daily Scheduled Task (prompts for the account password —
#    Scheduled Tasks with restart rights need a stored credential, this is
#    the only place one is entered, and it's never written to a file)
powershell -File monitoring\Register-DailyHealthCheck-Task.ps1
```

## Day to day

- **Reports:** `monitoring\reports\YYYY-MM-DD.html` (one per day) and
  `monitoring\reports\latest.html` — status of every check, what was
  auto-fixed, what wasn't.
- **Audit trail:** `monitoring\state\audit.jsonl` — one line per event
  (every check result, every remediation attempt, every validation,
  every alert), append-only, never rewritten. Never manually delete
  entries from it — it's the incident record.
- **Alerts:** emailed per `monitoring\.env`. Also always written to
  `monitoring\alerts\ALERT_<run-id>.html` and the Windows Application
  Event Log (source `MRP Daily Monitor`) as a second channel, in case
  mail delivery itself is the thing that's broken.

## Kill switch

Drop an empty file named `DISABLE_AUTOHEAL` in this folder:

```powershell
New-Item monitoring\DISABLE_AUTOHEAL -ItemType File
```

Detection, reporting, and alerting keep running as normal — only the
automatic restart/repair actions stop. Delete the file to re-enable.
Use this the moment autonomous recovery seems to be doing the wrong thing;
diagnose by hand with autoheal off rather than fighting it live.

## Circuit breaker

Each check gets at most **3 remediation attempts per calendar day**, with a
**10-minute cooldown** between attempts (`config.json` →
`circuit_breaker`). A check that keeps failing stops triggering restarts
after that and goes straight to escalation instead — this is what prevents
an unattended restart loop. If you see the same check hitting its daily
limit repeatedly, that is itself the signal something needs a real fix, not
another restart.

## Known limitations (read before trusting this unattended for weeks)

- **Disk space:** reported and escalated, never auto-remediated — this
  monitor will not delete files on its own. Freeing disk space
  automatically risks deleting something that mattered more than the
  disk pressure did.
- **Unrecognized high-CPU/memory processes:** reported with a
  recommendation, never killed — only processes this monitor already
  manages (Streamlit, the API) get restarted automatically.
- **SQLite corruption:** flagged, never auto-repaired — attempting to
  "fix" a corrupt tracking database automatically risks losing real data
  instead of losing an hour to a manual look.
- **Database network-path failures:** the monitor retries and tests
  reachability, but if the SQL Server itself is down or unreachable, no
  action here can fix that — it escalates with the network-test result
  attached so whoever responds doesn't start from zero.
- Every one of these is a design choice, not an oversight — see the "Risk
  assessment" section of the design doc for the full reasoning.
