# MRP Simulation / SKU Procurement Tracking — Project Handoff

Server: `FEIL1338VS` (Windows Server 2022), path `c:\inetpub\wwwroot\mrp_simulation`.
Accessed by the team remotely, from their own PCs over the network — not on the
server console. This fact drives the one open architectural blocker below.

## 1. Architecture

```
Users' PCs (browser)
        │  HTTP :8501
        ▼
Streamlit app  (app.py, run.bat/run.sh)
        │  HTTP, localhost only
        ▼
Data API  (FastAPI + uvicorn, api/main.py, run_api.bat/run_api.sh)
   bound to 127.0.0.1:8502 — deliberately not reachable from the network, no auth
        │
        ├── SQL Server (BOM / DWH / substitutes) — api/queries.py, pooled SQLAlchemy engine
        └── SQLite  outputs/sku_tracking.db — api/sku_db.py (SKU Procurement Tracking CRUD)
```

- **Streamlit (`app.py`)** is both UI and "backend glue." It has two modes: the
  MRP simulation, and SKU Procurement Tracking. `st.cache_data` caches (TTL 300s)
  are process-wide, shared by every user's session, and reset on process restart.
- **The API service** was built this session to stop the app from making 142
  sequential SQL Server queries per BOM load (now 1) and to get SQL credentials
  out of source into `.env`. It owns *all* database access — Streamlit never
  talks to SQL Server or SQLite directly anymore.
  - `src/db.py` — HTTP client for `/bom`, `/dwh-data`, `/substitutes`.
  - `src/sku_tracking.py` — HTTP client for `/sku-tracking` CRUD.
  - Both point at `MRP_API_URL` env var, default `http://127.0.0.1:8502`.
- **SKU Procurement Tracking storage**: migrated this session from a
  read-whole-file/rewrite-whole-file CSV (`outputs/sku_tracking.csv`, now an
  unused backup) to a real SQLite database (`outputs/sku_tracking.db`, WAL mode,
  15-column `sku_tracking` table, `id` primary key) owned exclusively by
  `api/sku_db.py`. 188 rows migrated and verified byte-for-byte, including
  multi-line notes.
- **`.env`** (gitignored) holds real SQL Server connection details; `.env.example`
  is the tracked placeholder template.
- **`web.config`** at the project root only sets `<webSocket enabled="true" />` —
  it does not currently make IIS front this app in any way.
- **IIS on this box**: running, has URL Rewrite installed, does **not** have
  Application Request Routing (ARR) — so it cannot reverse-proxy to another port
  yet. It does not front `mrp_simulation` today; the app is reached by hitting
  Streamlit's own port 8501 directly. There is an unrelated pre-existing site
  "StreamlitApp" (port 81, `C:\inetpub\wwwroot\tracking_system`) — a dormant,
  different Flask/WSGI project. Not reused, not relevant here.

## 2. Completed work

- **BOM performance fix**: eliminated N+1 SQL pattern (142 → 1 query), removed
  unconditional `.to_excel()` calls that ran on every page load
  (`src/data_prep.py`).
- **API layer built** (`api/` — `main.py`, `cache.py`, `config.py`, `queries.py`):
  owns SQL Server access via a pooled SQLAlchemy engine with a TTL cache; bound
  to `127.0.0.1:8502` for security.
- **SKU Tracking UX pass**: filter bar, quick-add form (searchable/creatable
  dropdowns via `accept_new_options=True`), a per-row "✏️ Edit this row" panel
  using real `st.text_area` fields so Enter/Alt+Enter genuinely produce line
  breaks (confirmed impossible inside `st.data_editor` grid cells — a hard
  Streamlit/glide-data-grid platform limitation, not a bug in our code).
- **Grid removed entirely**: per explicit user instruction, once the row-editor
  panel covered editing (including a delete button, added first so nothing
  regressed), the old `st.data_editor` "Edit existing rows" grid was deleted.
- **Full CRUD + SQLite migration**: `api/sku_db.py` (storage),
  `api/migrate_sku_tracking_csv.py` (one-time, idempotent migration, already
  run), new `/sku-tracking` REST endpoints in `api/main.py`, `src/sku_tracking.py`
  rewritten as a thin HTTP client. Verified live: create, read, update, delete
  all round-trip correctly, including Hebrew text and multi-line notes.

## 3. Bugs found and fixed this session

| Bug | Root cause | Fix |
|---|---|---|
| SQL Server password corrupted (`+` → space) | SQLAlchemy's `make_url()` parser and the `mssql+pyodbc` dialect's `create_connect_args` each call `unquote_plus()` once — building the URL as an f-string double-decoded it | Build with `URL.create("mssql+pyodbc", query={"odbc_connect": quote_plus(raw)})`, quoting exactly once |
| SQLite connections never closed | `with sqlite3.connect(...) as conn:` only manages commit/rollback, not closing | Rewrote every function in `api/sku_db.py` to `conn = _connect(); try: ... finally: conn.close()` |
| Hebrew text mangled to `?????` during manual CRUD testing | Artifact of passing Hebrew via a shell `-d` argument to `curl.exe` in this git-bash environment, not an API bug | Re-verified via Python `requests` (the real code path) — round-tripped correctly. Cleaned up 2 leftover mangled test rows afterward |

**No known open bugs.** The Enter/Alt+Enter-inside-the-grid issue is closed —
resolved by removing the grid, not by a workaround, since no real fix exists for
that specific Streamlit component.

## 4. Key implementation decisions (with reasoning)

- **SQLite over requesting new SQL Server write credentials**: user's explicit
  choice, avoids an IT ticket/wait, and SKU Tracking's volume/concurrency needs
  don't warrant SQL Server.
- **API stays bound to `127.0.0.1`, no auth added**: kept the API's attack
  surface at zero rather than exposing it on the network and having to build
  authentication from scratch.
- **IIS + ARR reverse proxy chosen (over exposing the API + building auth, or
  stopping)**: user's explicit choice, made after being asked directly and
  confirming remote/network access is real. This is the mechanism that will let
  a truly unified inline-editable table's JavaScript call the API from users'
  own browsers.
- **New, separate, dedicated IIS site for the proxy** (not modifying Streamlit's
  own serving): deliberately minimizes blast radius — Streamlit's WebSocket
  traffic on port 8501 is the single riskiest thing to break, so it stays
  completely untouched; only a new site on a new port (e.g. 8503) is added.
- **Grid removed rather than left partially working**: user's explicit
  correction — a broken-but-present editing affordance next to a working one is
  worse than no affordance at all.

## 5. Pending tasks — unified inline-editable SKU Tracking table

**Status: approved, paused mid-execution, not resumed.** Full plan saved at
`C:\Users\EINA-ADM\.claude\plans\enumerated-bouncing-neumann.md`. Nothing below
has touched the server yet.

1. **Install Application Request Routing 3.0 x64** on this server (confirmed
   live official download:
   `https://download.microsoft.com/download/e/9/8/e9849d6a-020e-47e4-9fd0-a023e99b54eb/requestRouter_amd64.msi`,
   v3.0.5311). Enable IIS's proxy feature (`system.webServer/proxy` →
   `enabled=True`).
2. **New IIS site** (e.g. `C:\inetpub\wwwroot\mrp_api_proxy`, new port such as
   8503) with one URL Rewrite rule proxying everything to
   `http://127.0.0.1:8502/{R:1}`. Streamlit keeps running untouched on 8501.
3. **Add CORS** to `api/main.py` (`CORSMiddleware`, `allow_origin_regex` scoped
   to "any host on port 8501" — not a bare `*`).
4. **Rebuild the SKU Tracking table** as one real HTML/JS editable table,
   replacing `render_sku_tracking_html()` and the current "🔎 Jump to a
   row"/"✏️ Edit this row" panel: per-row ✏️ swaps cells to real `<textarea>`
   (genuine Enter/Alt+Enter), `<input type="date">`, `<select>`, plain
   `<input>`; 💾/🗑️ call the proxied API via `fetch()` (PATCH/DELETE). The
   "➕ Add new tracking item" quick-add form is explicitly out of scope and
   stays as-is (it has searchable/creatable dropdowns a hand-rolled `<select>`
   can't replicate).
5. **Remove now-dead code**: old HTML renderer, jump-to-row selector, row-editor
   panel and its save/delete handlers in `app.py`.

**Disclosed trade-offs** (already agreed with the user): inline edits use plain
`<select>` for Planner/Supplier/etc. (existing values only, no
searchable-or-create-new); KPI counts above the table won't retally after a
JS-only edit without a manual refresh action.

**Verification plan** (in order): proxy reachable from a genuinely different
machine on the network → proxied CRUD matches direct-API behavior → Streamlit
itself unaffected → real multi-line Enter/Alt+Enter confirmed in-browser →
delete and API-down states show a clear inline error.

**Why paused**: `/compact` arrived twice while installation of ARR (a
system-modifying, production-server step) was about to begin. Judgment call was
to stop and wait for explicit re-confirmation rather than proceed unattended.
**Resuming requires the user's explicit go-ahead** before step 1 (downloading/
installing ARR) begins.
