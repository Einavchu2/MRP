import os
import pandas as pd

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "batch_history.csv")
BATCH_HISTORY_FILE = HISTORY_FILE

DEFAULT_COLUMNS = [
    "batch_id",
    "product",
    "batch_qty",
    "production_date",
    "step",
    "status",
    "from_location",
    "to_location",
    "notes",
    "user",
    "timestamp",
]


def _ensure_history_file() -> None:
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=DEFAULT_COLUMNS).to_csv(HISTORY_FILE, index=False)


def load_batch_history() -> pd.DataFrame:
    """Load the batch history CSV and return a normalized DataFrame."""
    _ensure_history_file()
    df = pd.read_csv(HISTORY_FILE, dtype=str)
    for col in DEFAULT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[DEFAULT_COLUMNS]
    df["batch_qty"] = pd.to_numeric(df["batch_qty"].fillna(0), errors="coerce").fillna(0)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["production_date"] = pd.to_datetime(df["production_date"], errors="coerce")
    return df.sort_values(["batch_id", "timestamp"])


def save_batch_history(df: pd.DataFrame) -> None:
    """Persist the batch history DataFrame to the history CSV."""
    _ensure_history_file()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["production_date"] = pd.to_datetime(df["production_date"], errors="coerce")
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["production_date"] = df["production_date"].dt.strftime("%Y-%m-%d")
    df.to_csv(HISTORY_FILE, index=False)


def add_batch_event(
    batch_id: str,
    product: str,
    step: str,
    status: str,
    from_location: str = "",
    to_location: str = "",
    notes: str = "",
    batch_qty: float = 0.0,
    production_date: str = "",
    user: str = "system",
    timestamp: str | None = None,
) -> pd.DataFrame:
    """Append a new event row for a batch and return the updated history."""
    history = load_batch_history()
    if not timestamp:
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "batch_id": str(batch_id).strip(),
        "product": str(product).strip(),
        "batch_qty": float(batch_qty) if batch_qty is not None else 0.0,
        "production_date": str(production_date).strip(),
        "step": str(step).strip(),
        "status": str(status).strip(),
        "from_location": str(from_location).strip(),
        "to_location": str(to_location).strip(),
        "notes": str(notes).strip(),
        "user": str(user).strip(),
        "timestamp": timestamp,
    }
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    save_batch_history(history)
    return load_batch_history()


def get_batch_trace(batch_id: str, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the full trace of events for one batch, sorted by timestamp."""
    if history is None:
        history = load_batch_history()
    batch_id = str(batch_id).strip()
    trace = history[history["batch_id"] == batch_id].copy()
    trace = trace.sort_values("timestamp")
    return trace


def summarize_batches(history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the latest status and last update for each batch."""
    if history is None:
        history = load_batch_history()
    if history.empty:
        return pd.DataFrame(columns=["batch_id", "product", "last_status", "last_step", "last_timestamp", "batch_qty"])
    latest = history.sort_values("timestamp").groupby("batch_id", as_index=False).last()
    return latest.rename(columns={
        "status": "last_status",
        "step": "last_step",
        "timestamp": "last_timestamp",
    })[["batch_id", "product", "batch_qty", "production_date", "last_step", "last_status", "last_timestamp"]]
