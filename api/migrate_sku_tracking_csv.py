"""
migrate_sku_tracking_csv.py — one-time migration of outputs/sku_tracking.csv
into the SQLite table sku_db owns. Run once: `python -m api.migrate_sku_tracking_csv`

Idempotent (INSERT OR IGNORE by id) — safe to re-run by accident. Does not
touch or delete the source CSV.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sku_tracking import load_tracking
from api import sku_db


def main():
    df = load_tracking()
    conn = sku_db._connect()
    try:
        n_before = conn.execute("SELECT COUNT(*) FROM sku_tracking").fetchone()[0]
        with conn:
            for _, row in df.iterrows():
                record = {c: str(row.get(c, "") or "") for c in sku_db.COLUMNS}
                placeholders = ", ".join(f":{c}" for c in sku_db.COLUMNS)
                conn.execute(
                    f'INSERT OR IGNORE INTO sku_tracking ({", ".join(sku_db.COLUMNS)}) VALUES ({placeholders})',
                    record,
                )
        n_after = conn.execute("SELECT COUNT(*) FROM sku_tracking").fetchone()[0]
    finally:
        conn.close()

    print(f"CSV rows: {len(df)}")
    print(f"SQLite rows before: {n_before}, after: {n_after}")


if __name__ == "__main__":
    main()
