"""
config.py — environment/credentials and the pooled DB engine for the data API.

Credentials live in .env (gitignored, see .env.example for the expected
keys) — this is the concrete fix for "DB credentials hardcoded in source"
that direct access from inside the Streamlit process had no way to avoid.
"""
import os
from functools import lru_cache
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

load_dotenv()


def _dwh_conn_str() -> str:
    server = os.environ["DB_SERVER"]
    database = os.environ["DB_DWH_DATABASE"]
    uid = os.environ["DB_DWH_UID"]
    pwd = os.environ["DB_DWH_PWD"]
    return (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};DATABASE={database};UID={uid};PWD={pwd};"
        "TrustServerCertificate=yes;"
    )


@lru_cache
def get_dwh_engine() -> Engine:
    """Pooled, reused connections — the actual fix for opening a fresh
    pyodbc connection on every call, which the direct-DB-access path did.

    The mssql+pyodbc dialect always calls unquote_plus() exactly once on the
    "odbc_connect" query value, so it must be quote_plus'd exactly once here
    — building the URL with an f-string instead double-decoded it (make_url
    parsing the connection URL string itself unquotes once, then the dialect
    unquotes again), silently turning a literal "+" in the password into a
    space. URL.create() bypasses that first parse; quote_plus supplies the
    one encoding the dialect expects to reverse.
    """
    url = URL.create("mssql+pyodbc", query={"odbc_connect": quote_plus(_dwh_conn_str())})
    return create_engine(
        url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=1800,
    )
