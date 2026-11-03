"""
cache.py — minimal TTL cache for the data API.

Mirrors the ttl=300 convention already used throughout the Streamlit app via
st.cache_data. This process is separate from Streamlit, so its cache has its
own lifecycle — it survives a Streamlit restart, which is the whole point.
"""
import threading
import time


def ttl_cache(ttl_seconds: float):
    def decorator(fn):
        lock = threading.Lock()
        store: dict = {}

        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            with lock:
                cached = store.get(key)
                if cached is not None and now - cached[1] < ttl_seconds:
                    return cached[0]
            value = fn(*args, **kwargs)
            with lock:
                store[key] = (value, now)
            return value

        return wrapper

    return decorator
