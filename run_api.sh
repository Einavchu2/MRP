#!/bin/bash
# Starts the MRP data API — must be running before the Streamlit app,
# since src/db.py calls it over HTTP. Bound to 127.0.0.1: only this
# machine can reach it, nothing outside needs to.
uvicorn api.main:app --host 127.0.0.1 --port 8502
