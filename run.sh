#!/bin/bash
# Linux/Mac startup script
export DATABRICKS_TOKEN="your_token_here"
# Start run_api.sh first (separate process) — the Simulation page calls
# it over HTTP and won't load data without it running.
streamlit run app.py
