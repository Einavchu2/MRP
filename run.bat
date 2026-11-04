@echo off
REM Windows startup script
REM Set your Databricks token before running
set DATABRICKS_TOKEN=your_token_here
REM Start run_api.bat first (in its own window) — the Simulation page
REM calls it over HTTP and won't load data without it running.
streamlit run app.py
