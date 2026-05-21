@echo off
setlocal
cd /d "%~dp0.."

streamlit run dashboard.py %*

endlocal
