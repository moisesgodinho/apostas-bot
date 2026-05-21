@echo off
setlocal
cd /d "%~dp0.."

echo [1/2] Verificando dashboard...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$running = Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | Where-Object { $_.CommandLine -like '*streamlit run dashboard.py*' }; if (-not $running) { Start-Process cmd.exe -ArgumentList '/k','cd /d ""%cd%"" && streamlit run dashboard.py' -WindowStyle Normal }"

echo [2/2] Rodando pipeline completo do Over 2.5...
python over25_ev_model.py ^
  --markets over25 ^
  --leagues E0 SP1 D1 I1 F1 ^
  --seasons 1920 2021 2122 2223 2324 2425 2526 ^
  --feature-profile extended ^
  %*

endlocal
