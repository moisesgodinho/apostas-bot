@echo off
setlocal
cd /d "%~dp0.."

python over25_ev_model.py ^
  --markets over25 ^
  --leagues E0 SP1 D1 I1 F1 ^
  --seasons 1920 2021 2122 2223 2324 2425 2526 ^
  --feature-profile extended ^
  --skip-model-comparison ^
  --skip-realistic-backtest ^
  %*

endlocal
