@echo off
setlocal
cd /d "%~dp0.."

python over25_ev_model.py ^
  --markets over25 ^
  --leagues E0 SP1 ^
  --seasons 2223 2324 2425 2526 ^
  --feature-profile extended ^
  --edge 0.02 ^
  --min-model-prob 0.52 ^
  --max-over-odd 2.00 ^
  --skip-model-comparison ^
  --skip-realistic-backtest ^
  %*

endlocal
