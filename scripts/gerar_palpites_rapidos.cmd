@echo off
setlocal
cd /d "%~dp0.."

python predict_upcoming.py ^
  --markets all ^
  --leagues E0 E1 SC0 D1 D2 SP1 SP2 I1 I2 F1 F2 N1 P1 B1 G1 T1 ^
  --seasons 0506 0607 0708 0809 0910 1011 1112 1213 1314 1415 1516 1617 1718 1819 1920 2021 2122 2223 2324 2425 2526 ^
  --feature-profile extended ^
  --days-ahead 7 ^
  --xgb-tuning-trials 4 ^
  --preferred-bookmaker Betano ^
  --force-refresh-fixtures ^
  --use-saved-models ^
  %*

endlocal
