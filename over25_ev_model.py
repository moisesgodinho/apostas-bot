"""Ponto de entrada do pipeline Apostasbot.

Mantido para compatibilidade com o comando:
    python over25_ev_model.py
"""

from __future__ import annotations

from cli import parse_args
from pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(parse_args())
