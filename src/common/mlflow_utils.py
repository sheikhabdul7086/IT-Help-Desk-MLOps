"""Shared MLflow tracking-URI setup: defaults to a local SQLite store, but
switches to a remote server (e.g. DagsHub) when MLFLOW_TRACKING_URI is set
in project/.env, with no other code changes needed.
"""
import os
import sys
from pathlib import Path

import mlflow

# MLflow prints an emoji in the "View run at: ..." message when logging to a
# remote server; Windows' default console codepage (cp1252) can't encode it
# and crashes mid-run. Force UTF-8 stdout/stderr so remote runs don't 500.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DIR = PROJECT_ROOT / "mlruns"


def _load_dotenv():
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def setup_mlflow(experiment_name: str) -> None:
    """Point MLflow at MLFLOW_TRACKING_URI if set (e.g. a DagsHub-hosted
    server), otherwise at the local sqlite store under mlruns/, and set the
    active experiment."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
