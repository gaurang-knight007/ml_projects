import os
from pathlib import Path

# Project root = parent of this file's parent (../)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def project_path(*parts):
    return ROOT.joinpath(*parts)
