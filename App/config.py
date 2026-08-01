"""Environment-based application configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Default runtime configuration."""

    SECRET_KEY = os.getenv("SECRET_KEY")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "16384"))
    MODEL_PATH = Path(
        os.getenv("MODEL_PATH", PROJECT_ROOT / "artifacts" / "model.joblib")
    )
    MODEL_METADATA_PATH = Path(
        os.getenv(
            "MODEL_METADATA_PATH",
            PROJECT_ROOT / "artifacts" / "model_metadata.json",
        )
    )
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    FACILITY_API_TIMEOUT = float(os.getenv("FACILITY_API_TIMEOUT", "5"))
    ARTIFACTS_REQUIRED = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"
