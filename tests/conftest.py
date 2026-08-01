from __future__ import annotations

from pathlib import Path

import pytest

from App import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "App" / "data" / "breast-cancer.csv"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"


@pytest.fixture()
def app():
    application = create_app(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
            "MODEL_PATH": ARTIFACTS_PATH / "model.joblib",
            "MODEL_METADATA_PATH": ARTIFACTS_PATH / "model_metadata.json",
            "GOOGLE_MAPS_API_KEY": None,
        }
    )
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def valid_payload() -> dict[str, str]:
    return {
        "name": "Test User",
        "email": "",
        "postalcode": "K1A 0B1",
        "radius": "17.99",
        "texture": "10.38",
        "perimeter": "122.8",
    }
