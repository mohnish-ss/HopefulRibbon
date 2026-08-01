"""Compatibility entry point for local development and WSGI servers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from . import create_app
except ImportError:  # Supports `python App/app.py` from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from App import create_app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=os.getenv("FLASK_DEBUG") == "1")
