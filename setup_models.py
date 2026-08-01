"""Backward-compatible wrapper for the dedicated training command."""

from scripts.train_model import main

if __name__ == "__main__":
    raise SystemExit(main())
