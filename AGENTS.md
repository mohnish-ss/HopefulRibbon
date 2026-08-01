# Repository Guidelines

## Project Structure & Module Organization

This is a Flask educational breast-cancer classification application. `App/__init__.py` owns the application factory, `App/routes.py` exposes the homepage and `/predict`, and `App/app.py` is the local/WSGI entry point. Dataset validation and training live under `App/ml/`; runtime integrations live under `App/services/`. Front-end files remain under `App/templates/` and `App/static/`. The training dataset is `App/data/breast-cancer.csv`, and generated model evidence is stored in `artifacts/`. Train with `scripts/train_model.py`; `setup_models.py` exists only as a compatibility wrapper.

## Build, Test, and Development Commands

Use the repository virtual environment when available:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
python App/app.py
```

The app runs at `http://127.0.0.1:5001`. If activation does not expose `python3`, use the direct executable: `./.venv/bin/python App/app.py`.

Run a quick syntax check before handing off Python changes:

```bash
./.venv/bin/python -m py_compile App/*.py App/ml/*.py App/services/*.py scripts/train_model.py
```

Run `pytest` after changes. Exercise the homepage and `/predict` manually after changing forms, validation, artifact loading, or facility lookup behavior.

## Coding Style & Naming Conventions

Use four spaces for Python indentation and follow PEP 8 naming: `snake_case` for functions and variables, `PascalCase` for classes. Keep Flask routes thin and place form validation in `forms.py` where practical. Use semantic HTML and class-based CSS in `home.html` and `styles.css`; avoid inline styles and keep the visual system flat (no CSS gradients or glass/blur effects). Keep user-facing clinical language cautious and retain the screening disclaimer.

## Security & Configuration

Local secrets belong in `.env`, which is ignored by Git. `SECRET_KEY` is required; `GOOGLE_MAPS_API_KEY` is optional for facility lookup. Never commit secrets or replace generated artifacts unless retraining is intentional and evaluation evidence has been reviewed.

## Commits & Pull Requests

Recent history favors short imperative commit subjects, such as `Fix deployment startup and biopsy form UX`. Keep commits focused and avoid staging generated files such as `__pycache__`. Pull requests should explain the behavior change, list validation performed, link any issue, and include screenshots for visible UI updates.
