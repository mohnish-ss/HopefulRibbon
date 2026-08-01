"""Hopeful Ribbon Flask application factory."""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request
from flask_wtf.csrf import CSRFError, CSRFProtect

from .config import Config
from .routes import web
from .services.prediction import ArtifactError, PredictionService

csrf = CSRFProtect()


def create_app(config: dict[str, object] | None = None) -> Flask:
    """Create the Flask app and load the trained pipeline exactly once."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)
    if config:
        app.config.update(config)

    _validate_secret_key(app)
    _configure_logging(app)
    csrf.init_app(app)

    try:
        app.extensions["prediction_service"] = PredictionService.load(
            app.config["MODEL_PATH"], app.config["MODEL_METADATA_PATH"]
        )
    except ArtifactError:
        if app.config.get("ARTIFACTS_REQUIRED", True):
            raise
        app.extensions["prediction_service"] = None

    app.register_blueprint(web)
    _register_error_handlers(app)
    return app


def _validate_secret_key(app: Flask) -> None:
    if app.config.get("SECRET_KEY"):
        return
    if app.config.get("TESTING"):
        app.config["SECRET_KEY"] = "test-only-secret-key"
        return
    raise RuntimeError(
        "SECRET_KEY is required. Copy .env.example to .env and set a random value."
    )


def _configure_logging(app: Flask) -> None:
    level = logging.DEBUG if app.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(CSRFError)
    def handle_csrf_error(error: CSRFError):
        if request.path == "/predict":
            return (
                jsonify(error="The form expired. Refresh the page and try again."),
                400,
            )
        return render_template("error.html", message="The form expired."), 400

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify(error="The submitted request is too large."), 413

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(
            "Unhandled request error",
            exc_info=(type(error), error, error.__traceback__),
        )
        if request.path == "/predict":
            return jsonify(error="The request could not be completed."), 500
        return (
            render_template(
                "error.html", message="The request could not be completed."
            ),
            500,
        )
