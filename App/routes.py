"""Web routes for the educational classifier."""

from __future__ import annotations

import math

from flask import Blueprint, current_app, jsonify, render_template, request

from .forms import ServiceForm
from .services.facilities import FacilityLookupError, find_nearby_facilities
from .services.prediction import InputValidationError, PredictionService

web = Blueprint("web", __name__)


@web.route("/", methods=["GET", "POST"])
@web.route("/home", methods=["GET", "POST"])
def home():
    service = _prediction_service()
    form = ServiceForm()
    prediction_result = None
    form_error = None
    if request.method == "POST":
        if form.validate_on_submit():
            try:
                result = service.predict(request.form)
                prediction_result = {
                    "is_malignant": result.predicted_label == "M",
                    "name": form.name.data,
                    "message": result.cautious_message,
                    "malignant_probability": result.malignant_probability,
                }
            except InputValidationError:
                form_error = (
                    "Invalid measurements. Check all required values and allowed ranges."
                )
        else:
            form_error = "Please correct the highlighted form fields and try again."
    return render_template(
        "home.html",
        form=form,
        model_summary=service.public_summary,
        prediction_result=prediction_result,
        form_error=form_error,
    )


@web.post("/predict")
def predict():
    """Validate measurements and return an educational classification."""
    try:
        result = _prediction_service().predict(request.form)
    except InputValidationError:
        return jsonify(
            error="Invalid measurements. Check all required values and allowed ranges."
        ), 400

    facilities: list[dict[str, str]] = []
    facility_message: str | None = None
    location_query = _location_query()
    if result.predicted_label == "M" and location_query:
        try:
            facilities = find_nearby_facilities(
                location_query=location_query,
                api_key=current_app.config.get("GOOGLE_MAPS_API_KEY"),
                timeout=current_app.config["FACILITY_API_TIMEOUT"],
            )
            if not facilities:
                facility_message = (
                    "No nearby facilities were returned for that location."
                )
        except FacilityLookupError as error:
            current_app.logger.warning(
                "Facility lookup unavailable: %s", error.log_reason
            )
            facility_message = "Nearby facility recommendations are temporarily unavailable."

    return jsonify(
        prediction=result.display_label,
        predicted_class=result.predicted_label,
        malignant_probability=result.malignant_probability,
        confidence=result.confidence,
        message=result.cautious_message,
        disclaimer=(
            "Educational demonstration only; this classification is not a medical "
            "diagnosis and cannot replace assessment by a qualified medical professional."
        ),
        facilities=facilities,
        facility_message=facility_message,
    )


def _location_query() -> str:
    latitude = request.form.get("latitude", "").strip()
    longitude = request.form.get("longitude", "").strip()
    if latitude or longitude:
        try:
            lat = float(latitude)
            lng = float(longitude)
        except ValueError:
            return request.form.get("postalcode", "").strip()
        if (
            math.isfinite(lat)
            and math.isfinite(lng)
            and -90 <= lat <= 90
            and -180 <= lng <= 180
        ):
            return f"{lat},{lng}"
    return request.form.get("postalcode", "").strip()


def _prediction_service() -> PredictionService:
    service = current_app.extensions.get("prediction_service")
    if not isinstance(service, PredictionService):
        raise RuntimeError("The prediction service is unavailable.")
    return service
