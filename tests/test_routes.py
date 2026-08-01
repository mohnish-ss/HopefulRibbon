from __future__ import annotations

from App.services.facilities import FacilityLookupError


def test_homepage_contains_prominent_disclaimer(client):
    response = client.get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "not a clinical tool" in page
    assert "cannot replace assessment by a qualified medical professional" in page


def test_valid_prediction_route_returns_class_and_probability(client, valid_payload):
    response = client.post("/predict", data=valid_payload)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["predicted_class"] in {"B", "M"}
    assert 0 <= payload["malignant_probability"] <= 1
    assert 0 <= payload["confidence"] <= 1
    assert "not a medical diagnosis" in payload["disclaimer"]


def test_missing_input_returns_generic_validation_error(client, valid_payload):
    valid_payload.pop("texture")
    response = client.post("/predict", data=valid_payload)
    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid measurements. Check all required values and allowed ranges."
    }


def test_nonnumeric_input_returns_validation_error(client, valid_payload):
    valid_payload["texture"] = "private-health-value"
    response = client.post("/predict", data=valid_payload)
    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid measurements. Check all required values and allowed ranges."
    }
    assert "private-health-value" not in response.get_data(as_text=True)


def test_out_of_range_input_returns_validation_error(client, valid_payload):
    valid_payload["radius"] = "999"
    response = client.post("/predict", data=valid_payload)
    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid measurements. Check all required values and allowed ranges."
    }


def test_location_api_failure_is_nonfatal(app, client, valid_payload, monkeypatch):
    def fail_lookup(**kwargs):
        raise FacilityLookupError(
            "Nearby facility recommendations are temporarily unavailable.",
            "mocked timeout",
        )

    class MalignantResult:
        predicted_label = "M"
        display_label = "Malignant"
        malignant_probability = 0.8
        confidence = 0.8
        cautious_message = "The model classified the values as similar to malignant."

    monkeypatch.setattr("App.routes.find_nearby_facilities", fail_lookup)
    monkeypatch.setattr(
        app.extensions["prediction_service"], "predict", lambda _data: MalignantResult()
    )
    response = client.post("/predict", data=valid_payload)
    assert response.status_code == 200
    assert response.get_json()["facility_message"] == (
        "Nearby facility recommendations are temporarily unavailable."
    )


def test_disclaimer_is_present_in_prediction_result(client, valid_payload):
    payload = client.post("/predict", data=valid_payload).get_json()
    assert "Educational demonstration only" in payload["disclaimer"]


def test_non_javascript_form_submission_renders_result(client, valid_payload):
    response = client.post("/", data=valid_payload)
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Malignant-class probability" in page
    assert "not a medical diagnosis" in page
