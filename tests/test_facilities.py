from __future__ import annotations

import requests
import pytest

from App.services.facilities import FacilityLookupError, find_nearby_facilities


class TimeoutSession:
    def get(self, *args, **kwargs):
        raise requests.Timeout("provider timed out")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class SuccessfulSession:
    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return FakeResponse(
                {
                    "status": "OK",
                    "results": [
                        {"geometry": {"location": {"lat": 45.4, "lng": -75.7}}}
                    ],
                }
            )
        return FakeResponse(
            {
                "status": "OK",
                "results": [
                    {"name": "Example Hospital", "vicinity": "123 Example Street"},
                    {"name": "Second Clinic", "vicinity": "456 Sample Avenue"},
                ],
            }
        )


def test_location_api_failure_is_wrapped_without_provider_details():
    with pytest.raises(FacilityLookupError) as captured:
        find_nearby_facilities("K1A 0B1", "test-key", session=TimeoutSession())
    assert (
        captured.value.user_message
        == "Nearby facility recommendations are temporarily unavailable."
    )
    assert "provider timed out" not in captured.value.user_message


def test_location_api_is_mocked_and_results_are_bounded():
    facilities = find_nearby_facilities(
        "K1A 0B1", "test-key", session=SuccessfulSession()
    )
    assert facilities == [
        {"name": "Example Hospital", "address": "123 Example Street"},
        {"name": "Second Clinic", "address": "456 Sample Avenue"},
    ]
