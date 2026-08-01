"""Timeout-bounded Google Maps facility lookup."""

from __future__ import annotations

from dataclasses import dataclass

import requests

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"


@dataclass(frozen=True)
class FacilityLookupError(RuntimeError):
    user_message: str
    log_reason: str


def find_nearby_facilities(
    location_query: str,
    api_key: str | None,
    timeout: float = 5.0,
    session: requests.Session | None = None,
) -> list[dict[str, str]]:
    """Return up to three nearby hospitals without exposing provider details."""
    if not api_key:
        raise FacilityLookupError(
            "Nearby facility recommendations are not configured.",
            "GOOGLE_MAPS_API_KEY is unset",
        )
    if not 3 <= len(location_query) <= 50:
        raise FacilityLookupError(
            "Enter a valid postal code or ZIP code to search nearby facilities.",
            "location query length rejected",
        )

    client = session or requests.Session()
    try:
        geocode_response = client.get(
            GEOCODE_URL,
            params={"address": location_query, "key": api_key},
            timeout=timeout,
        )
        geocode_response.raise_for_status()
        geocode_payload = geocode_response.json()
        geocode_results = geocode_payload.get("results", [])
        if geocode_payload.get("status") != "OK" or not geocode_results:
            raise FacilityLookupError(
                "That location could not be resolved.",
                f"geocoding status={geocode_payload.get('status', 'missing')}",
            )
        location = geocode_results[0]["geometry"]["location"]

        places_response = client.get(
            PLACES_URL,
            params={
                "location": f"{location['lat']},{location['lng']}",
                "radius": 20000,
                "type": "hospital",
                "key": api_key,
            },
            timeout=timeout,
        )
        places_response.raise_for_status()
        places_payload = places_response.json()
        if places_payload.get("status") not in {"OK", "ZERO_RESULTS"}:
            raise FacilityLookupError(
                "Nearby facility recommendations are temporarily unavailable.",
                f"places status={places_payload.get('status', 'missing')}",
            )
    except FacilityLookupError:
        raise
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        raise FacilityLookupError(
            "Nearby facility recommendations are temporarily unavailable.",
            f"provider request failed: {type(error).__name__}",
        ) from error

    facilities = []
    for item in places_payload.get("results", [])[:3]:
        name = str(item.get("name", "")).strip()
        address = str(item.get("vicinity", "")).strip()
        if name:
            facilities.append({"name": name, "address": address})
    return facilities
