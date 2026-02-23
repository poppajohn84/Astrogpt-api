from functools import lru_cache
from pathlib import Path
import traceback
from datetime import date, datetime, time
from typing import Any, Dict, List, Literal, Optional, Tuple

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
try:
    import swisseph as swe
except Exception:
    swe = None
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, Field, field_validator

app = FastAPI(title="AstroGPT API", version="1.0.0")

# Absolute ephemeris path (Render-safe)
EPHE_PATH = Path(__file__).resolve().parent.parent / "ephemeris"
if swe:
    swe.set_ephe_path(str(EPHE_PATH))


# --------- Models ---------
class BirthData(BaseModel):
    name: Optional[str] = None

    # Required by design (so GPT will ask for them)
    date: date
    city: str = Field(..., description="Birth city")
    country: str = Field(..., description="Birth country")

    # Optional fields
    birth_time: Optional[time] = Field(
        default=None,
        validation_alias=AliasChoices("birth_time", "time"),
        description="Birth time (optional).",
    )
    state: Optional[str] = None
    place: Optional[str] = None

    lat: Optional[float] = None
    lon: Optional[float] = None
    timezone_offset_hours: Optional[float] = None

    @field_validator("city", "state", "country", "place")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        collapsed = " ".join(value.split())
        return collapsed or None


class NatalRequest(BaseModel):
    person: BirthData
    zodiac: Literal["tropical", "sidereal"] = "tropical"
    house_system: Literal["whole_sign", "placidus"] = "whole_sign"


# --------- Helpers ---------
PLANETS = (
    {
        "Sun": swe.SUN,
        "Moon": swe.MOON,
        "Mercury": swe.MERCURY,
        "Venus": swe.VENUS,
        "Mars": swe.MARS,
        "Jupiter": swe.JUPITER,
        "Saturn": swe.SATURN,
        "Uranus": swe.URANUS,
        "Neptune": swe.NEPTUNE,
        "Pluto": swe.PLUTO,
    }
    if swe
    else {}
)

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
TIMEZONE_LOOKUP_URL = "https://api.open-meteo.com/v1/forecast"


def sanity_check_ephemeris() -> None:
    ephe_dir = Path(EPHE_PATH)
    if not ephe_dir.exists():
        raise ValueError(f"Ephemeris directory not found at: {ephe_dir}")
    if not any(ephe_dir.glob("*.se1")):
        raise ValueError(f"No .se1 ephemeris files found in: {ephe_dir}")


def build_place(p: BirthData) -> str:
    parts = [p.city, p.state, p.country]
    return ", ".join([x.strip() for x in parts if x and x.strip()])


def parse_datetime_to_jd(
    birth_date: date,
    birth_time: Optional[time],
    tz_offset_hours: Optional[float]
) -> float:
    if swe is None:
        raise ValueError("Swiss Ephemeris not available")

    if birth_time:
        if tz_offset_hours is None:
            raise ValueError("timezone_offset_hours is required when birth time is provided.")
        ut_hours = (
            birth_time.hour
            - tz_offset_hours
            + (birth_time.minute / 60.0)
            + (birth_time.second / 3600.0)
        )
    else:
        # Stabilize planets when time unknown (houses disabled anyway)
        ut_hours = 12.0
    return swe.julday(birth_date.year, birth_date.month, birth_date.day, ut_hours)


def lon_to_sign_deg(lon: float):
    lon = lon % 360.0
    sign_index = int(lon // 30)
    deg_in_sign = lon - sign_index * 30
    return SIGNS[sign_index], deg_in_sign


@lru_cache(maxsize=512)
def geocode_place(city: str, state: Optional[str], country: str) -> Tuple[float, float, str]:
    def normalize(value: Optional[str]) -> str:
        if not value:
            return ""
        return " ".join(
            "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in value).split()
        )

    def normalize_country_code(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        letters = "".join(ch for ch in value.upper() if ch.isalpha())
        return letters if len(letters) == 2 else None

    def display_name_for(candidate: Dict[str, Any]) -> str:
        parts = [candidate.get("name"), candidate.get("admin1"), candidate.get("country")]
        cleaned = [str(part).strip() for part in parts if part and str(part).strip()]
        return ", ".join(cleaned) if cleaned else city

    city_norm = normalize(city)
    state_norm = normalize(state)
    country_norm = normalize(country)
    country_code = normalize_country_code(country)

    params = {
        "name": city,
        "count": 5,
        "language": "en",
        "format": "json",
    }
    try:
        response = httpx.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not resolve location. Please confirm City + State/Province + Country.",
        ) from exc

    payload = response.json()
    results = payload.get("results") or []
    if not results:
        raise HTTPException(
            status_code=400,
            detail="Could not resolve location. Please confirm City + State/Province + Country.",
        )

    def score(candidate: Dict[str, Any]) -> int:
        points = 0

        candidate_country_code = str(candidate.get("country_code", "")).upper()
        candidate_country_name = normalize(candidate.get("country"))
        candidate_admin1 = normalize(candidate.get("admin1"))
        candidate_city = normalize(candidate.get("name"))

        if country:
            if country_code and candidate_country_code == country_code:
                points += 3
            elif (
                country_norm
                and candidate_country_name
                and (
                    candidate_country_name == country_norm
                    or country_norm in candidate_country_name
                    or candidate_country_name in country_norm
                )
            ):
                points += 3

        if state:
            if (
                state_norm
                and candidate_admin1
                and (
                    candidate_admin1 == state_norm
                    or state_norm in candidate_admin1
                    or candidate_admin1 in state_norm
                )
            ):
                points += 3

        if city_norm and candidate_city == city_norm:
            points += 1

        return points

    best = max(results, key=lambda cand: (score(cand), int(cand.get("population") or 0)))

    try:
        lat = float(best["latitude"])
        lon = float(best["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not resolve location. Please confirm City + State/Province + Country.",
        ) from exc

    return lat, lon, display_name_for(best)


@lru_cache(maxsize=512)
def lookup_timezone_name(lat: float, lon: float) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        response = httpx.get(TIMEZONE_LOOKUP_URL, params=params, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Timezone lookup service error: {exc}") from exc

    payload = response.json()
    tz_name = payload.get("timezone")
    if not tz_name:
        raise HTTPException(
            status_code=400,
            detail="Could not determine timezone from location. Provide timezone_offset_hours.",
        )
    return tz_name


def resolve_timezone_offset_hours(lat: float, lon: float, birth_date: date, birth_time: time) -> Tuple[float, str]:
    tz_name = lookup_timezone_name(round(lat, 4), round(lon, 4))
    try:
        tzinfo = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Resolved timezone '{tz_name}' is not supported. Provide timezone_offset_hours.",
        ) from exc

    local_dt = datetime.combine(birth_date, birth_time).replace(tzinfo=tzinfo)
    offset = local_dt.utcoffset()
    if offset is None:
        raise HTTPException(
            status_code=400,
            detail="Could not compute timezone offset for birth datetime. Provide timezone_offset_hours.",
        )
    return offset.total_seconds() / 3600.0, tz_name


# --------- Routes ---------
@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/natal")
def natal(req: NatalRequest) -> Dict[str, Any]:
    try:
        if swe is None:
            raise HTTPException(
                status_code=500,
                detail="Swiss Ephemeris not available"
            )

        sanity_check_ephemeris()

        p = req.person

        place = build_place(p)
        place = " ".join(place.strip().split())
        if not place:
            raise ValueError("place could not be built from city/state/country.")

        if (p.lat is None) != (p.lon is None):
            raise ValueError("Provide both lat and lon or neither.")

        if p.lat is not None and p.lon is not None:
            birth_lat = p.lat
            birth_lon = p.lon
            location_resolved = place
        else:
            birth_lat, birth_lon, _location_display = geocode_place(p.city, p.state, p.country)
            location_resolved = place

        timezone_offset_hours = p.timezone_offset_hours
        timezone_name = None
        timezone_source = "input" if timezone_offset_hours is not None else None

        if p.birth_time is not None and timezone_offset_hours is None:
            timezone_offset_hours, timezone_name = resolve_timezone_offset_hours(
                birth_lat, birth_lon, p.date, p.birth_time
            )
            timezone_source = "resolved"

        jd = parse_datetime_to_jd(p.date, p.birth_time, timezone_offset_hours)

        # Swiss Ephemeris settings
        swe.set_topo(birth_lon, birth_lat, 0)
        flag = swe.FLG_SWIEPH | swe.FLG_SPEED
        if req.zodiac == "sidereal":
            flag |= swe.FLG_SIDEREAL

        placements: List[Dict[str, Any]] = []
        for name, pid in PLANETS.items():
            planet_lon, _, _, _ = swe.calc_ut(jd, pid, flag)[0]
            sign, deg = lon_to_sign_deg(planet_lon)
            placements.append(
                {
                    "body": name,
                    "longitude": round(planet_lon, 6),
                    "sign": sign,
                    "degree_in_sign": round(deg, 3),
                }
            )

        aspects: List[Dict[str, Any]] = []

        # Time-known rule: no houses/angles without time
        time_provided = p.birth_time is not None
        house_note = None
        if not time_provided:
            house_note = "Birth time missing -> houses/angles omitted."
        elif req.house_system == "placidus":
            house_note = "Placidus requested; this build currently returns placements/aspects only."

        return {
            "meta": {
                "ephemeris": "Swiss Ephemeris via pyswisseph",
                "ephemeris_path": str(EPHE_PATH),
                "time_provided": time_provided,
                "houses_included": False,
                "house_system": req.house_system,
                "zodiac": req.zodiac,
                "location_input": place,
                "location_resolved": location_resolved,
                "timezone_offset_hours": timezone_offset_hours,
                "timezone_name": timezone_name,
                "timezone_source": timezone_source,
                "note": house_note,
            },
            "placements": placements,
            "aspects": aspects,  # major aspects can be added in a future change
        }

    except HTTPException:
        raise
    except ValueError as exc:
        print(f"Error in /natal: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"Internal error in /natal: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
