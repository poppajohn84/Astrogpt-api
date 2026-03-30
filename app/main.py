import copy
import html
from functools import lru_cache
from itertools import combinations
import math
from pathlib import Path
import traceback
import time as time_module
from datetime import date, datetime, time
from typing import Any, Dict, List, Literal, Optional, Tuple

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from app import models
from app.db import Base, engine, get_db
from app.schemas import (
    BirthProfileCreate,
    BirthProfileListItem,
    BirthProfileResponse,
    normalize_profile_name,
)
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from timezonefinder import TimezoneFinder

EPHE_PATH = Path(__file__).resolve().parent.parent / "ephemeris"
EPHE_FILE = EPHE_PATH / "seas_18.se1"

try:
    import swisseph as swe
    swe.set_ephe_path(str(EPHE_PATH))
    print(f"[startup] EPHE_PATH={EPHE_PATH} seas_18_exists={EPHE_FILE.exists()}")
    if not EPHE_FILE.exists():
        raise RuntimeError(
            "Swiss Ephemeris dataset missing: "
            f"EPHE_PATH='{EPHE_PATH}', "
            f"folder_exists={EPHE_PATH.exists()}, "
            f"ephe_file_exists={EPHE_FILE.exists()}"
        )
except Exception:
    swe = None

from pydantic import AliasChoices, BaseModel, Field, field_validator

app = FastAPI(title="AstroGPT API", version="1.0.0")


@app.on_event("startup")
def create_db_tables() -> None:
    Base.metadata.create_all(bind=engine)


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


class NatalChartImageRequest(BaseModel):
    person: BirthData
    zodiac: Literal["tropical", "sidereal"] = "tropical"
    house_system: Literal["whole_sign", "placidus"] = "whole_sign"
    width: int = 900
    height: int = 900
    include_aspects: bool = True


class TransitRequest(BaseModel):
    person: BirthData
    transit_date: Optional[date] = None
    transit_time: Optional[time] = None
    transit_timezone_offset_hours: Optional[float] = None
    zodiac: Literal["tropical", "sidereal"] = "tropical"


class CompositeRequest(BaseModel):
    person_a: BirthData
    person_b: BirthData
    zodiac: Literal["tropical", "sidereal"] = "tropical"
    house_system: Literal["whole_sign", "placidus"] = "whole_sign"


class SynastryRequest(BaseModel):
    person_a: BirthData
    person_b: BirthData
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

NODE_BODIES = (
    {
        "North Node": swe.TRUE_NODE,
    }
    if swe
    else {}
)

if swe:
    _ASTEROID_CONSTANTS = {
        "Chiron": "CHIRON",
        "Ceres": "CERES",
        "Pallas": "PALLAS",
        "Juno": "JUNO",
        "Vesta": "VESTA",
    }
    _missing_asteroid_constants = [
        const_name for const_name in _ASTEROID_CONSTANTS.values() if not hasattr(swe, const_name)
    ]
    if _missing_asteroid_constants:
        raise RuntimeError(
            "Swiss Ephemeris missing required asteroid constants: "
            + ", ".join(_missing_asteroid_constants)
        )
    ASTEROID_BODIES = {name: getattr(swe, const_name) for name, const_name in _ASTEROID_CONSTANTS.items()}
else:
    ASTEROID_BODIES = {}

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
SIGN_TO_ELEMENT = {
    "Aries": "Fire",
    "Leo": "Fire",
    "Sagittarius": "Fire",
    "Taurus": "Earth",
    "Virgo": "Earth",
    "Capricorn": "Earth",
    "Gemini": "Air",
    "Libra": "Air",
    "Aquarius": "Air",
    "Cancer": "Water",
    "Scorpio": "Water",
    "Pisces": "Water",
}
SIGN_TO_MODALITY = {
    "Aries": "Cardinal",
    "Cancer": "Cardinal",
    "Libra": "Cardinal",
    "Capricorn": "Cardinal",
    "Taurus": "Fixed",
    "Leo": "Fixed",
    "Scorpio": "Fixed",
    "Aquarius": "Fixed",
    "Gemini": "Mutable",
    "Virgo": "Mutable",
    "Sagittarius": "Mutable",
    "Pisces": "Mutable",
}
DOMINANCE_BODIES = {
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
}
TRADITIONAL_SIGN_RULERS = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}
PLANETARY_STRENGTH_BASE = {
    "Sun": 5,
    "Moon": 5,
    "Mercury": 3,
    "Venus": 3,
    "Mars": 3,
    "Jupiter": 2,
    "Saturn": 2,
    "Uranus": 1,
    "Neptune": 1,
    "Pluto": 1,
    "North Node": 2,
    "South Node": 2,
    "Chiron": 1,
    "Ceres": 1,
    "Pallas": 1,
    "Juno": 1,
    "Vesta": 1,
}

OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ASPECT_SPECS = (
    ("conjunction", 0.0, 8.0),
    ("sextile", 60.0, 5.0),
    ("square", 90.0, 7.0),
    ("trine", 120.0, 7.0),
    ("opposition", 180.0, 8.0),
)
HOUSE_SYSTEM_CODE = {
    "whole_sign": b"W",
    "placidus": b"P",
}
SIGN_LABELS = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]
SIGN_GLYPHS = {
    "Aries": "\u2648",
    "Taurus": "\u2649",
    "Gemini": "\u264A",
    "Cancer": "\u264B",
    "Leo": "\u264C",
    "Virgo": "\u264D",
    "Libra": "\u264E",
    "Scorpio": "\u264F",
    "Sagittarius": "\u2650",
    "Capricorn": "\u2651",
    "Aquarius": "\u2652",
    "Pisces": "\u2653",
}
PLANET_GLYPHS = {
    "Sun": "\u2609",
    "Moon": "\u263D",
    "Mercury": "\u263F",
    "Venus": "\u2640",
    "Mars": "\u2642",
    "Jupiter": "\u2643",
    "Saturn": "\u2644",
    "Uranus": "\u2645",
    "Neptune": "\u2646",
    "Pluto": "\u2647",
}
NODE_GLYPHS = {
    "North Node": "\u260A",
    "South Node": "\u260B",
}
ASTEROID_LABELS = {
    "Chiron": "Ch",
    "Ceres": "Ce",
    "Pallas": "Pa",
    "Juno": "Ju",
    "Vesta": "Ve",
}
BODY_LABELS = {
    **PLANET_GLYPHS,
    **NODE_GLYPHS,
    **ASTEROID_LABELS,
}
TIMEZONE_FINDER = TimezoneFinder(in_memory=True)
GEOCODE_HEADERS = {"User-Agent": "AstroGPT/1.0 (contact: admin@yourdomain.com)"}
US_COUNTRY_ALIASES = {
    "us",
    "usa",
    "u s",
    "u s a",
    "u.s",
    "u.s.",
    "u.s.a",
    "u.s.a.",
    "united states",
    "united states of america",
}


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


def calc_longitude(jd: float, body: int, flag: Optional[int] = None) -> float:
    if swe is None:
        raise ValueError("Swiss Ephemeris not available")
    swe.set_ephe_path(str(EPHE_PATH))
    if flag is None:
        xx, _ = swe.calc_ut(jd, body)
    else:
        xx, _ = swe.calc_ut(jd, body, flag)
    return xx[0]


def midpoint_longitude(lon_a: float, lon_b: float) -> float:
    lon_a = lon_a % 360.0
    lon_b = lon_b % 360.0
    delta = ((lon_b - lon_a + 540.0) % 360.0) - 180.0
    return (lon_a + (delta / 2.0)) % 360.0


def placements_to_longitude_map(placements: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        placement["body"]: float(placement["longitude"])
        for placement in placements
        if "body" in placement and "longitude" in placement
    }


def _normalize_location_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(
        "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in value).split()
    )


def _normalize_country_key(value: Optional[str]) -> str:
    normalized = _normalize_location_text(value)
    if normalized in US_COUNTRY_ALIASES:
        return "us"
    letters = "".join(ch for ch in normalized if ch.isalpha())
    if len(letters) == 2 and letters == "us":
        return "us"
    return normalized


def _country_code_from_key(country_key: str) -> Optional[str]:
    if not country_key:
        return None
    if country_key == "us":
        return "US"
    letters = "".join(ch for ch in country_key.upper() if ch.isalpha())
    return letters if len(letters) == 2 else None


def _display_name_for(candidate: Dict[str, Any], fallback_city: str) -> str:
    parts = [candidate.get("name"), candidate.get("admin1"), candidate.get("country")]
    cleaned = [str(part).strip() for part in parts if part and str(part).strip()]
    return ", ".join(cleaned) if cleaned else fallback_city


def _score_candidate(
    candidate: Dict[str, Any],
    city_key: str,
    state_key: str,
    country_key: str,
    country_code: Optional[str],
) -> int:
    points = 0

    candidate_country_code = str(candidate.get("country_code", "")).upper()
    candidate_country_name = _normalize_country_key(candidate.get("country"))
    candidate_admin1 = _normalize_location_text(candidate.get("admin1"))
    candidate_city = _normalize_location_text(candidate.get("name"))

    if country_key:
        if country_code and candidate_country_code == country_code:
            points += 3
        elif (
            candidate_country_name
            and (
                candidate_country_name == country_key
                or country_key in candidate_country_name
                or candidate_country_name in country_key
            )
        ):
            points += 3

    if state_key:
        if (
            candidate_admin1
            and (
                candidate_admin1 == state_key
                or state_key in candidate_admin1
                or candidate_admin1 in state_key
            )
        ):
            points += 3

    if city_key and candidate_city == city_key:
        points += 1

    return points


@lru_cache(maxsize=2048)
def _geocode_cached(city_key: str, state_key: str, country_key: str) -> Tuple[float, float, str]:
    retries = [0.5, 1.0, 2.0]
    country_code = _country_code_from_key(country_key)
    last_status: Optional[int] = None

    for attempt in range(len(retries) + 1):
        try:
            response = httpx.get(
                OPEN_METEO_GEOCODE_URL,
                params={"name": city_key, "count": 5, "language": "en", "format": "json"},
                headers=GEOCODE_HEADERS,
                timeout=10.0,
            )
            status_code = response.status_code
            if status_code == 429:
                last_status = 429
                if attempt < len(retries):
                    time_module.sleep(retries[attempt])
                    continue
                raise HTTPException(
                    status_code=503,
                    detail="Geocoding is temporarily rate-limited. Please retry in ~30 seconds.",
                )
            if 500 <= status_code <= 599:
                last_status = status_code
                if attempt < len(retries):
                    time_module.sleep(retries[attempt])
                    continue
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []
            if not results:
                raise HTTPException(
                    status_code=400,
                    detail="Could not resolve location. Please confirm City + State/Province + Country.",
                )
            best = max(
                results,
                key=lambda cand: (
                    _score_candidate(cand, city_key, state_key, country_key, country_code),
                    int(cand.get("population") or 0),
                ),
            )
            lat = float(best["latitude"])
            lon = float(best["longitude"])
            return lat, lon, _display_name_for(best, city_key)
        except HTTPException:
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise HTTPException(
                    status_code=503,
                    detail="Geocoding is temporarily rate-limited. Please retry in ~30 seconds.",
                ) from exc
            raise HTTPException(
                status_code=400,
                detail="Could not resolve location. Please confirm City + State/Province + Country.",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=400,
                detail="Could not resolve location. Please confirm City + State/Province + Country.",
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Could not resolve location. Please confirm City + State/Province + Country.",
            ) from exc

    if last_status == 429:
        raise HTTPException(
            status_code=503,
            detail="Geocoding is temporarily rate-limited. Please retry in ~30 seconds.",
        )
    raise HTTPException(
        status_code=400,
        detail="Could not resolve location. Please confirm City + State/Province + Country.",
    )


def geocode_place(city: str, state: Optional[str], country: str) -> Tuple[float, float, str]:
    city_key = _normalize_location_text(city)
    state_key = _normalize_location_text(state)
    country_key = _normalize_country_key(country)
    if not city_key or not country_key:
        raise HTTPException(
            status_code=400,
            detail="Could not resolve location. Please confirm City + State/Province + Country.",
        )
    return _geocode_cached(city_key, state_key, country_key)


@lru_cache(maxsize=512)
def lookup_timezone_name(lat_rounded: float, lon_rounded: float) -> str:
    tz_name = TIMEZONE_FINDER.timezone_at(lat=lat_rounded, lng=lon_rounded)
    if not tz_name:
        raise HTTPException(
            status_code=400,
            detail="Could not determine timezone from location. Please confirm City + State/Province + Country.",
        )
    return tz_name


def resolve_timezone_offset_hours(
    lat: float,
    lon: float,
    birth_date: date,
    birth_time: time,
    tz_name: Optional[str] = None,
) -> Tuple[float, str]:
    resolved_tz_name = tz_name or lookup_timezone_name(round(lat, 4), round(lon, 4))
    try:
        tzinfo = ZoneInfo(resolved_tz_name)
    except ZoneInfoNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Resolved timezone '{resolved_tz_name}' is not supported.",
        ) from exc

    local_dt = datetime.combine(birth_date, birth_time).replace(tzinfo=tzinfo)
    offset = local_dt.utcoffset()
    if offset is None:
        raise HTTPException(
            status_code=400,
            detail="Could not compute timezone offset for birth datetime.",
        )
    return offset.total_seconds() / 3600.0, resolved_tz_name


def normalize_cusps(cusps_raw: Tuple[float, ...]) -> List[float]:
    cusps = [float(cusp) % 360.0 for cusp in cusps_raw]
    if len(cusps) == 13 and abs(cusps[0]) < 1e-8:
        cusps = cusps[1:13]
    elif len(cusps) >= 12:
        cusps = cusps[:12]
    if len(cusps) != 12:
        raise ValueError("House cusp calculation failed.")
    return cusps


def longitude_in_segment(value: float, start: float, end: float) -> bool:
    if start <= end:
        return start <= value < end
    return value >= start or value < end


def house_for_longitude(lon: float, cusps: List[float]) -> int:
    value = lon % 360.0
    for idx in range(12):
        start = cusps[idx] % 360.0
        end = cusps[(idx + 1) % 12] % 360.0
        if longitude_in_segment(value, start, end):
            return idx + 1
    return 12


def whole_sign_house_for_longitude(planet_lon: float, asc_lon: float) -> int:
    asc_sign = int((asc_lon % 360.0) // 30.0)
    planet_sign = int((planet_lon % 360.0) // 30.0)
    return ((planet_sign - asc_sign) % 12) + 1


def find_matching_aspect(delta: float) -> Optional[Tuple[str, float, float]]:
    best_match: Optional[Tuple[str, float, float]] = None
    for aspect_name, aspect_angle, max_orb in ASPECT_SPECS:
        orb = abs(delta - aspect_angle)
        if orb <= max_orb and (best_match is None or orb < best_match[2]):
            best_match = (aspect_name, aspect_angle, orb)
    return best_match


def compute_major_aspects(longitudes: Dict[str, float]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    bodies = list(longitudes.keys())

    for body1, body2 in combinations(bodies, 2):
        lon1 = longitudes[body1]
        lon2 = longitudes[body2]
        delta = abs(lon1 - lon2) % 360.0
        delta = min(delta, 360.0 - delta)

        best_match = find_matching_aspect(delta)
        if best_match is not None:
            aspect_name, aspect_angle, orb = best_match
            results.append(
                {
                    "between": [body1, body2],
                    "type": aspect_name,
                    "angle": aspect_angle,
                    "delta": round(delta, 3),
                    "orb": round(orb, 3),
                }
            )

    results.sort(key=lambda item: item["orb"])
    return results


def compute_body_positions(
    jd: float,
    zodiac: Literal["tropical", "sidereal"],
    house_cusps: Optional[List[float]] = None,
    asc_lon: Optional[float] = None,
    house_system: Literal["whole_sign", "placidus"] = "whole_sign",
) -> Tuple[List[Dict[str, Any]], Dict[str, float], bool, List[str]]:
    flag = swe.FLG_SWIEPH | swe.FLG_SPEED
    if zodiac == "sidereal":
        flag |= swe.FLG_SIDEREAL

    placements: List[Dict[str, Any]] = []
    body_longitudes: Dict[str, float] = {}
    nodes_included = False
    asteroids_included: List[str] = []

    for name, pid in PLANETS.items():
        body_lon = calc_longitude(jd, pid, flag) % 360.0
        sign, deg = lon_to_sign_deg(body_lon)
        placement: Dict[str, Any] = {
            "body": name,
            "longitude": round(body_lon, 6),
            "sign": sign,
            "degree_in_sign": round(deg, 3),
        }
        if house_cusps is not None:
            if house_system == "whole_sign" and asc_lon is not None:
                placement["house"] = whole_sign_house_for_longitude(body_lon, asc_lon)
            else:
                placement["house"] = house_for_longitude(body_lon, house_cusps)
        placements.append(placement)
        body_longitudes[name] = body_lon

    if "North Node" in NODE_BODIES:
        north_node_lon = calc_longitude(jd, NODE_BODIES["North Node"], flag) % 360.0
        south_node_lon = (north_node_lon + 180.0) % 360.0
        for node_name, node_lon in (("North Node", north_node_lon), ("South Node", south_node_lon)):
            sign, deg = lon_to_sign_deg(node_lon)
            node_placement: Dict[str, Any] = {
                "body": node_name,
                "longitude": round(node_lon, 6),
                "sign": sign,
                "degree_in_sign": round(deg, 3),
            }
            if house_cusps is not None:
                if house_system == "whole_sign" and asc_lon is not None:
                    node_placement["house"] = whole_sign_house_for_longitude(node_lon, asc_lon)
                else:
                    node_placement["house"] = house_for_longitude(node_lon, house_cusps)
            placements.append(node_placement)
            body_longitudes[node_name] = node_lon
        nodes_included = True

    for asteroid_name, asteroid_id in ASTEROID_BODIES.items():
        asteroid_lon = calc_longitude(jd, asteroid_id, flag) % 360.0
        sign, deg = lon_to_sign_deg(asteroid_lon)
        asteroid_placement: Dict[str, Any] = {
            "body": asteroid_name,
            "longitude": round(asteroid_lon, 6),
            "sign": sign,
            "degree_in_sign": round(deg, 3),
        }
        if house_cusps is not None:
            if house_system == "whole_sign" and asc_lon is not None:
                asteroid_placement["house"] = whole_sign_house_for_longitude(asteroid_lon, asc_lon)
            else:
                asteroid_placement["house"] = house_for_longitude(asteroid_lon, house_cusps)
        placements.append(asteroid_placement)
        body_longitudes[asteroid_name] = asteroid_lon
        asteroids_included.append(asteroid_name)

    return placements, body_longitudes, nodes_included, asteroids_included


def resolve_transit_moment(
    transit_date: Optional[date],
    transit_time: Optional[time],
) -> Tuple[date, time, bool]:
    if transit_date is None and transit_time is None:
        now_utc = datetime.now(ZoneInfo("UTC"))
        return now_utc.date(), time(now_utc.hour, now_utc.minute, now_utc.second), True
    if transit_date is None:
        now_utc = datetime.now(ZoneInfo("UTC"))
        return now_utc.date(), transit_time, False
    if transit_time is None:
        return transit_date, time(12, 0, 0), False
    return transit_date, transit_time, False


def compute_transit_positions(
    transit_date: date,
    transit_time: time,
    transit_timezone_offset_hours: float,
    zodiac: Literal["tropical", "sidereal"],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    jd = parse_datetime_to_jd(
        transit_date,
        transit_time,
        transit_timezone_offset_hours,
    )
    placements, transit_longitudes, _, _ = compute_body_positions(jd=jd, zodiac=zodiac)
    return placements, transit_longitudes


def compute_transit_aspects(
    natal_longitudes: Dict[str, float],
    transit_longitudes: Dict[str, float],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for transit_body, transit_lon in transit_longitudes.items():
        for natal_body, natal_lon in natal_longitudes.items():
            delta = abs(transit_lon - natal_lon) % 360.0
            delta = min(delta, 360.0 - delta)

            best_match = find_matching_aspect(delta)
            if best_match is None:
                continue

            aspect_name, aspect_angle, orb = best_match
            results.append(
                {
                    "between": [f"Transit {transit_body}", f"Natal {natal_body}"],
                    "type": aspect_name,
                    "angle": aspect_angle,
                    "delta": round(delta, 3),
                    "orb": round(orb, 3),
                }
            )

    results.sort(key=lambda item: item["orb"])
    return results


def build_synastry_natal_charts(
    person_a: BirthData,
    person_b: BirthData,
    zodiac: Literal["tropical", "sidereal"],
    house_system: Literal["whole_sign", "placidus"],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def build_chart(person: BirthData, label: str) -> Dict[str, Any]:
        try:
            return build_natal_chart_for_person(
                person=person,
                zodiac=zodiac,
                house_system=house_system,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            raise HTTPException(
                status_code=400,
                detail=f"Could not generate natal chart for {label}: {detail}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not generate natal chart for {label}: {exc}",
            ) from exc

    return build_chart(person_a, "person_a"), build_chart(person_b, "person_b")


def compute_synastry_aspects(
    map_a: Dict[str, float],
    map_b: Dict[str, float],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for body_a, lon_a in map_a.items():
        for body_b, lon_b in map_b.items():
            delta = abs(lon_a - lon_b) % 360.0
            delta = min(delta, 360.0 - delta)

            best_match = find_matching_aspect(delta)
            if best_match is None:
                continue

            aspect_name, aspect_angle, orb = best_match
            results.append(
                {
                    "between": [f"{body_a} (A)", f"{body_b} (B)"],
                    "type": aspect_name,
                    "angle": aspect_angle,
                    "delta": round(delta, 3),
                    "orb": round(orb, 3),
                }
            )

    results.sort(key=lambda item: item["orb"])
    return results


def polar_to_cartesian(cx: float, cy: float, radius: float, angle_deg: float) -> Tuple[float, float]:
    radians = math.radians(angle_deg - 90.0)
    return (
        cx + (radius * math.cos(radians)),
        cy + (radius * math.sin(radians)),
    )


def longitude_to_wheel_angle(longitude: float, asc_longitude: Optional[float] = None) -> float:
    reference = asc_longitude if asc_longitude is not None else 0.0
    relative_longitude = (longitude - reference) % 360.0
    # Keep ASC fixed at 9 o'clock, then reverse the rendered wheel by
    # subtracting the zodiac delta from the left-hand anchor.
    return (270.0 - relative_longitude) % 360.0


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = "#1f2933",
    stroke_width: float = 1.0,
    opacity: float = 1.0,
    dasharray: Optional[str] = None,
) -> str:
    dash_attr = f' stroke-dasharray="{html.escape(dasharray, quote=True)}"' if dasharray else ""
    return (
        f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
        f'stroke="{html.escape(stroke, quote=True)}" stroke-width="{stroke_width:.3f}" '
        f'opacity="{opacity:.3f}" stroke-linecap="round"{dash_attr} />'
    )


def svg_circle(
    cx: float,
    cy: float,
    radius: float,
    stroke: str = "#1f2933",
    stroke_width: float = 1.0,
    fill: str = "none",
    opacity: float = 1.0,
) -> str:
    return (
        f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{radius:.3f}" '
        f'stroke="{html.escape(stroke, quote=True)}" stroke-width="{stroke_width:.3f}" '
        f'fill="{html.escape(fill, quote=True)}" opacity="{opacity:.3f}" />'
    )


def svg_text(
    x: float,
    y: float,
    text: str,
    font_size: float = 12.0,
    fill: str = "#1f2933",
    text_anchor: str = "middle",
    dominant_baseline: str = "middle",
    font_weight: str = "normal",
    rotate: Optional[float] = None,
) -> str:
    rotate_attr = (
        f' transform="rotate({rotate:.3f} {x:.3f} {y:.3f})"'
        if rotate is not None
        else ""
    )
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" font-size="{font_size:.3f}" '
        f'fill="{html.escape(fill, quote=True)}" text-anchor="{html.escape(text_anchor, quote=True)}" '
        f'dominant-baseline="{html.escape(dominant_baseline, quote=True)}" '
        f'font-family="Helvetica, Arial, sans-serif" '
        f'font-weight="{html.escape(font_weight, quote=True)}"{rotate_attr}>'
        f"{html.escape(text)}"
        "</text>"
    )


def build_natal_chart_svg(
    chart: Dict[str, Any],
    width: int,
    height: int,
    include_aspects: bool,
) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")

    placements = [
        placement
        for placement in chart.get("placements", [])
        if "body" in placement and "longitude" in placement
    ]
    if not placements:
        raise ValueError("Natal chart did not include placements.")

    meta = chart.get("meta", {})
    angles = meta.get("angles") if isinstance(meta.get("angles"), dict) else {}
    asc_longitude = float(angles["asc"]) if isinstance(angles, dict) and "asc" in angles else None
    house_cusps = meta.get("house_cusps") if isinstance(meta.get("house_cusps"), list) else []

    cx = width / 2.0
    cy = height / 2.0
    outer_radius = min(width, height) * 0.46
    zodiac_inner_radius = outer_radius * 0.86
    planet_ring_radius = outer_radius * 0.72
    house_inner_radius = outer_radius * 0.30
    aspect_radius = outer_radius * 0.22
    label_font_size = max(10.0, outer_radius * 0.035)
    sign_font_size = max(11.0, outer_radius * 0.040)
    angle_font_size = max(10.0, outer_radius * 0.034)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Natal chart wheel">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        svg_circle(cx, cy, outer_radius, stroke="#111111", stroke_width=2.2),
        svg_circle(cx, cy, zodiac_inner_radius, stroke="#111111", stroke_width=1.6),
        svg_circle(cx, cy, planet_ring_radius, stroke="#444444", stroke_width=1.0, opacity=0.7),
        svg_circle(cx, cy, house_inner_radius, stroke="#555555", stroke_width=1.0, opacity=0.65),
    ]

    if include_aspects and chart.get("aspects"):
        placement_map = {
            str(placement["body"]): placement
            for placement in placements
        }
        for aspect in chart["aspects"]:
            between = aspect.get("between")
            if not isinstance(between, list) or len(between) != 2:
                continue
            body_a, body_b = str(between[0]), str(between[1])
            placement_a = placement_map.get(body_a)
            placement_b = placement_map.get(body_b)
            if placement_a is None or placement_b is None:
                continue
            angle_a = longitude_to_wheel_angle(float(placement_a["longitude"]), asc_longitude)
            angle_b = longitude_to_wheel_angle(float(placement_b["longitude"]), asc_longitude)
            x1, y1 = polar_to_cartesian(cx, cy, aspect_radius, angle_a)
            x2, y2 = polar_to_cartesian(cx, cy, aspect_radius, angle_b)
            parts.append(svg_line(x1, y1, x2, y2, stroke="#7a7a7a", stroke_width=1.2, opacity=0.55))

    for sign_index, sign_name in enumerate(SIGN_LABELS):
        boundary_angle = longitude_to_wheel_angle(sign_index * 30.0, asc_longitude)
        inner_x, inner_y = polar_to_cartesian(cx, cy, zodiac_inner_radius, boundary_angle)
        outer_x, outer_y = polar_to_cartesian(cx, cy, outer_radius, boundary_angle)
        parts.append(svg_line(inner_x, inner_y, outer_x, outer_y, stroke="#111111", stroke_width=1.2))

        label_angle = longitude_to_wheel_angle((sign_index * 30.0) + 15.0, asc_longitude)
        label_x, label_y = polar_to_cartesian(
            cx,
            cy,
            (outer_radius + zodiac_inner_radius) / 2.0,
            label_angle,
        )
        parts.append(
            svg_text(
                label_x,
                label_y,
                SIGN_GLYPHS.get(sign_name, sign_name),
                font_size=sign_font_size,
                fill="#111111",
                font_weight="bold",
            )
        )

    if len(house_cusps) == 12:
        for idx, cusp in enumerate(house_cusps):
            cusp_angle = longitude_to_wheel_angle(float(cusp), asc_longitude)
            start_x, start_y = polar_to_cartesian(cx, cy, house_inner_radius, cusp_angle)
            end_x, end_y = polar_to_cartesian(cx, cy, zodiac_inner_radius, cusp_angle)
            stroke_width = 1.8 if idx in (0, 9) else 1.0
            parts.append(
                svg_line(start_x, start_y, end_x, end_y, stroke="#2f2f2f", stroke_width=stroke_width, opacity=0.85)
            )

    if isinstance(angles, dict):
        for key, label in (("asc", "ASC"), ("mc", "MC")):
            if key not in angles:
                continue
            angle = longitude_to_wheel_angle(float(angles[key]), asc_longitude)
            start_x, start_y = polar_to_cartesian(cx, cy, house_inner_radius * 0.75, angle)
            end_x, end_y = polar_to_cartesian(cx, cy, outer_radius + 10.0, angle)
            label_x, label_y = polar_to_cartesian(cx, cy, outer_radius + 24.0, angle)
            if label_x > cx + 8.0:
                angle_text_anchor = "start"
                label_x += 6.0
            elif label_x < cx - 8.0:
                angle_text_anchor = "end"
                label_x -= 6.0
            else:
                angle_text_anchor = "middle"
            parts.append(svg_line(start_x, start_y, end_x, end_y, stroke="#000000", stroke_width=2.4))
            parts.append(
                svg_text(
                    label_x,
                    label_y,
                    label,
                    font_size=angle_font_size,
                    fill="#000000",
                    text_anchor=angle_text_anchor,
                    font_weight="bold",
                )
            )

    placement_entries = sorted(
        [
            {
                "placement": placement,
                "angle": longitude_to_wheel_angle(float(placement["longitude"]), asc_longitude),
            }
            for placement in placements
        ],
        key=lambda entry: float(entry["angle"]),
    )
    cluster_threshold = 7.5
    body_clusters: List[List[Dict[str, Any]]] = []

    for entry in placement_entries:
        if not body_clusters:
            body_clusters.append([entry])
            continue
        previous_angle = float(body_clusters[-1][-1]["angle"])
        if (float(entry["angle"]) - previous_angle) < cluster_threshold:
            body_clusters[-1].append(entry)
        else:
            body_clusters.append([entry])

    if len(body_clusters) > 1:
        wrap_gap = (float(body_clusters[0][0]["angle"]) + 360.0) - float(body_clusters[-1][-1]["angle"])
        if wrap_gap < cluster_threshold:
            body_clusters[0] = body_clusters[-1] + body_clusters[0]
            body_clusters.pop()

    min_marker_radius = house_inner_radius + 42.0
    min_label_radius = house_inner_radius + 18.0
    stagger_step = 14.0

    for cluster in body_clusters:
        for cluster_index, entry in enumerate(cluster):
            placement = entry["placement"]
            body = str(placement["body"])
            angle = float(entry["angle"])
            marker_radius = max(min_marker_radius, planet_ring_radius - (cluster_index * stagger_step))
            label_radius = max(min_label_radius, marker_radius - 22.0 - (cluster_index * 2.0))
            marker_x, marker_y = polar_to_cartesian(cx, cy, marker_radius, angle)
            label_x, label_y = polar_to_cartesian(cx, cy, label_radius, angle)

            parts.append(
                svg_line(label_x, label_y, marker_x, marker_y, stroke="#666666", stroke_width=0.9, opacity=0.75)
            )
            parts.append(svg_circle(marker_x, marker_y, 4.2, stroke="#111111", stroke_width=1.1, fill="#ffffff"))
            parts.append(
                svg_text(
                    label_x,
                    label_y,
                    BODY_LABELS.get(body, body),
                    font_size=label_font_size,
                    fill="#111111",
                    font_weight="bold",
                )
            )

    parts.append("</svg>")
    return "\n".join(parts)


def build_exact_chart_snapshot(chart: Dict[str, Any]) -> Dict[str, Any]:
    placements = {
        placement["body"]: placement
        for placement in chart.get("placements", [])
        if "body" in placement
    }
    meta = chart.get("meta", {})

    sun_placement = placements.get("Sun")
    moon_placement = placements.get("Moon")
    if sun_placement is None or moon_placement is None:
        raise ValueError("Synastry requires confirmed Sun and Moon placements.")

    snapshot: Dict[str, Any] = {
        "sun_sign": sun_placement["sign"],
        "moon_sign": moon_placement["sign"],
    }

    angles = meta.get("angles")
    if isinstance(angles, dict) and "asc" in angles:
        rising_sign, _ = lon_to_sign_deg(float(angles["asc"]))
        snapshot["rising_sign"] = rising_sign
    else:
        snapshot["rising_sign"] = "Unavailable without birth time"

    return snapshot


def summarize_element_modality(placements: List[Dict[str, Any]]) -> Dict[str, Any]:
    elements = {"Fire": 0, "Earth": 0, "Air": 0, "Water": 0}
    modalities = {"Cardinal": 0, "Fixed": 0, "Mutable": 0}

    for placement in placements:
        body = placement.get("body")
        sign = placement.get("sign")
        if body not in DOMINANCE_BODIES or sign not in SIGN_TO_ELEMENT:
            continue
        elements[SIGN_TO_ELEMENT[sign]] += 1
        modalities[SIGN_TO_MODALITY[sign]] += 1

    def dominant_label(counts: Dict[str, int]) -> str:
        highest = max(counts.values()) if counts else 0
        if highest == 0:
            return "None"
        leaders = [name for name, count in counts.items() if count == highest]
        if len(leaders) == 1:
            return leaders[0]
        return "Tie: " + ", ".join(leaders)

    return {
        "elements": elements,
        "modalities": modalities,
        "dominant_element": dominant_label(elements),
        "dominant_modality": dominant_label(modalities),
    }


def detect_stelliums(placements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    configurations: List[Dict[str, Any]] = []
    sign_buckets: Dict[str, List[str]] = {}
    house_buckets: Dict[int, List[str]] = {}

    for placement in placements:
        body = placement.get("body")
        sign = placement.get("sign")
        house = placement.get("house")
        if body is None:
            continue
        if sign:
            sign_buckets.setdefault(sign, []).append(body)
        if isinstance(house, int):
            house_buckets.setdefault(house, []).append(body)

    for sign in SIGNS:
        bodies = sign_buckets.get(sign, [])
        if len(bodies) >= 3:
            configurations.append(
                {
                    "type": "sign_stellium",
                    "sign": sign,
                    "bodies": bodies,
                }
            )

    for house in sorted(house_buckets):
        bodies = house_buckets[house]
        if len(bodies) >= 3:
            configurations.append(
                {
                    "type": "house_stellium",
                    "house": house,
                    "bodies": bodies,
                }
            )

    return configurations


def compute_chart_ruler(meta: Dict[str, Any]) -> Optional[str]:
    angles = meta.get("angles")
    if not isinstance(angles, dict) or "asc" not in angles:
        return None
    asc_sign, _ = lon_to_sign_deg(float(angles["asc"]))
    return TRADITIONAL_SIGN_RULERS.get(asc_sign)


def compute_planetary_strength_index(
    placements: List[Dict[str, Any]],
    aspects: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    placement_order = [placement["body"] for placement in placements if "body" in placement]
    scores = {
        body: PLANETARY_STRENGTH_BASE.get(body, 0)
        for body in placement_order
    }

    chart_ruler = compute_chart_ruler(meta)
    if chart_ruler in scores:
        scores[chart_ruler] += 4

    angular_houses = {1, 4, 7, 10}
    succedent_houses = {2, 5, 8, 11}
    luminaries = {"Sun", "Moon"}

    for placement in placements:
        body = placement.get("body")
        house = placement.get("house")
        if body not in scores or not isinstance(house, int):
            continue
        if house in angular_houses:
            scores[body] += 3
        elif house in succedent_houses:
            scores[body] += 2
        else:
            scores[body] += 1

    for aspect in aspects:
        between = aspect.get("between", [])
        orb = float(aspect.get("orb", 999.0))
        aspect_bonus = 0
        if orb <= 1:
            aspect_bonus = 4
        elif orb <= 2:
            aspect_bonus = 3
        elif orb <= 4:
            aspect_bonus = 2
        elif orb <= 6:
            aspect_bonus = 1

        for body in between:
            if body in scores:
                scores[body] += aspect_bonus

        if len(between) == 2:
            body_a, body_b = between
            if body_a in luminaries and body_b in scores:
                scores[body_b] += 2
            if body_b in luminaries and body_a in scores:
                scores[body_a] += 2

    stellium_bodies = {
        body
        for configuration in detect_stelliums(placements)
        for body in configuration.get("bodies", [])
        if body in scores
    }
    for body in stellium_bodies:
        scores[body] += 4

    return sorted(
        ({"body": body, "score": score} for body, score in scores.items()),
        key=lambda item: (-item["score"], placement_order.index(item["body"])),
    )


def compute_natal_chart(
    birth_date: date,
    birth_time: Optional[time],
    city: str,
    state: Optional[str],
    country: str,
    zodiac: Literal["tropical", "sidereal"],
    house_system: Literal["whole_sign", "placidus"],
    location_input: str,
    birth_lat: float,
    birth_lon: float,
    timezone_offset_hours: Optional[float],
    location_resolved: str,
) -> Dict[str, Any]:
    timezone_name = None
    timezone_source = "input" if timezone_offset_hours is not None else None

    if birth_time is not None:
        if timezone_offset_hours is None:
            timezone_name = lookup_timezone_name(round(birth_lat, 4), round(birth_lon, 4))
            timezone_offset_hours, timezone_name = resolve_timezone_offset_hours(
                birth_lat, birth_lon, birth_date, birth_time, tz_name=timezone_name
            )
            timezone_source = "resolved"

    jd = parse_datetime_to_jd(birth_date, birth_time, timezone_offset_hours)

    swe.set_topo(birth_lon, birth_lat, 0)

    houses_included = birth_time is not None
    house_cusps: Optional[List[float]] = None
    angles: Optional[Dict[str, float]] = None
    asc_lon: Optional[float] = None
    if houses_included:
        house_code = HOUSE_SYSTEM_CODE[house_system]
        cusps_raw, ascmc = swe.houses(jd, birth_lat, birth_lon, house_code)
        house_cusps = normalize_cusps(tuple(cusps_raw))
        asc_lon = float(ascmc[0]) if len(ascmc) > 0 else 0.0
        asc = asc_lon % 360.0
        mc = float(ascmc[1]) % 360.0 if len(ascmc) > 1 else 0.0
        angles = {
            "asc": round(asc, 6),
            "mc": round(mc, 6),
        }

    placements, planet_longitudes, nodes_included, asteroids_included = compute_body_positions(
        jd=jd,
        zodiac=zodiac,
        house_cusps=house_cusps,
        asc_lon=asc_lon,
        house_system=house_system,
    )

    aspects = compute_major_aspects(planet_longitudes)

    meta: Dict[str, Any] = {
        "ephemeris": "Swiss Ephemeris via pyswisseph",
        "ephemeris_path": str(EPHE_PATH),
        "time_provided": birth_time is not None,
        "houses_included": houses_included,
        "house_system": house_system,
        "zodiac": zodiac,
        "location_input": location_input,
        "location_resolved": location_resolved,
        "timezone_offset_hours": timezone_offset_hours,
        "timezone_name": timezone_name,
        "timezone_source": timezone_source,
    }
    if house_cusps is not None and angles is not None:
        meta["house_cusps"] = [round(cusp, 6) for cusp in house_cusps]
        meta["angles"] = angles
    if nodes_included:
        meta["nodes_included"] = True
    if asteroids_included:
        meta["asteroids_included"] = asteroids_included
        meta["asteroid_aspects_included"] = True
    if birth_time is None:
        meta["limitations"] = "Birth time missing; houses and angles omitted."

    planet_bodies = {
        "Sun", "Moon", "Mercury", "Venus", "Mars",
        "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
    }
    point_bodies = {"North Node", "South Node"}
    asteroid_bodies = {"Chiron", "Ceres", "Pallas", "Juno", "Vesta"}
    grouped_placements: Dict[str, List[Dict[str, Any]]] = {
        "planets": [],
        "points": [],
        "asteroids": [],
    }
    for placement in placements:
        body = placement.get("body")
        if body in planet_bodies:
            grouped_placements["planets"].append(placement)
        elif body in point_bodies:
            grouped_placements["points"].append(placement)
        elif body in asteroid_bodies:
            grouped_placements["asteroids"].append(placement)

    element_modality_summary = summarize_element_modality(placements)
    configurations = detect_stelliums(placements)
    planetary_strength_index = compute_planetary_strength_index(placements, aspects, meta)
    chart_summary = {
        "dominant_element": element_modality_summary["dominant_element"],
        "dominant_modality": element_modality_summary["dominant_modality"],
        "elements": element_modality_summary["elements"],
        "modalities": element_modality_summary["modalities"],
        "configurations": configurations,
        "planetary_strength_index": planetary_strength_index,
    }

    return {
        "meta": meta,
        "placements": placements,
        "grouped_placements": grouped_placements,
        "aspects": aspects,
        "chart_summary": chart_summary,
    }


def build_natal_chart_for_person(
    person: BirthData,
    zodiac: Literal["tropical", "sidereal"],
    house_system: Literal["whole_sign", "placidus"],
) -> Dict[str, Any]:
    location_input = build_place(person)
    location_input = " ".join(location_input.strip().split())
    if not location_input:
        raise ValueError("place could not be built from city/state/country.")

    if (person.lat is None) != (person.lon is None):
        raise ValueError("Provide both lat and lon or neither.")

    if person.lat is None and person.lon is None and person.timezone_offset_hours is None:
        return copy.deepcopy(
            compute_natal_chart_cached(
                birth_date=person.date,
                birth_time=person.birth_time,
                city=person.city,
                state=person.state,
                country=person.country,
                zodiac=zodiac,
                house_system=house_system,
            )
        )

    if person.lat is not None and person.lon is not None:
        birth_lat = person.lat
        birth_lon = person.lon
        location_resolved = location_input
    else:
        birth_lat, birth_lon, location_resolved = geocode_place(
            person.city, person.state, person.country
        )

    return compute_natal_chart(
        birth_date=person.date,
        birth_time=person.birth_time,
        city=person.city,
        state=person.state,
        country=person.country,
        zodiac=zodiac,
        house_system=house_system,
        location_input=location_input,
        birth_lat=birth_lat,
        birth_lon=birth_lon,
        timezone_offset_hours=person.timezone_offset_hours,
        location_resolved=location_resolved,
    )


@lru_cache(maxsize=512)
def compute_natal_chart_cached(
    birth_date: date,
    birth_time: Optional[time],
    city: str,
    state: Optional[str],
    country: str,
    zodiac: Literal["tropical", "sidereal"],
    house_system: Literal["whole_sign", "placidus"],
) -> Dict[str, Any]:
    location_input = ", ".join([part for part in [city, state, country] if part])
    birth_lat, birth_lon, resolved_name = geocode_place(city, state, country)
    return compute_natal_chart(
        birth_date=birth_date,
        birth_time=birth_time,
        city=city,
        state=state,
        country=country,
        zodiac=zodiac,
        house_system=house_system,
        location_input=location_input,
        birth_lat=birth_lat,
        birth_lon=birth_lon,
        timezone_offset_hours=None,
        location_resolved=resolved_name,
    )


def compute_midpoint_composite_chart(
    person_a: BirthData,
    person_b: BirthData,
    zodiac: Literal["tropical", "sidereal"],
    house_system: Literal["whole_sign", "placidus"],
) -> Dict[str, Any]:
    def natal_for_person(person: BirthData) -> Dict[str, Any]:
        location_input = build_place(person)
        location_input = " ".join(location_input.strip().split())
        if not location_input:
            raise ValueError("place could not be built from city/state/country.")

        if (person.lat is None) != (person.lon is None):
            raise ValueError("Provide both lat and lon or neither.")

        if person.lat is None and person.lon is None:
            birth_lat, birth_lon, location_resolved = geocode_place(
                person.city, person.state, person.country
            )
        else:
            birth_lat = person.lat
            birth_lon = person.lon
            if birth_lat is None or birth_lon is None:
                raise ValueError("Provide both lat and lon or neither.")
            location_resolved = location_input

        return compute_natal_chart(
            birth_date=person.date,
            birth_time=person.birth_time,
            city=person.city,
            state=person.state,
            country=person.country,
            zodiac=zodiac,
            house_system=house_system,
            location_input=location_input,
            birth_lat=birth_lat,
            birth_lon=birth_lon,
            timezone_offset_hours=person.timezone_offset_hours,
            location_resolved=location_resolved,
        )

    chart_a = natal_for_person(person_a)
    chart_b = natal_for_person(person_b)

    longitudes_a = placements_to_longitude_map(chart_a["placements"])
    longitudes_b = placements_to_longitude_map(chart_b["placements"])

    placements: List[Dict[str, Any]] = []
    composite_longitudes: Dict[str, float] = {}
    body_order = [placement["body"] for placement in chart_a["placements"]]
    shared_bodies = [body for body in body_order if body in longitudes_b]

    for body in shared_bodies:
        composite_lon = midpoint_longitude(longitudes_a[body], longitudes_b[body])
        sign, deg = lon_to_sign_deg(composite_lon)
        placements.append(
            {
                "body": body,
                "longitude": round(composite_lon, 6),
                "sign": sign,
                "degree_in_sign": round(deg, 3),
            }
        )
        composite_longitudes[body] = composite_lon

    aspects = compute_major_aspects(composite_longitudes)

    houses_included = False
    meta: Dict[str, Any] = {
        "chart_type": "midpoint_composite",
        "zodiac": zodiac,
        "house_system": house_system,
        "time_provided_a": person_a.birth_time is not None,
        "time_provided_b": person_b.birth_time is not None,
        "houses_included": houses_included,
    }
    if not houses_included:
        meta["note"] = "Composite chart uses midpoint method; houses and angles omitted."

    return {
        "meta": meta,
        "placements": placements,
        "aspects": aspects,
    }


def get_birth_profile_or_404(db: Session, profile_name: str) -> models.BirthProfile:
    try:
        normalized_name = normalize_profile_name(profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Profile not found") from exc
    profile = (
        db.query(models.BirthProfile)
        .filter(models.BirthProfile.profile_name == normalized_name)
        .first()
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


# --------- Routes ---------
@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def get_status() -> Dict[str, Any]:
    ephemeris_loaded = bool(swe) and EPHE_PATH.exists() and any(EPHE_PATH.glob("*.se1"))
    return {"status": "ok", "version": app.version, "ephemeris_loaded": ephemeris_loaded}


@app.post("/profiles", response_model=BirthProfileResponse, status_code=status.HTTP_201_CREATED)
def create_profile(
    payload: BirthProfileCreate,
    db: Session = Depends(get_db),
) -> models.BirthProfile:
    existing_profile = (
        db.query(models.BirthProfile)
        .filter(models.BirthProfile.profile_name == payload.profile_name)
        .first()
    )
    if existing_profile is not None:
        raise HTTPException(status_code=409, detail="Profile already exists")

    profile = models.BirthProfile(**payload.model_dump())
    db.add(profile)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Profile already exists") from exc
    db.refresh(profile)
    return profile


@app.get("/profiles", response_model=List[BirthProfileListItem])
def list_profiles(db: Session = Depends(get_db)) -> List[models.BirthProfile]:
    return (
        db.query(models.BirthProfile)
        .order_by(models.BirthProfile.profile_name.asc())
        .all()
    )


@app.get("/profiles/{profile_name}", response_model=BirthProfileResponse)
def get_profile(profile_name: str, db: Session = Depends(get_db)) -> models.BirthProfile:
    return get_birth_profile_or_404(db, profile_name)


@app.delete("/profiles/{profile_name}")
def delete_profile(profile_name: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    profile = get_birth_profile_or_404(db, profile_name)
    normalized_name = profile.profile_name
    db.delete(profile)
    db.commit()
    return {"deleted": True, "profile_name": normalized_name}


@app.post("/natal")
def natal(req: NatalRequest) -> Dict[str, Any]:
    try:
        if swe is None:
            raise HTTPException(
                status_code=500,
                detail="Swiss Ephemeris not available"
            )

        sanity_check_ephemeris()

        return build_natal_chart_for_person(
            person=req.person,
            zodiac=req.zodiac,
            house_system=req.house_system,
        )

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


@app.post("/chart-image/natal")
def natal_chart_image(req: NatalChartImageRequest) -> Dict[str, Any]:
    try:
        if swe is None:
            raise HTTPException(
                status_code=500,
                detail="Swiss Ephemeris not available"
            )

        sanity_check_ephemeris()

        chart = build_natal_chart_for_person(
            person=req.person,
            zodiac=req.zodiac,
            house_system=req.house_system,
        )
        svg = build_natal_chart_svg(
            chart=chart,
            width=req.width,
            height=req.height,
            include_aspects=req.include_aspects,
        )

        return {
            "meta": {
                "chart_type": "natal_svg",
                "width": req.width,
                "height": req.height,
                "zodiac": req.zodiac,
                "house_system": req.house_system,
            },
            "svg": svg,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        print(f"Error in /chart-image/natal: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"Internal error in /chart-image/natal: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chart-image/natal/raw")
def natal_chart_image_raw(req: NatalChartImageRequest) -> Response:
    result = natal_chart_image(req)
    return Response(content=result["svg"], media_type="image/svg+xml")


@app.post("/transits")
def transits(req: TransitRequest) -> Dict[str, Any]:
    try:
        if swe is None:
            raise HTTPException(
                status_code=500,
                detail="Swiss Ephemeris not available"
            )

        sanity_check_ephemeris()

        natal_chart = build_natal_chart_for_person(
            person=req.person,
            zodiac=req.zodiac,
            house_system="whole_sign",
        )
        natal_longitudes = placements_to_longitude_map(natal_chart["placements"])

        user_supplied_transit_moment = req.transit_date is not None or req.transit_time is not None
        if user_supplied_transit_moment and req.transit_timezone_offset_hours is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "To calculate transits accurately for that exact time, I need the timezone "
                    "or city for the transit moment. Otherwise I can calculate it using UTC or "
                    "your current timezone as an assumption."
                ),
            )

        resolved_transit_date, resolved_transit_time, used_utc_defaults = resolve_transit_moment(
            req.transit_date,
            req.transit_time,
        )
        resolved_transit_offset = 0.0 if used_utc_defaults else req.transit_timezone_offset_hours
        transit_placements, transit_longitudes = compute_transit_positions(
            transit_date=resolved_transit_date,
            transit_time=resolved_transit_time,
            transit_timezone_offset_hours=resolved_transit_offset,
            zodiac=req.zodiac,
        )

        return {
            "meta": {
                "chart_type": "transits",
                "transit_date": resolved_transit_date,
                "transit_time": resolved_transit_time,
                "transit_timezone_offset_hours": resolved_transit_offset,
                "transit_timezone_source": "utc_now" if used_utc_defaults else "input",
                "zodiac": req.zodiac,
            },
            "natal_placements": natal_chart["placements"],
            "transit_placements": transit_placements,
            "transit_aspects": compute_transit_aspects(natal_longitudes, transit_longitudes),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        print(f"Error in /transits: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"Internal error in /transits: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/composite")
def composite(req: CompositeRequest) -> Dict[str, Any]:
    try:
        if swe is None:
            raise HTTPException(
                status_code=500,
                detail="Swiss Ephemeris not available"
            )

        sanity_check_ephemeris()

        return compute_midpoint_composite_chart(
            person_a=req.person_a,
            person_b=req.person_b,
            zodiac=req.zodiac,
            house_system=req.house_system,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        print(f"Error in /composite: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"Internal error in /composite: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/synastry")
def synastry(req: SynastryRequest) -> Dict[str, Any]:
    try:
        if swe is None:
            raise HTTPException(
                status_code=500,
                detail="Swiss Ephemeris not available"
            )

        sanity_check_ephemeris()

        chart_a, chart_b = build_synastry_natal_charts(
            person_a=req.person_a,
            person_b=req.person_b,
            zodiac=req.zodiac,
            house_system=req.house_system,
        )

        synastry_aspects = compute_synastry_aspects(
            placements_to_longitude_map(chart_a["placements"]),
            placements_to_longitude_map(chart_b["placements"]),
        )

        return {
            "meta": {
                "chart_type": "synastry",
                "zodiac": req.zodiac,
                "house_system": req.house_system,
                "birth_time_used_a": bool(chart_a.get("meta", {}).get("time_provided")),
                "birth_time_used_b": bool(chart_b.get("meta", {}).get("time_provided")),
            },
            "chart_a": chart_a,
            "chart_b": chart_b,
            "chart_snapshot": {
                "person_a": build_exact_chart_snapshot(chart_a),
                "person_b": build_exact_chart_snapshot(chart_b),
            },
            "synastry_aspects": synastry_aspects,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        print(f"Error in /synastry: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"Internal error in /synastry: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
