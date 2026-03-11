import copy
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import traceback
import time as time_module
from datetime import date, datetime, time
from typing import Any, Dict, List, Literal, Optional, Tuple

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
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

from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, Field, field_validator

app = FastAPI(title="AstroGPT API", version="1.0.0")


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


class CompositeRequest(BaseModel):
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


def compute_major_aspects(longitudes: Dict[str, float]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    bodies = list(longitudes.keys())

    for body1, body2 in combinations(bodies, 2):
        lon1 = longitudes[body1]
        lon2 = longitudes[body2]
        delta = abs(lon1 - lon2) % 360.0
        delta = min(delta, 360.0 - delta)

        best_match: Optional[Tuple[str, float, float]] = None
        for aspect_name, aspect_angle, max_orb in ASPECT_SPECS:
            orb = abs(delta - aspect_angle)
            if orb <= max_orb and (best_match is None or orb < best_match[2]):
                best_match = (aspect_name, aspect_angle, orb)

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
    flag = swe.FLG_SWIEPH | swe.FLG_SPEED
    if zodiac == "sidereal":
        flag |= swe.FLG_SIDEREAL

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

    placements: List[Dict[str, Any]] = []
    planet_longitudes: Dict[str, float] = {}
    nodes_included = False
    asteroids_included: List[str] = []
    north_node_lon: Optional[float] = None
    south_node_lon: Optional[float] = None
    for name, pid in PLANETS.items():
        planet_lon = calc_longitude(jd, pid, flag) % 360.0
        sign, deg = lon_to_sign_deg(planet_lon)
        placement: Dict[str, Any] = {
            "body": name,
            "longitude": round(planet_lon, 6),
            "sign": sign,
            "degree_in_sign": round(deg, 3),
        }
        if house_cusps is not None:
            if house_system == "whole_sign" and asc_lon is not None:
                placement["house"] = whole_sign_house_for_longitude(planet_lon, asc_lon)
            else:
                placement["house"] = house_for_longitude(planet_lon, house_cusps)
        placements.append(placement)
        planet_longitudes[name] = planet_lon

    if "North Node" in NODE_BODIES:
        north_node_lon = calc_longitude(jd, NODE_BODIES["North Node"], flag) % 360.0
        south_node_lon = (north_node_lon + 180.0) % 360.0
        for node_name, node_lon in [("North Node", north_node_lon), ("South Node", south_node_lon)]:
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
        planet_longitudes[asteroid_name] = asteroid_lon
        asteroids_included.append(asteroid_name)

    if north_node_lon is not None and south_node_lon is not None:
        planet_longitudes["North Node"] = north_node_lon
        planet_longitudes["South Node"] = south_node_lon

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

    return {
        "meta": meta,
        "placements": placements,
        "grouped_placements": grouped_placements,
        "aspects": aspects,
    }


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


# --------- Routes ---------
@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> Dict[str, Any]:
    ephemeris_loaded = bool(swe) and EPHE_PATH.exists() and any(EPHE_PATH.glob("*.se1"))
    return {"status": "ok", "version": app.version, "ephemeris_loaded": ephemeris_loaded}


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

        location_input = build_place(p)
        location_input = " ".join(location_input.strip().split())
        if not location_input:
            raise ValueError("place could not be built from city/state/country.")

        if (p.lat is None) != (p.lon is None):
            raise ValueError("Provide both lat and lon or neither.")

        if p.lat is None and p.lon is None and p.timezone_offset_hours is None:
            return copy.deepcopy(
                compute_natal_chart_cached(
                    birth_date=p.date,
                    birth_time=p.birth_time,
                    city=p.city,
                    state=p.state,
                    country=p.country,
                    zodiac=req.zodiac,
                    house_system=req.house_system,
                )
            )

        if p.lat is not None and p.lon is not None:
            birth_lat = p.lat
            birth_lon = p.lon
            location_resolved = location_input
        else:
            birth_lat, birth_lon, location_resolved = geocode_place(p.city, p.state, p.country)

        return compute_natal_chart(
            birth_date=p.date,
            birth_time=p.birth_time,
            city=p.city,
            state=p.state,
            country=p.country,
            zodiac=req.zodiac,
            house_system=req.house_system,
            location_input=location_input,
            birth_lat=birth_lat,
            birth_lon=birth_lon,
            timezone_offset_hours=p.timezone_offset_hours,
            location_resolved=location_resolved,
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
