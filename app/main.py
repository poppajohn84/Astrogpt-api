from functools import lru_cache
from typing import Any, Dict, Optional

import httpx
import swisseph as swe
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AstroGPT Local API", version="0.1.0")


# --------- Models ---------
class BirthData(BaseModel):
    date: str  # "YYYY-MM-DD"
    time: Optional[str] = None  # "HH:MM" 24h, optional
    place: str  # "City, State" (US) or "City, Country"
    timezone_offset_hours: float  # e.g. -7 for America/Phoenix
    lat: Optional[float] = None
    lon: Optional[float] = None


class NatalRequest(BaseModel):
    person: BirthData
    house_system: str = "W"  # "W" Whole Sign (we'll compute sign-houses ourselves later)
    zodiac: str = "tropical"  # placeholder for future sidereal support


# --------- Helpers ---------
PLANETS = {
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

SIGNS = [
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

GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
GEOCODE_HEADERS = {
    "User-Agent": "AstroGPT-Local/0.1 (contact: YOUR_EMAIL_OR_SITE)",
}


def parse_datetime_to_jd(date_str: str, time_str: Optional[str], tz_offset_hours: float) -> float:
    y, m, d = map(int, date_str.split("-"))
    if time_str:
        hh, mm = map(int, time_str.split(":"))
        # Convert local time to UT by subtracting offset.
        ut_hours = hh - tz_offset_hours + (mm / 60.0)
    else:
        # Default to noon UT-ish to stabilize planets if no time (houses disabled anyway).
        ut_hours = 12.0
    return swe.julday(y, m, d, ut_hours)


def lon_to_sign_deg(lon: float):
    lon = lon % 360.0
    sign_index = int(lon // 30)
    deg_in_sign = lon - sign_index * 30
    return SIGNS[sign_index], deg_in_sign


@lru_cache(maxsize=512)
def geocode_place(place: str) -> tuple[float, float, str]:
    params = {
        "q": place,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }
    try:
        response = httpx.get(
            GEOCODE_URL,
            params=params,
            headers=GEOCODE_HEADERS,
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Geocoding service error: {exc}") from exc

    results = response.json()
    if not results:
        raise HTTPException(status_code=400, detail=f"Could not resolve place: {place}")

    top = results[0]
    return float(top["lat"]), float(top["lon"]), top.get("display_name", place)


# --------- Routes ---------
@app.post("/natal")
def natal(req: NatalRequest) -> Dict[str, Any]:
    p = req.person
    place = " ".join(p.place.strip().split())
    if not place:
        raise HTTPException(status_code=400, detail="place cannot be empty")

    if (p.lat is None) != (p.lon is None):
        raise HTTPException(status_code=400, detail="Provide both lat and lon or neither.")

    if p.lat is not None and p.lon is not None:
        birth_lat = p.lat
        birth_lon = p.lon
        location_resolved = place
    else:
        birth_lat, birth_lon, location_resolved = geocode_place(place)

    jd = parse_datetime_to_jd(p.date, p.time, p.timezone_offset_hours)

    # Swiss Ephemeris settings
    swe.set_topo(birth_lon, birth_lat, 0)  # topo optional
    flag = swe.FLG_SWIEPH | swe.FLG_SPEED

    placements = []
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

    # IMPORTANT RULE: if no birth time, do NOT compute houses/angles
    has_time = bool(p.time)

    return {
        "meta": {
            "ephemeris": "Swiss Ephemeris via pyswisseph",
            "time_provided": has_time,
            "houses_included": has_time,  # only if time is present and houses are implemented
            "house_system": "Whole Sign (default)" if req.house_system.upper() == "W" else req.house_system,
            "location_input": place,
            "location_resolved": location_resolved,
            "lat": birth_lat,
            "lon": birth_lon,
            "note": "Birth time missing -> houses/angles omitted." if not has_time else None,
        },
        "placements": placements,
        "aspects": [],  # we'll add major aspects next
    }


# Smoke test example:
# curl -X POST "http://127.0.0.1:8080/natal" ^
#   -H "Content-Type: application/json" ^
#   -d "{\"person\":{\"date\":\"1990-08-25\",\"time\":\"14:35\",\"place\":\"Phoenix, Arizona, USA\",\"timezone_offset_hours\":-7},\"house_system\":\"W\",\"zodiac\":\"tropical\"}"
