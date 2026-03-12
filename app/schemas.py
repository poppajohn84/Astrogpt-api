from datetime import date, time
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


def normalize_profile_name(value: str) -> str:
    normalized = "_".join(value.strip().lower().split())
    if not normalized:
        raise ValueError("profile_name is required")
    return normalized


def _normalize_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    collapsed = " ".join(value.split())
    return collapsed or None


class BirthProfileBase(BaseModel):
    profile_name: str
    name: Optional[str] = None
    birth_date: date
    birth_time: Optional[time] = None
    city: str
    state: Optional[str] = None
    country: str

    @field_validator("profile_name")
    @classmethod
    def validate_profile_name(cls, value: str) -> str:
        return normalize_profile_name(value)

    @field_validator("name", "state")
    @classmethod
    def normalize_optional_text_fields(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_optional_string(value)

    @field_validator("city", "country")
    @classmethod
    def normalize_required_text_fields(cls, value: str) -> str:
        normalized = _normalize_optional_string(value)
        if normalized is None:
            raise ValueError("value cannot be empty")
        return normalized


class BirthProfileCreate(BirthProfileBase):
    pass


class BirthProfileResponse(BirthProfileBase):
    model_config = ConfigDict(from_attributes=True)


class BirthProfileListItem(BaseModel):
    profile_name: str
    name: Optional[str] = None
    city: str
    state: Optional[str] = None
    country: str

    model_config = ConfigDict(from_attributes=True)
