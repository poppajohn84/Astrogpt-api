from sqlalchemy import Column, Date, DateTime, Integer, String, Time, func

from app.db import Base


class BirthProfile(Base):
    __tablename__ = "birth_profiles"

    id = Column(Integer, primary_key=True, index=True)
    profile_name = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    birth_date = Column(Date, nullable=False)
    birth_time = Column(Time, nullable=True)
    city = Column(String, nullable=False)
    state = Column(String, nullable=True)
    country = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
