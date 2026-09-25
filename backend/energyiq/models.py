from datetime import datetime
from sqlalchemy import (
    BigInteger,
    Double,
    ForeignKey,
    Index,
    Integer,
    Sequence,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass


# DuckDB has no SERIAL type, so auto-increment IDs need an explicit sequence.
building_id_seq = Sequence("seq_building_id", start=1)


class Building(Base):
    __tablename__ = "buildings"

    id: Mapped[int] = mapped_column(
        Integer,
        building_id_seq,
        server_default=building_id_seq.next_value(),
        primary_key=True,
    )
    sitename: Mapped[str] = mapped_column(String, nullable=False)
    simscode: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    readings: Mapped[list["MeterReading"]] = relationship(back_populates="building")


class MeterReading(Base):
    __tablename__ = "meter_readings"
    __table_args__ = (
        Index(
            "idx_meter_readings_building_utility_time",
            "building_id",
            "utility",
            "readingtime",
        ),
    )

    meterid: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    readingtime: Mapped[datetime] = mapped_column(primary_key=True)
    building_id: Mapped[int] = mapped_column(ForeignKey("buildings.id"), nullable=False)
    utility: Mapped[str] = mapped_column(String, nullable=False)
    readingvalue: Mapped[float | None] = mapped_column(Double)
    readingunits: Mapped[str | None] = mapped_column(String)

    building: Mapped[Building] = relationship(back_populates="readings")


class Weather(Base):
    __tablename__ = "weather"

    date: Mapped[datetime] = mapped_column(primary_key=True)
    latitude: Mapped[float | None] = mapped_column(Double)
    longitude: Mapped[float | None] = mapped_column(Double)
    temperature_2m: Mapped[float | None] = mapped_column(Double)
    shortwave_radiation: Mapped[float | None] = mapped_column(Double)
    direct_radiation: Mapped[float | None] = mapped_column(Double)
    diffuse_radiation: Mapped[float | None] = mapped_column(Double)
    direct_normal_irradiance: Mapped[float | None] = mapped_column(Double)
    relative_humidity_2m: Mapped[float | None] = mapped_column(Double)
    dew_point_2m: Mapped[float | None] = mapped_column(Double)
    precipitation: Mapped[float | None] = mapped_column(Double)
    wind_speed_10m: Mapped[float | None] = mapped_column(Double)
    wind_speed_100m: Mapped[float | None] = mapped_column(Double)
    wind_direction_100m: Mapped[float | None] = mapped_column(Double)
    wind_direction_10m: Mapped[float | None] = mapped_column(Double)
    cloud_cover: Mapped[float | None] = mapped_column(Double)
    apparent_temperature: Mapped[float | None] = mapped_column(Double)