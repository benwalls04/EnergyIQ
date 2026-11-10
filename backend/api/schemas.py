"""Request and response bodies, plus the types shared by query parameters."""
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

# the database stores these in upper case; the API accepts either
Utility = Annotated[
    Literal["ELECTRICITY", "HEAT", "COOLING", "GAS"],
    BeforeValidator(lambda v: v.upper() if isinstance(v, str) else v),
]
Interval = Literal["15min", "hour", "day", "month"]
WeatherInterval = Literal["hour", "day", "month"]
AnomalyStatus = Literal["open", "investigating", "resolved", "false_alarm"]
Direction = Literal["spike", "drop"]
AnomalySort = Literal["severity", "time"]


class Health(BaseModel):
    status: str
    readings: int
    flagged_readings: int
    models_loaded: int


class Meter(BaseModel):
    meterid: int
    utility: str
    units: str | None
    first_reading: datetime
    last_reading: datetime
    n_readings: int


class Building(BaseModel):
    id: int
    sitename: str
    simscode: str
    meters: list[Meter]


class ConsumptionPoint(BaseModel):
    time: datetime
    utility: str
    units: str | None
    value: float | None
    n_readings: int
    n_flagged: int


class Consumption(BaseModel):
    building_id: int
    sitename: str
    interval: Interval
    points: list[ConsumptionPoint]


class PredictionPoint(BaseModel):
    time: datetime
    utility: str
    units: str | None
    actual: float | None
    predicted: float | None
    residual: float | None = Field(description="actual - predicted; positive means more than expected")
    n_readings: int
    n_flagged: int


class Predictions(BaseModel):
    building_id: int
    sitename: str
    interval: Interval
    points: list[PredictionPoint]


class Anomaly(BaseModel):
    meterid: int
    readingtime: datetime
    building_id: int
    sitename: str
    utility: str
    units: str | None
    value: float | None
    severity: float | None = Field(
        description="(value - median) / (distance from median to the bound). Null when the bound "
                    "equals the median, so any departure from it is flagged")
    direction: Direction
    median: float | None
    lower_bound: float | None
    upper_bound: float | None
    train_end: str | None = Field(description="Which replay's thresholds flagged this reading")
    status: AnomalyStatus
    notes: str | None
    updated_at: datetime | None = Field(description="When staff last changed the review (UTC)")


class AnomalyPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[Anomaly]


class AnomalyCount(BaseModel):
    building_id: int
    sitename: str
    utility: str
    status: AnomalyStatus
    count: int


class AnomalySummary(BaseModel):
    total: int
    by_status: dict[AnomalyStatus, int]
    groups: list[AnomalyCount]


class AnomalyUpdate(BaseModel):
    """Send either field or both. `notes: null` clears the notes."""
    model_config = ConfigDict(extra="forbid")

    status: AnomalyStatus | None = None
    notes: Annotated[str, Field(max_length=2000)] | None = None

    @model_validator(mode="after")
    def check_fields(self):
        if not self.model_fields_set:
            raise ValueError("send at least one of status, notes")
        if "status" in self.model_fields_set and self.status is None:
            raise ValueError("status can't be null")
        return self


class WeatherPoint(BaseModel):
    time: datetime
    temperature_2m: float | None
    apparent_temperature: float | None
    relative_humidity_2m: float | None
    dew_point_2m: float | None
    precipitation: float | None
    cloud_cover: float | None
    wind_speed_10m: float | None
    wind_speed_100m: float | None
    wind_direction_10m: float | None
    wind_direction_100m: float | None
    shortwave_radiation: float | None
    direct_radiation: float | None
    diffuse_radiation: float | None
    direct_normal_irradiance: float | None


class Weather(BaseModel):
    interval: WeatherInterval
    points: list[WeatherPoint]


class ModelInfo(BaseModel):
    train_end: str
    predicts_from: datetime
    predicts_until: datetime | None
    n_features: int


class ModelMetrics(BaseModel):
    utility: str | None = Field(description="Null on the row covering every utility")
    building_id: int | None = Field(description="Null on rows covering every building")
    sitename: str | None
    mae: float | None
    rmse: float | None
    r2: float | None
    mae_untrimmed: float | None
    n_readings: int
    n_flagged: int


class ModelPerformance(BaseModel):
    train_end: str
    evaluated_from: datetime
    evaluated_until: datetime | None
    groups: list[ModelMetrics]


class PerformanceReport(BaseModel):
    trim_pct: float = Field(description="Largest share of residuals dropped before mae, rmse and r2")
    models: list[ModelPerformance]
