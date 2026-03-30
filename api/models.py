import joblib
from pydantic import BaseModel, Field, field_validator

options = joblib.load("model/options_info.pkl")

class PredictionRequest(BaseModel):
    date: str = Field(..., example="2025-06-15")
    country: str = Field(..., example=options["countries"][0], json_schema_extra={"enum": options["countries"]})
    category: str = Field(..., example=options["categories"][0], json_schema_extra={"enum": options["categories"]})
    device_type: str = Field(..., example=options["device_types"][0], json_schema_extra={"enum": options["device_types"]})

    @field_validator("country")
    @classmethod
    def validate_country(cls, v):
        if v not in options["countries"]:
            raise ValueError(f"Must be one of {options['countries']}")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if v not in options["categories"]:
            raise ValueError(f"Must be one of {options['categories']}")
        return v

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, v):
        if v not in options["device_types"]:
            raise ValueError(f"Must be one of {options['device_types']}")
        return v
    
class PredictionResponse(BaseModel):
    predicted_sales: float
    date: str
    country: str
    category: str
    device_type: str

class BatchPredictionRequest(BaseModel):
    requests: list[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_forecasted_value: float
    average_forecasted_value: float