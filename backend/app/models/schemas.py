from typing import Any, Literal

from pydantic import BaseModel, Field


Decision = Literal["AUTO_HANDLE", "ESCALATE"]


class PredictRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)


class Evidence(BaseModel):
    customer_message: str
    brand_response: str
    intent: str | None = None
    similarity: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    intent: str
    intent_confidence: float
    reply: str
    decision: Decision
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    brand: str | None = None
