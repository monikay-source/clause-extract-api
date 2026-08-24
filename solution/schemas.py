"""Pydantic request/response schemas for the clause-extraction API."""
from typing import List
from pydantic import BaseModel, Field, field_validator

MAX_TEXT_CHARS = 2000
MIN_BATCH_ITEMS = 1
MAX_BATCH_ITEMS = 32


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class ExtractResponse(BaseModel):
    clause_type: str
    extracted_value: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class BatchExtractRequest(BaseModel):
    items: List[ExtractRequest] = Field(
        ..., min_length=MIN_BATCH_ITEMS, max_length=MAX_BATCH_ITEMS
    )


class BatchExtractResponse(BaseModel):
    results: List[ExtractResponse]


class HealthResponse(BaseModel):
    status: str
    adapter_loaded: bool
