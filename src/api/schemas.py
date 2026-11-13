from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TicketIn(BaseModel):
    subject: Optional[str] = Field(default="", description="Ticket subject line")
    description: str = Field(..., min_length=1, description="Full ticket body")


class PredictionOut(BaseModel):
    ticket_id: int
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    model_used: dict


class SuggestResolutionOut(BaseModel):
    ticket_id: int
    suggestion: str
    backend: str
    retrieved_count: int


class FeedbackIn(BaseModel):
    ticket_id: int
    agent_accepted: str = Field(..., pattern="^(yes|no|edited)$")
    corrected_category: Optional[str] = None
    corrected_priority: Optional[str] = None
    corrected_resolution: Optional[str] = None
    notes: Optional[str] = None


class FeedbackOut(BaseModel):
    id: int
    ticket_id: int
    agent_accepted: str


class TicketSummary(BaseModel):
    id: int
    subject: Optional[str]
    description: str
    predicted_category: Optional[str]
    predicted_priority: Optional[str]
    category_confidence: Optional[float]
    priority_confidence: Optional[float]
    suggested_resolution: Optional[str]
    rag_backend: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class StatsOut(BaseModel):
    total_tickets: int
    by_category: Dict[str, int]
    by_priority: Dict[str, int]
    feedback_accept_rate: Optional[float]
