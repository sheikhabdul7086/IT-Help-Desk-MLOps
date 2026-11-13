"""Phase 5: FastAPI serving layer.

Endpoints:
  POST /predict             -> category + priority classification
  POST /suggest-resolution  -> RAG-generated troubleshooting playbook
  POST /feedback            -> log agent accept/edit/reject for active learning
  GET  /health              -> liveness check

Run locally with: uvicorn src.api.main:app --reload --port 8000
(local only for this submission — no Docker/cloud deployment yet)
"""
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session

from . import db
from .inference import predict_ticket
from .schemas import (
    FeedbackIn,
    FeedbackOut,
    PredictionOut,
    StatsOut,
    SuggestResolutionOut,
    TicketIn,
    TicketSummary,
)
from ..rag.config import load_config
from ..rag.generate import suggest_resolution

app = FastAPI(
    title="GenAI IT Helpdesk API",
    description="Ticket classification + RAG resolution suggestions (local dev server)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    db.init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(ticket: TicketIn, session: Session = Depends(db.get_session)):
    text = f"{ticket.subject} {ticket.description}".strip()
    if not text:
        raise HTTPException(400, "Ticket description cannot be empty")

    try:
        preds = predict_ticket(text)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    row = db.Ticket(
        subject=ticket.subject,
        description=ticket.description,
        predicted_category=preds["category"]["label"],
        predicted_priority=preds["priority"]["label"],
        category_confidence=preds["category"]["confidence"],
        priority_confidence=preds["priority"]["confidence"],
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    return PredictionOut(
        ticket_id=row.id,
        category=preds["category"]["label"],
        category_confidence=preds["category"]["confidence"],
        priority=preds["priority"]["label"],
        priority_confidence=preds["priority"]["confidence"],
        model_used={"category": preds["category"]["model"], "priority": preds["priority"]["model"]},
    )


@app.post("/suggest-resolution", response_model=SuggestResolutionOut)
def suggest(ticket_id: int, session: Session = Depends(db.get_session)):
    row = session.get(db.Ticket, ticket_id)
    if row is None:
        raise HTTPException(404, f"Ticket {ticket_id} not found. Call /predict first.")

    text = f"{row.subject} {row.description}".strip()
    cfg = load_config()
    result = suggest_resolution(
        text, category=row.predicted_category or "General Inquiry",
        priority=row.predicted_priority or "medium", cfg=cfg,
    )

    row.suggested_resolution = result["suggestion"]
    row.rag_backend = result["backend"]
    session.commit()

    return SuggestResolutionOut(
        ticket_id=ticket_id,
        suggestion=result["suggestion"],
        backend=result["backend"],
        retrieved_count=len(result["retrieved"]),
    )


@app.post("/feedback", response_model=FeedbackOut)
def feedback(fb: FeedbackIn, session: Session = Depends(db.get_session)):
    ticket = session.get(db.Ticket, fb.ticket_id)
    if ticket is None:
        raise HTTPException(404, f"Ticket {fb.ticket_id} not found")

    row = db.Feedback(
        ticket_id=fb.ticket_id,
        agent_accepted=fb.agent_accepted,
        corrected_category=fb.corrected_category,
        corrected_priority=fb.corrected_priority,
        corrected_resolution=fb.corrected_resolution,
        notes=fb.notes,
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    return FeedbackOut(id=row.id, ticket_id=row.ticket_id, agent_accepted=row.agent_accepted)


@app.get("/tickets", response_model=List[TicketSummary])
def list_tickets(limit: int = 20, session: Session = Depends(db.get_session)):
    rows = session.query(db.Ticket).order_by(db.Ticket.id.desc()).limit(limit).all()
    return rows


@app.get("/tickets/{ticket_id}", response_model=TicketSummary)
def get_ticket(ticket_id: int, session: Session = Depends(db.get_session)):
    row = session.get(db.Ticket, ticket_id)
    if row is None:
        raise HTTPException(404, f"Ticket {ticket_id} not found")
    return row


@app.get("/stats", response_model=StatsOut)
def stats(session: Session = Depends(db.get_session)):
    total = session.query(func.count(db.Ticket.id)).scalar() or 0

    by_category = dict(
        session.query(db.Ticket.predicted_category, func.count(db.Ticket.id))
        .filter(db.Ticket.predicted_category.isnot(None))
        .group_by(db.Ticket.predicted_category)
        .all()
    )
    by_priority = dict(
        session.query(db.Ticket.predicted_priority, func.count(db.Ticket.id))
        .filter(db.Ticket.predicted_priority.isnot(None))
        .group_by(db.Ticket.predicted_priority)
        .all()
    )

    total_fb = session.query(func.count(db.Feedback.id)).scalar() or 0
    accepted_fb = session.query(func.count(db.Feedback.id)).filter(db.Feedback.agent_accepted == "yes").scalar() or 0
    accept_rate = (accepted_fb / total_fb) if total_fb else None

    return StatsOut(
        total_tickets=total, by_category=by_category, by_priority=by_priority,
        feedback_accept_rate=accept_rate,
    )


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
