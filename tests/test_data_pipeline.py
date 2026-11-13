"""Smoke tests for the Phase 1 data pipeline."""
import pandas as pd
import pytest

from src.data.clean import sanitize_text, clean


def test_sanitize_text_strips_html_and_pii():
    raw = "<p>Contact me at john@example.com or 555-123-4567. Visit https://example.com</p>"
    cleaned = sanitize_text(raw)
    assert "<p>" not in cleaned
    assert "[EMAIL]" in cleaned
    assert "[URL]" in cleaned


def test_clean_drops_empty_body_and_dedupes():
    df = pd.DataFrame(
        {
            "ticket_subject": ["Issue A", "Issue B", "Issue C"],
            "description": ["VPN not connecting", "", "VPN not connecting"],
            "resolution_notes": ["Restart client", "N/A", "Restart client"],
            "category": ["IT Support", "IT Support", "IT Support"],
            "priority": ["high", "low", "high"],
            "ticket_type": ["Incident", "Incident", "Incident"],
            "language": ["en", "en", "en"],
            "ticket_id": [1, 2, 3],
            "tags": [[], [], []],
        }
    )
    out = clean(df)
    assert len(out) == 1
    assert out.iloc[0]["description"] == "VPN not connecting"
