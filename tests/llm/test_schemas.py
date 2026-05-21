from __future__ import annotations

import json

import pytest

from aiam.llm.schemas import AssetView, ParseError, ViewSet, parse_viewset


def _valid_payload(**overrides) -> dict:
    base = {
        "views": [
            {"asset": "SPY", "expected_excess_return": 0.08, "confidence": 0.7},
            {"asset": "IEF", "expected_excess_return": -0.03, "confidence": 0.5},
        ]
    }
    base.update(overrides)
    return base


# ── Valid cases ──────────────────────────────────────────────────────────────

def test_valid_viewset_parses():
    vs = ViewSet.model_validate(_valid_payload())
    assert len(vs.views) == 2
    assert vs.views[0].asset == "SPY"
    assert vs.rationale is None


def test_valid_viewset_with_rationale():
    payload = _valid_payload()
    payload["rationale"] = "Momentum signal positive"
    vs = ViewSet.model_validate(payload)
    assert vs.rationale == "Momentum signal positive"


def test_parse_viewset_plain_json():
    raw = json.dumps(_valid_payload())
    vs = parse_viewset(raw)
    assert len(vs.views) == 2


def test_parse_viewset_strips_markdown_fences():
    raw = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    vs = parse_viewset(raw)
    assert len(vs.views) == 2


def test_parse_viewset_strips_plain_fences():
    raw = "```\n" + json.dumps(_valid_payload()) + "\n```"
    vs = parse_viewset(raw)
    assert len(vs.views) == 2


def test_zero_confidence_accepted():
    payload = {"views": [{"asset": "A", "expected_excess_return": 0.0, "confidence": 0.0}]}
    vs = ViewSet.model_validate(payload)
    assert vs.views[0].confidence == 0.0


def test_one_confidence_accepted():
    payload = {"views": [{"asset": "A", "expected_excess_return": 0.05, "confidence": 1.0}]}
    vs = ViewSet.model_validate(payload)
    assert vs.views[0].confidence == 1.0


def test_boundary_return_accepted():
    payload = {"views": [{"asset": "A", "expected_excess_return": 1.0, "confidence": 0.5}]}
    vs = ViewSet.model_validate(payload)
    assert vs.views[0].expected_excess_return == 1.0


# ── Rejection cases ──────────────────────────────────────────────────────────

def test_rejects_confidence_above_one():
    payload = {"views": [{"asset": "SPY", "expected_excess_return": 0.1, "confidence": 1.1}]}
    with pytest.raises(Exception):
        ViewSet.model_validate(payload)


def test_rejects_confidence_below_zero():
    payload = {"views": [{"asset": "SPY", "expected_excess_return": 0.1, "confidence": -0.1}]}
    with pytest.raises(Exception):
        ViewSet.model_validate(payload)


def test_rejects_return_above_one():
    payload = {"views": [{"asset": "SPY", "expected_excess_return": 1.5, "confidence": 0.5}]}
    with pytest.raises(Exception):
        ViewSet.model_validate(payload)


def test_rejects_return_below_neg_one():
    payload = {"views": [{"asset": "SPY", "expected_excess_return": -5.0, "confidence": 0.5}]}
    with pytest.raises(Exception):
        ViewSet.model_validate(payload)


def test_rejects_duplicate_assets():
    payload = {
        "views": [
            {"asset": "SPY", "expected_excess_return": 0.1, "confidence": 0.5},
            {"asset": "SPY", "expected_excess_return": -0.05, "confidence": 0.3},
        ]
    }
    with pytest.raises(Exception):
        ViewSet.model_validate(payload)


def test_rejects_empty_asset_name():
    payload = {"views": [{"asset": "", "expected_excess_return": 0.1, "confidence": 0.5}]}
    with pytest.raises(Exception):
        ViewSet.model_validate(payload)


def test_rejects_whitespace_asset_name():
    payload = {"views": [{"asset": "   ", "expected_excess_return": 0.1, "confidence": 0.5}]}
    with pytest.raises(Exception):
        ViewSet.model_validate(payload)


def test_parse_viewset_raises_on_malformed_json():
    with pytest.raises(ParseError):
        parse_viewset("not json at all {{{")


def test_parse_viewset_raises_on_wrong_schema():
    with pytest.raises(ParseError):
        parse_viewset(json.dumps({"wrong_key": []}))


def test_parse_viewset_raises_on_empty_string():
    with pytest.raises(ParseError):
        parse_viewset("")
