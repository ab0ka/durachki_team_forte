"""Контракт v1 заморожен: любой байт изменения должен ронять этот тест."""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import contracts
from contracts import sentinel_contracts as sc

CONTRACT_SHA256 = "3b870961e62dac86e9457029009a37a753c0fdc73543639690b0493d8e134cac"


def test_contract_file_is_byte_identical():
    data = Path(sc.__file__).read_bytes().replace(b"\r\n", b"\n")  # Windows autocrlf
    assert hashlib.sha256(data).hexdigest() == CONTRACT_SHA256, (
        "contracts/sentinel_contracts.py изменён! Контракт v1 заморожен — расширения только через "
        "extra/meta/signals и предложения в README."
    )


def test_labels_and_signals():
    assert len(sc.LABELS) == len(set(sc.LABELS)) == 16
    assert len(sc.SIGNALS) == len(set(sc.SIGNALS)) == 17
    assert "person" in sc.LABELS and "cash_bag" in sc.LABELS
    assert "workplace_unattended_with_cash" in sc.SIGNALS


def test_reexport_matches_module():
    for name in contracts.__all__:
        if name == "CONTRACT_VERSION":
            continue
        assert getattr(contracts, name) is getattr(sc, name)


def test_star_import_exposes_contract_names():
    ns: dict = {}
    exec("from contracts.sentinel_contracts import *", ns)  # так импортируют все модули
    for name in ("Detection", "Track", "Zone", "PoseFrame", "SceneState", "Event", "EvidenceFrame",
                 "RuleRef", "Violation", "AnalysisResult", "LABELS", "SIGNALS", "VideoClip"):
        assert name in ns


def test_dataclass_defaults_are_independent():
    a = sc.Event(type_id="V01", t_start=0, t_end=1, t_peak=0.5, confidence=0.9, source="rules")
    b = sc.Event(type_id="V02", t_start=0, t_end=1, t_peak=0.5, confidence=0.9, source="rules")
    a.facts.append("x")
    a.extra["k"] = 1
    assert b.facts == [] and b.extra == {}
    r = sc.AnalysisResult(job_id="j", status="processing")
    assert r.verdict is None and r.violations == [] and r.timeline == [] and r.meta == {}
    t = sc.Track(track_id=1, label="person")
    assert t.role == "unknown" and t.boxes == []
    assert dataclasses.is_dataclass(sc.Violation)
