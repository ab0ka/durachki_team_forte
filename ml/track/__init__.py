"""ml.track (A06): трекинг людей и предметов, роли без лиц, авто-зоны без хардкода координат.

Публичное API (контракт v1):
    track(clip, dets) -> tuple[list[Track], list[Zone]]
Дополнительно:
    track_full(clip, dets, cfg=None) -> TrackResult(tracks, zones, info)   # + provenance
    events: entries_exits, dwell, proximity, gaps, presence_state, person_signals
"""
from __future__ import annotations

from .api import TrackResult, track, track_full
from .events import dwell, entries_exits, gaps, person_signals, presence_state, proximity

__all__ = ["track", "track_full", "TrackResult", "entries_exits", "dwell", "proximity", "gaps",
           "presence_state", "person_signals"]
