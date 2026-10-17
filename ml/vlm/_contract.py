"""Единая точка импорта замороженного контракта для ml.vlm."""
from __future__ import annotations

try:
    from contracts.sentinel_contracts import *  # noqa: F401,F403
    from contracts.sentinel_contracts import BBox, Event, LABELS, SIGNALS, VideoClip  # noqa: F401
except ImportError:  # пока ветка A01 не смерджена
    from ._contract_fallback import *  # noqa: F401,F403
    from ._contract_fallback import BBox, Event, LABELS, SIGNALS, VideoClip  # noqa: F401
