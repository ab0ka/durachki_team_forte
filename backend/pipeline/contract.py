"""Единая точка импорта замороженного контракта для оркестратора.

Пока ветка A01 не смерджена, используется локальная копия ``_contract_fallback``.
Интегратор удалит fallback-копию — этот файл продолжит работать без изменений.
"""
from __future__ import annotations

try:
    from contracts.sentinel_contracts import *  # noqa: F401,F403
    from contracts.sentinel_contracts import (  # noqa: F401  (явно, для линтеров/IDE)
        AnalysisResult, BBox, Detection, Event, EvidenceFrame, LABELS, PoseFrame, RuleRef,
        SceneState, SIGNALS, Track, Violation, Zone,
    )
    CONTRACT_SOURCE = "contracts.sentinel_contracts"
except ImportError:  # пока ветка A01 не смерджена
    from ._contract_fallback import *  # noqa: F401,F403
    from ._contract_fallback import (  # noqa: F401
        AnalysisResult, BBox, Detection, Event, EvidenceFrame, LABELS, PoseFrame, RuleRef,
        SceneState, SIGNALS, Track, Violation, Zone,
    )
    CONTRACT_SOURCE = "backend.pipeline._contract_fallback"
