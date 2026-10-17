"""ml.fusion (A12) — слияние событий, вердикт, кадры-доказательства, описания.

Публичная функция (сигнатура заморожена контрактом v1)::

    fuse(clip, events, risk, catalog, job_id) -> AnalysisResult
"""
from __future__ import annotations

from .core import FUSION_VERSION, fuse, fuse_scores

__all__ = ["FUSION_VERSION", "fuse", "fuse_scores"]
