"""Генерация contracts/examples/*.json из сериализаторов (воспроизводимо; тест сверяет файлы байт в байт).

    python -m contracts.examples.build          # перезаписать примеры
"""
from __future__ import annotations

import json
from pathlib import Path

from contracts.extensions import compute_coverage
from contracts.sentinel_contracts import AnalysisResult, EvidenceFrame, RuleRef, Violation
from contracts.serialize import result_to_jury_api, result_to_ui_report

HERE = Path(__file__).resolve().parent
# Крошечный валидный JPEG (синтетика 16x24, не кадр датасета).
TINY_JPEG_B64 = json.loads((HERE / "job_done_tz.json").read_text(encoding="utf-8"))["violations"][0][
    "evidence_frames"][0]["image_base64"]
DESCRIPTION = ("На 22.0 с в зоне за кассовым столом находится второй человек, не являющийся кассиром. Признаки "
               "возможного нарушения: присутствие постороннего лица в кассовом помещении.")
# Заглушка формы нормы — НЕ норма: реальные ссылки только из norms/catalog.yaml (A13).
RULE = RuleRef(document="<название документа из norms/catalog.yaml>", clause="<пункт из norms/catalog.yaml>",
               url="https://adilet.zan.kz/rus/docs/<id из norms/catalog.yaml>",
               quote="<дословная цитата из norms/catalog.yaml>", verified=False)
STAGES = {"ingest": {"status": "ok"}, "detect": {"status": "ok"}, "track": {"status": "ok"},
          "scene": {"status": "ok"}, "rules": {"status": "ok"}, "vlm_verify": {"status": "disabled"},
          "fusion": {"status": "ok"}}
REQUIRED = ["ingest", "detect", "track", "scene", "rules", "fusion"]


def build_examples() -> dict[str, dict]:
    cov = compute_coverage(48.3, 48.3, stages=STAGES, required=REQUIRED)
    meta = {"pipeline_version": "0.1.0", "duration_sec": 48.3, "progress": 1.0, "stage": "done", "coverage": cov,
            "stages": STAGES, "required_stages": REQUIRED, "complete": True, "mock": False, "warnings": [],
            "timings": {"ingest": 1.2, "detect": 9.8, "total": 31.5},
            "violation_extra": [{"provenance": {"sources": ["rules", "vlm"]}}]}
    v = Violation(type="Посторонний в кассовом помещении", timecode_sec=22.0, description=DESCRIPTION,
                  evidence_frames=[EvidenceFrame(22.0, TINY_JPEG_B64), EvidenceFrame(23.4, TINY_JPEG_B64)],
                  type_id="V01", confidence=0.87, t_start=21.2, t_end=26.8, rules=[RULE])
    full = AnalysisResult(job_id="3f2b6c1e-0000-4000-8000-000000000002", status="done", verdict="violation",
                          violations=[v], timeline=[(0.0, 0.02), (21.2, 0.61), (22.0, 0.93), (26.8, 0.4)], meta=meta)
    proc = AnalysisResult(job_id="3f2b6c1e-0000-4000-8000-000000000003", status="processing",
                          meta={"progress": 0.35, "stage": "detect"})
    clean = AnalysisResult(job_id="3f2b6c1e-0000-4000-8000-000000000004", status="done", verdict="no_violation",
                           meta={"coverage": cov, "duration_sec": 48.3})
    unverified = AnalysisResult(job_id="3f2b6c1e-0000-4000-8000-000000000006", status="done",
                                verdict="no_violation", meta={"duration_sec": 48.3})
    failed = AnalysisResult(job_id="3f2b6c1e-0000-4000-8000-000000000005", status="failed",
                            meta={"error": "не удалось декодировать видео"})
    return {
        "job_done_full": result_to_ui_report(full, allow_mock=False),
        "jury_done_full": result_to_jury_api(full, allow_mock=False),
        "job_processing": result_to_jury_api(proc, allow_mock=False),
        "ui_processing": result_to_ui_report(proc, allow_mock=False),
        "job_no_violation": result_to_jury_api(clean, allow_mock=False),
        "jury_unverified_no_violation": result_to_jury_api(unverified, allow_mock=False),
        "job_failed": result_to_jury_api(failed, allow_mock=False),
        "ui_failed": result_to_ui_report(failed, allow_mock=False),
    }


def render(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    for name, obj in build_examples().items():
        (HERE / f"{name}.json").write_text(render(obj), encoding="utf-8")
        print(f"{name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
