"""Абляция «одно изменение»: baseline и кандидат одной процедурой на одних и тех же входах → PASS/FAIL/INCONCLUSIVE.

Одна команда для L2 (один GPU → руки гоняются последовательно, каждая в своём процессе)::

    python -m eval.ablate --name H07 --candidate-env SENTINEL_FLAG_X=1 \\
        --videos data/raw --labels data/labels.csv --manifest data/manifest.csv --out runs/ablate/H07

Выход: <out>/baseline, <out>/candidate (preds, metrics, protocol), compare.json (локально, с ID видео),
compare_public.json (только агрегаты — можно в облако), строки в reports/runs.jsonl и reports/runs.html.

Правила вердикта (пороги — config.yaml → ablate):
* FAIL: FN вырос; FP вырос больше допуска (даже если recall вырос); failed/partial/timeouts выросли;
  p90 латентности > SLA или регресс > latency_regress_rel без выигрыша в качестве; VRAM > бюджета;
* PASS: не FAIL и есть измеримый выигрыш (FN↓, FP↓ при FN не хуже, hit@2 ≥ +min_hit2_gain, p90 ≥ 20% быстрее);
* INCONCLUSIVE: иначе. На 42 разведочных видео PASS — разведочный; включать по умолчанию после
  подтверждения (замороженный hold-out eval.extra) — решение команды.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .common import REPO_ROOT, load_config, read_json, write_json

CORRECT = {"TP", "TN", "early", "late", "wrong_type"}   # вердикт по видео верный (локализация/тип — отдельно)


def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Двусторонний точный тест Макнемара по несогласованным парам (b — исправлено, c — сломано)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


def _sc(m: dict) -> dict:
    return m.get("scorecard") or {}


def compare(base: dict, cand: dict, cfg: Optional[dict] = None, prot_base: Optional[dict] = None,
            prot_cand: Optional[dict] = None) -> dict:
    cfg = cfg or load_config()
    a_cfg = cfg.get("ablate") or {}
    sa, sb = _sc(base), _sc(cand)
    by_a = {e["video_id"]: e for e in base["per_video"]}
    by_b = {e["video_id"]: e for e in cand["per_video"]}
    common = sorted(set(by_a) & set(by_b))
    fixed, broken, changed = [], [], []
    for vid in common:
        ca, cb = by_a[vid]["category"], by_b[vid]["category"]
        if ca in ("label_conflict", "label_unknown"):
            continue
        ra, rb = ca in CORRECT, cb in CORRECT
        if not ra and rb:
            fixed.append(vid)
        elif ra and not rb:
            broken.append(vid)
        if ca != cb:
            changed.append({"video_id": vid, "baseline": ca, "candidate": cb})

    def d(key: str, src_a: dict = sa, src_b: dict = sb) -> Optional[float]:
        x, y = src_a.get(key), src_b.get(key)
        return None if x is None or y is None else round(y - x, 4)

    hit_a, hit_b = sa.get("hit_events_all") or {}, sb.get("hit_events_all") or {}
    lat_a, lat_b = (sa.get("latency_s") or {}), (sb.get("latency_s") or {})
    deltas = {
        "fn": d("fn"), "fp": d("fp"), "failed": d("failed"), "partial": d("partial"), "timeouts": d("timeouts"),
        "unresolved_neg": d("unresolved_neg"), "recall": d("recall"), "precision": d("precision"),
        "specificity": d("specificity"), "evidence_ok": d("evidence_ok_all"),
        "hit": {k: (None if hit_a.get(k) is None or hit_b.get(k) is None else round(hit_b[k] - hit_a[k], 4))
                for k in sorted(set(hit_a) | set(hit_b), key=float)},
        "latency_p90_s": d("p90", lat_a, lat_b), "latency_p50_s": d("median", lat_a, lat_b),
        "vram_peak_mb": d("vram_peak_mb_max"), "extra_candidates": d("extra_candidates_total"),
    }
    n = int(sa.get("n_videos") or 0)
    fp_tol = max(float(a_cfg.get("fp_tolerance_abs", 1)), float(a_cfg.get("fp_tolerance_rel", 0.05)) * n)
    reasons_fail, reasons_pass = [], []
    if (deltas["fn"] or 0) > 0:
        reasons_fail.append(f"FN вырос на {deltas['fn']:g}")
    if (deltas["fp"] or 0) > fp_tol:
        reasons_fail.append(f"FP вырос на {deltas['fp']:g} (> допуска {fp_tol:g})"
                            + (" — рост recall при росте FP не PASS" if (deltas["recall"] or 0) > 0 else ""))
    bad_runs = sum(max(0, deltas[k] or 0) for k in ("failed", "partial", "timeouts"))
    if bad_runs > 0:
        reasons_fail.append(f"failed/partial/timeouts выросли на {bad_runs:g}")
    p90b = lat_b.get("p90")
    if p90b is not None and p90b > float(a_cfg.get("sla_p90_s", 60.0)):
        reasons_fail.append(f"p90 латентности {p90b:g} с > SLA {a_cfg.get('sla_p90_s', 60.0)} с")
    vb = sb.get("vram_peak_mb_max")
    if vb is not None and vb > float(a_cfg.get("vram_budget_mb", 7500)):
        reasons_fail.append(f"пик VRAM {vb:g} МБ > бюджета {a_cfg.get('vram_budget_mb', 7500)} МБ")
    gain_quality = ((deltas["fn"] or 0) < 0 or ((deltas["fp"] or 0) < 0 and (deltas["fn"] or 0) <= 0)
                    or (deltas["hit"].get("2") or 0) >= float(a_cfg.get("min_hit2_gain", 0.05)))
    p90a = lat_a.get("p90")
    rel_lat = (p90b - p90a) / p90a if (p90a and p90b is not None) else None
    if rel_lat is not None and rel_lat > float(a_cfg.get("latency_regress_rel", 0.2)) and not gain_quality:
        reasons_fail.append(f"p90 латентности +{rel_lat:.0%} без выигрыша в качестве")
    if (deltas["fn"] or 0) < 0:
        reasons_pass.append(f"FN −{-deltas['fn']:g}")
    if (deltas["fp"] or 0) < 0 and (deltas["fn"] or 0) <= 0:
        reasons_pass.append(f"FP −{-deltas['fp']:g}")
    if (deltas["hit"].get("2") or 0) >= float(a_cfg.get("min_hit2_gain", 0.05)):
        reasons_pass.append(f"hit@2 +{deltas['hit']['2']:.2f}")
    if rel_lat is not None and rel_lat <= -0.2 and (deltas["fn"] or 0) <= 0 and (deltas["fp"] or 0) <= fp_tol:
        reasons_pass.append(f"p90 латентности {rel_lat:.0%}")
    verdict = "FAIL" if reasons_fail else ("PASS" if reasons_pass else "INCONCLUSIVE")
    same = []
    if prot_base and prot_cand:
        from .protocol import same_procedure

        same = same_procedure(prot_base, prot_cand)
        if same:
            verdict = "INVALID"
    env_a, env_b = (prot_base or {}).get("pipeline_env") or {}, (prot_cand or {}).get("pipeline_env") or {}
    diff_keys = sorted(k for k in set(env_a) | set(env_b) if env_a.get(k) != env_b.get(k))
    if (prot_base or {}).get("pipeline_config_sha") != (prot_cand or {}).get("pipeline_config_sha"):
        diff_keys.append("pipeline_config")
    warnings = []
    if len(diff_keys) > 1:
        warnings.append(f"изменено больше одного параметра: {diff_keys} — «одно изменение за раз»")
    if not diff_keys and prot_base and prot_cand:
        warnings.append("руки не отличаются ни флагами, ни конфигом пайплайна")
    return {
        "verdict": verdict, "reasons_fail": reasons_fail, "reasons_pass": reasons_pass, "invalid": same,
        "warnings": warnings, "changed_params": diff_keys,
        "n_videos": n, "evaluation_kind": sa.get("evaluation_kind"),
        "paired": {"n_common": len(common), "fixed": len(fixed), "broken": len(broken),
                   "mcnemar_p": mcnemar_exact(len(fixed), len(broken)),
                   "fixed_ids": fixed, "broken_ids": broken, "category_changes": changed},
        "deltas": deltas, "fp_tolerance": fp_tol,
        "baseline": {k: sa.get(k) for k in ("fn", "fp", "recall", "precision", "failed", "partial", "timeouts",
                                            "hit_events_all", "evidence_ok_all", "latency_s", "vram_peak_mb_max")},
        "candidate": {k: sb.get(k) for k in ("fn", "fp", "recall", "precision", "failed", "partial", "timeouts",
                                             "hit_events_all", "evidence_ok_all", "latency_s", "vram_peak_mb_max")},
        "note": "Разведочная оценка на одних и тех же видео; p-value Макнемара при n≈40 — ориентир, не доказательство.",
    }


def public(cmp: dict) -> dict:
    """Без ID видео — только агрегаты."""
    out = {k: v for k, v in cmp.items() if k != "paired"}
    p = cmp["paired"]
    out["paired"] = {k: p[k] for k in ("n_common", "fixed", "broken", "mcnemar_p")}
    return out


def format_compare(c: dict) -> str:
    dl = c["deltas"]
    lines = [f"АБЛЯЦИЯ: {c['verdict']}  (n={c['n_videos']}, {c['evaluation_kind']}; изменено: {c['changed_params'] or '—'})",
             f"  baseline : FN={c['baseline']['fn']} FP={c['baseline']['fp']} hit={c['baseline']['hit_events_all']} "
             f"p90={(c['baseline']['latency_s'] or {}).get('p90')} VRAM={c['baseline']['vram_peak_mb_max']}",
             f"  candidate: FN={c['candidate']['fn']} FP={c['candidate']['fp']} hit={c['candidate']['hit_events_all']} "
             f"p90={(c['candidate']['latency_s'] or {}).get('p90')} VRAM={c['candidate']['vram_peak_mb_max']}",
             f"  Δ: FN={dl['fn']} FP={dl['fp']} failed={dl['failed']} partial={dl['partial']} recall={dl['recall']} "
             f"precision={dl['precision']} hit={dl['hit']} p90={dl['latency_p90_s']} с VRAM={dl['vram_peak_mb']} МБ",
             f"  по видео: исправлено {c['paired']['fixed']}, сломано {c['paired']['broken']}, "
             f"p(Макнемар)={c['paired']['mcnemar_p']}"]
    lines += [f"  FAIL: {r}" for r in c["reasons_fail"]] + [f"  +: {r}" for r in c["reasons_pass"]]
    lines += [f"  ⚠ {w}" for w in c["warnings"]] + [f"  INVALID: {x}" for x in c["invalid"]]
    return "\n".join(lines)


def _arm_args(args: argparse.Namespace, arm: str, out: Path) -> list[str]:
    a = ["--videos", args.videos, "--labels", args.labels, "--types", args.types, "--out", str(out),
         "--role", arm, "--label", f"{args.name}:{arm}", "--log", args.log, "--ablation-id", args.name]
    for k in ("subset", "splits", "only", "config", "api", "key"):
        v = getattr(args, k, None)
        if v:
            a += [f"--{k}", str(v)]
    if args.final:
        a.append("--final")
    if args.mode:
        a += ["--mode", args.mode]
    if args.raw:
        a.append("--raw")
    pc = getattr(args, f"{arm}_config")
    if pc:
        a += ["--pipeline-config", pc]
    for e in getattr(args, f"{arm}_env") or []:
        a += ["--env", e]
    return a


def run_arm(args: argparse.Namespace, arm: str, out: Path) -> int:
    argv = _arm_args(args, arm, out)
    if args.inproc_arms:  # тесты / отладка: тот же процесс (флаги пайплайн должен читать при вызове run)
        import os

        from . import run as runmod

        saved = dict(os.environ)
        try:
            return runmod.main(argv)
        finally:  # флаги одной руки не должны протечь в другую
            os.environ.clear()
            os.environ.update(saved)
    return subprocess.run([sys.executable, "-m", "eval.run"] + argv, cwd=str(REPO_ROOT), check=False).returncode


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Абляция baseline vs candidate (одно изменение)")
    ap.add_argument("--name", required=True, help="ID гипотезы, напр. H07-pooling-top3")
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--types", default="data/labels_types.csv")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline-env", action="append", default=[])
    ap.add_argument("--candidate-env", action="append", default=[])
    ap.add_argument("--baseline-config", default=None)
    ap.add_argument("--candidate-config", default=None)
    ap.add_argument("--reuse-baseline", default=None, help="готовая папка baseline той же процедуры")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--api", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--no-raw", dest="raw", action="store_false", help="не сохранять scores_raw (санити без скоров)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--log", default="reports/runs.jsonl")
    ap.add_argument("--inproc-arms", action="store_true", help="руки в этом же процессе (тесты)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out or f"runs/ablate/{args.name}")
    base_dir = Path(args.reuse_baseline) if args.reuse_baseline else out / "baseline"
    if not args.reuse_baseline:
        rc = run_arm(args, "baseline", base_dir)
        if rc != 0:
            print(f"baseline: код {rc}")
            return rc
    rc = run_arm(args, "candidate", out / "candidate")
    if rc != 0:
        print(f"candidate: код {rc}")
        return rc
    mb, mc = read_json(base_dir / "metrics.json"), read_json(out / "candidate" / "metrics.json")
    pb = read_json(base_dir / "protocol.json") if (base_dir / "protocol.json").exists() else None
    pc = read_json(out / "candidate" / "protocol.json") if (out / "candidate" / "protocol.json").exists() else None
    cmp = compare(mb, mc, cfg, pb, pc)
    try:  # санити по обеим рукам: AUC скоров и корреляция с длительностью
        from .labels import load_ground_truth
        from .metrics import load_predictions
        from .sanity import load_durations, run_scores, sanity

        preds = load_predictions(base_dir / "preds")[0]
        sc = {f"baseline/{k}": v for k, v in run_scores(base_dir).items()}
        sc.update({f"candidate/{k}": v for k, v in run_scores(out / "candidate").items()})
        cmp["sanity"] = sanity(load_ground_truth(args.labels), load_durations(args.manifest, None, preds), sc, cfg)
    except Exception as e:  # noqa: BLE001
        cmp["sanity_error"] = repr(e)[:300]
    write_json(out / "compare.json", cmp)
    write_json(out / "compare_public.json", public(cmp))
    from .runs import append, load, render_html

    append(args.log, {"role": "ablation", "ablation_id": args.name, "label": args.name, "verdict": cmp["verdict"],
                      "deltas": cmp["deltas"], "paired": public(cmp)["paired"], "changed_params": cmp["changed_params"]})
    html_path = Path(args.log).with_name("runs.html")
    html_path.write_text(render_html(load(args.log)), encoding="utf-8")
    print(format_compare(cmp))
    if cmp.get("sanity"):
        from .sanity import format_sanity

        print(format_sanity(cmp["sanity"]))
    print(f"→ {out}/compare.json (локально), compare_public.json (агрегаты), {html_path}")
    return 0 if cmp["verdict"] in ("PASS", "INCONCLUSIVE", "FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
