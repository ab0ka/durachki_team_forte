from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import check_licenses  # noqa: E402
import check_ownership  # noqa: E402
import ci_requirements  # noqa: E402
from ownership import agent_from_title, forbidden_reason, is_shared, owner_of  # noqa: E402


@pytest.mark.parametrize("path,owner", [
    ("contracts/serialize.py", "A01"), ("README.md", "A01"), (".github/workflows/ci.yml", "A01"),
    ("backend/api/main.py", "A02"), ("backend/pipeline/run.py", "A03"), ("ml/detect/x.py", "A05"),
    ("ml/fusion/fuse.py", "A12"), ("norms/catalog.yaml", "A13"), ("frontend/src/App.tsx", "A15"),
    ("docs/arch.md", "A18"), ("pitch/deck.md", "A18"), ("./testkit/fixtures.py", "A01"),
    ("random/file.py", None), ("ml/other/x.py", None),
])
def test_owner_of(path, owner):
    assert owner_of(path) == owner


def test_shared_and_forbidden():
    assert is_shared("ml/__init__.py") and is_shared("backend/__init__.py")
    assert not is_shared("ml/detect/__init__.py")
    assert forbidden_reason("data/X-1.MOV") and forbidden_reason("ml/detect/w.onnx")
    assert forbidden_reason("eval/frames/f.jpg") and not forbidden_reason("docs/assets/arch.jpg")
    assert forbidden_reason("ml/x.py") is None


def test_agent_from_title():
    assert agent_from_title("[A05] Детекция") == "A05"
    assert agent_from_title("[a12] fusion") == "A12"
    assert agent_from_title("Integration") is None


def test_analyze_explicit_and_inferred():
    files = ["ml/detect/a.py", "ml/detect/tests/test_a.py", "ml/__init__.py", "ml/rules/x.py"]
    res = check_ownership.analyze(files, "A05")
    assert res["foreign"] == [("ml/rules/x.py", "A09")] and res["explicit"]
    assert check_ownership.report("t", res, "list") == 1
    inferred = check_ownership.analyze(files, None)
    assert inferred["agent"] == "A05" and check_ownership.report("t", inferred, "list") == 0
    assert check_ownership.main(["--files", "ml/detect/a.py", "--agent", "A05"]) == 0
    assert check_ownership.main(["--files", "data/v.mov"]) == 1


def test_ci_requirements_filter(tmp_path):
    lines = ["torch==2.3.1", "torchvision>=0.18", "opencv-python==4.10.0.84", "numpy>=1.26  # ok",
             "--extra-index-url https://download.pytorch.org/whl/cu121", "nvidia-cudnn-cu12", "transformers==4.44.2",
             "onnxruntime-gpu", "onnxruntime==1.19.0", "Pillow", "ultralytics", "-r base.txt", ""]
    assert ci_requirements.filter_lines(lines) == ["opencv-python-headless==4.10.0.84", "numpy>=1.26",
                                                   "transformers==4.44.2", "onnxruntime==1.19.0", "Pillow"]
    mod = tmp_path / "ml" / "x"
    mod.mkdir(parents=True)
    (mod / "requirements.txt").write_text("torch\nscipy>=1.11\n", encoding="utf-8")
    assert ci_requirements.module_requirements("ml/x", tmp_path) == ["scipy>=1.11"]
    (mod / "requirements-ci.txt").write_text("# ci\nscipy\n", encoding="utf-8")
    assert ci_requirements.module_requirements("ml/x", tmp_path) == ["scipy"]


def test_check_licenses_on_tree(tmp_path):
    good = tmp_path / "ml" / "detect"
    good.mkdir(parents=True)
    (good / "models.json").write_text(json.dumps([
        {"name": "RT-DETRv2", "hf_id_or_url": "PekingU/rtdetr_v2_r18vd", "license": "Apache-2.0"},
        {"name": "YOLOX-s", "hf_id_or_url": "https://github.com/Megvii-BaseDetection/YOLOX", "license": "Apache-2.0"},
    ]), encoding="utf-8")
    (good / "requirements.txt").write_text("onnxruntime\nopenai\n", encoding="utf-8")
    assert check_licenses.main(["--root", str(tmp_path), "--no-git"]) == 0
    bad = tmp_path / "ml" / "vlm"
    bad.mkdir(parents=True)
    (bad / "models.json").write_text(json.dumps([
        {"name": "Qwen2.5-VL-3B", "hf_id_or_url": "Qwen/Qwen2.5-VL-3B-Instruct", "license": "qwen-research"},
        {"name": "YOLOv8n", "hf_id_or_url": "ultralytics/yolov8", "license": "AGPL-3.0"},
        {"name": "Some", "hf_id_or_url": "x/y", "license": "CC-BY-NC-4.0"},
        {"name": "Qwen2.5-VL-7B", "hf_id_or_url": "Qwen/Qwen2.5-VL-7B-Instruct", "license": "Apache-2.0"},
    ]), encoding="utf-8")
    (bad / "requirements.txt").write_text("ultralytics==8.2\nanthropic\n", encoding="utf-8")
    (bad / "client.py").write_text('URL = "https://api.openai.com/v1/chat"\n', encoding="utf-8")
    errs = check_licenses.check_models(tmp_path)
    assert len(errs) == 3 and not any("7B" in e for e in errs)
    req_errs, warns = check_licenses.check_requirements(tmp_path)
    assert len(req_errs) == 2 and len(warns) == 1
    assert len(check_licenses.check_code(tmp_path)) == 1
    assert check_licenses.main(["--root", str(tmp_path), "--no-git"]) == 1


@pytest.mark.parametrize("lic,ok", [
    ("Apache-2.0", True), ("apache-2.0 (наследует базовый чекпойнт; датасет — внутренний)", True),
    ("Apache-2.0 (собственный код/веса команды)", True), ("MIT / BSD-3-Clause", True), ("BSD", True),
    ("AGPL-3.0", False), ("CC-BY-NC-4.0", False), ("qwen-research", False), ("", False),
    ("Apache-2.0 / CC-BY-NC-4.0", False), ("llama3", False),
])
def test_license_ok(lic, ok):
    assert check_licenses.license_ok(lic) is ok


def test_repo_itself_is_clean():
    root = SCRIPTS.parent
    rc = subprocess.run([sys.executable, str(SCRIPTS / "check_licenses.py"), "--root", str(root)],
                        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr


def test_check_fallbacks(tmp_path):
    import check_fallbacks

    (tmp_path / "contracts").mkdir()
    canon = (SCRIPTS.parent / "contracts" / "sentinel_contracts.py").read_text(encoding="utf-8")
    (tmp_path / "contracts" / "sentinel_contracts.py").write_text(canon, encoding="utf-8")
    for mod, text in {"ml/a": canon, "ml/b": "# копия\n" + canon,
                      "ml/c": canon.replace("    source: str = \"\"\n", "    source: str = \"\"\n    extra: int = 0\n")}.items():
        (tmp_path / mod).mkdir(parents=True)
        (tmp_path / mod / "_contract_fallback.py").write_text(text, encoding="utf-8")
    assert check_fallbacks.compare(tmp_path / "ml/a/_contract_fallback.py", canon)[0] == "identical"
    assert check_fallbacks.compare(tmp_path / "ml/b/_contract_fallback.py", canon)[0] == "same-structure"
    assert check_fallbacks.compare(tmp_path / "ml/c/_contract_fallback.py", canon)[0] == "DIVERGED"
    assert check_fallbacks.main(["--root", str(tmp_path)]) == 1
    (tmp_path / "ml/c/_contract_fallback.py").unlink()
    (tmp_path / "ml/b/tests").mkdir()
    (tmp_path / "ml/b/tests/test_x.py").write_text("from ml.b._contract_fallback import LABELS\n", encoding="utf-8")
    assert check_fallbacks.unguarded_imports(tmp_path) == {"ml/b": ["ml/b/tests/test_x.py:1"]}
    assert check_fallbacks.main(["--root", str(tmp_path), "--drop"]) == 0
    assert [p.parent.name for p in tmp_path.rglob("_contract_fallback.py")] == ["b"]  # с прямым импортом — оставлен


def test_guard_media(tmp_path):
    import guard_media

    assert guard_media.name_problem("data/X-1.MOV") and guard_media.name_problem("ml/detect/w.safetensors")
    assert guard_media.name_problem("eval/frames/f.jpg") and not guard_media.name_problem("docs/assets/a.jpg")
    assert guard_media.name_problem("ml/detect/code.py") is None
    (tmp_path / "ok.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "big.bin").write_bytes(b"0" * (6 * 1024 * 1024))
    (tmp_path / "video.dat").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 32)
    (tmp_path / "frame.dat").write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 32)
    (tmp_path / "w.bin").write_bytes(b"GGUF" + b"0" * 32)
    assert guard_media.check_file("ok.py", tmp_path) == []
    assert any("МБ" in p for p in guard_media.check_file("big.bin", tmp_path))
    assert guard_media.check_file("video.dat", tmp_path) and guard_media.check_file("w.bin", tmp_path)
    assert guard_media.check_file("frame.dat", tmp_path)
    assert guard_media.main(["--root", str(tmp_path), "ok.py"]) == 0
    assert guard_media.main(["--root", str(tmp_path), "ok.py", "video.dat"]) == 1
    assert guard_media.main(["--root", str(tmp_path), "--max-mb", "10", "big.bin"]) == 0


def test_gitignore_is_anchored():
    lines = [ln.strip() for ln in (SCRIPTS.parent / ".gitignore").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    for d in ("/data/", "/weights/", "/cache/", "/reports/frames/"):
        assert d in lines
    for bad in ("data/", "weights/", "cache/", "outputs/", "runs/"):
        assert bad not in lines, f"неякорный шаблон {bad} спрячет папки внутри кода"
    for ext in ("*.mov", "*.MOV", "*.mp4", "*.avi", "*.mkv", "*.onnx", "*.pt", "*.pth", "*.safetensors", "*.gguf",
                "*.engine"):
        assert ext in lines


def test_repo_tracked_files_pass_guard():
    assert subprocess.run([sys.executable, str(SCRIPTS / "guard_media.py"), "--all", "--root", str(SCRIPTS.parent)],
                          capture_output=True).returncode == 0


def test_compat_matrix_compare_and_scan(tmp_path):
    import compat_matrix as cm

    ok = cm.compare("A06", "ml.track", "track", ["clip", "dets"], {}, True,
                    {"ok": True, "params": [["clip", "POSITIONAL_OR_KEYWORD", None],
                                            ["dets", "POSITIONAL_OR_KEYWORD", None]], "qualname": "ml.track.track"})
    assert ok.verdict == "OK"
    renamed = cm.compare("A06", "ml.track", "track", ["clip", "dets"], {}, True,
                         {"ok": True, "params": [["c", "POSITIONAL_OR_KEYWORD", None],
                                                 ["d", "POSITIONAL_OR_KEYWORD", None]]})
    assert renamed.verdict == "ЗАМЕЧАНИЕ"
    short = cm.compare("A06", "ml.track", "track", ["clip", "dets"], {}, True,
                       {"ok": True, "params": [["clip", "POSITIONAL_OR_KEYWORD", None],
                                               ["zones", "KEYWORD_ONLY", None]]})
    assert short.verdict == "РАСХОЖДЕНИЕ"
    missing = cm.compare("A06", "ml.track", "track", ["clip", "dets"], {}, True, {"ok": False, "error": "нет"})
    assert missing.verdict == "НЕТ"
    pkg = tmp_path / "ml" / "x"
    (pkg / "tests").mkdir(parents=True)
    (pkg / "tests" / "test_a.py").write_text("from ml.x._contract_fallback import LABELS\n", encoding="utf-8")
    (pkg / "core.py").write_text(
        "try:\n    from contracts.sentinel_contracts import *\nexcept ImportError:\n"
        "    from ._contract_fallback import *\n"
        "class Event:\n    pass\n"
        "v = signals.get('bogus_signal', 0)\nw = sig['safe_open']\nmeta['coverage'] = 1.0\n", encoding="utf-8")
    scan = cm.static_scan(tmp_path)
    assert scan["unguarded_fallback"] == ["ml/x/tests/test_a.py:1"]
    assert scan["duplicate_classes"] == ["ml/x/core.py:5 class Event"]
    assert scan["unknown_signals"] == ["ml/x/core.py:7 `bogus_signal`"]
    assert scan["unknown_as_zero"] == ["ml/x/core.py:7 `bogus_signal`"]
    assert scan["coverage_producers"] == ["ml/x/core.py:9"]
    probe = cm.probe(tmp_path, "json", "dumps")
    assert probe["ok"] and probe["params"][0][0] == "obj"


def test_resolve_deps(tmp_path):
    import resolve_deps as rd

    lock = rd.parse_lock("# lock\ntorch==2.11.0+cu128\nopencv-python-headless==4.14.0.94\nPyYAML==6.0.3\n"
                         "transformers==5.20.1\n--extra-index-url x\n")
    assert lock == {"torch": "2.11.0+cu128", "opencv-python-headless": "4.14.0.94", "pyyaml": "6.0.3",
                    "transformers": "5.20.1"}
    for mod, text in {"ml/a": "torch>=2.3\nPyYAML>=6,<7  # ok\nopencv-python==4.10.0.84\n",
                      "ml/b": "transformers>=4.57,<5\nfastapi>=0.110\nultralytics\n-r other.txt\n",
                      "eval": "pyyaml==6.0.1\n"}.items():
        (tmp_path / mod).mkdir(parents=True)
        (tmp_path / mod / "requirements.txt").write_text(text, encoding="utf-8")
    pkgs, skipped = rd.collect(tmp_path)
    rd.resolve(pkgs, lock)
    st = {k: p.status for k, p in pkgs.items()}
    assert st == {"torch": "OK", "pyyaml": "КОНФЛИКТ", "opencv-python": "ДУБЛЬ", "transformers": "КОНФЛИКТ",
                  "fastapi": "НЕТ В LOCK", "ultralytics": "ЗАПРЕЩЁН"}
    assert "eval: ==6.0.1" in pkgs["pyyaml"].detail and "ml/b" in pkgs["transformers"].detail
    out = tmp_path / "resolved.txt"
    rd.write_resolved(pkgs, out)
    text = out.read_text(encoding="utf-8")
    assert "torch==2.11.0+cu128" in text and "# РЕШИТЬ fastapi" in text and "fastapi>=" not in text
    lock_file = tmp_path / "l.txt"
    lock_file.write_text("torch==2.11.0+cu128\n", encoding="utf-8")
    assert rd.main(["--tree", str(tmp_path), "--lock", str(lock_file), "--out", str(tmp_path / "r.md")]) == 0
    assert rd.main(["--tree", str(tmp_path), "--lock", str(lock_file), "--out", str(tmp_path / "r.md"),
                    "--strict"]) == 1
