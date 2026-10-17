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
    assert check_fallbacks.main(["--root", str(tmp_path), "--drop"]) == 0
    assert not list(tmp_path.rglob("_contract_fallback.py"))
