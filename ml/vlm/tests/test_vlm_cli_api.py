import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import ml.vlm as vlm
from ml.vlm._contract import Event
from ml.vlm.backends.base import GenResult, VLMBackend
from ml.vlm.backends.mock import MockBackend

REPO = Path(__file__).resolve().parents[3]


def _write_video(path: Path, seconds: float = 6.0, fps: int = 30):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (464, 848))
    for i in range(int(seconds * fps)):
        f = np.full((848, 464, 3), 70, np.uint8)
        f[300:400, 50 + i % 200:110 + i % 200] = (0, 200, 0)
        w.write(f)
    w.release()
    return path


# ------------------------------------------------------------------ публичный API


def test_public_functions_stay_callable_after_submodule_imports():
    import ml.vlm.verifier  # noqa: F401  — подмодули не должны затирать функции пакета
    import ml.vlm.discoverer  # noqa: F401
    assert callable(vlm.verify) and callable(vlm.discover)


def test_verify_never_raises(clip):
    class Exploding(VLMBackend):
        name = "boom"

        def generate(self, *a, **k):
            raise RuntimeError("GPU on fire")

    vlm.set_backend(Exploding())
    ev = Event("V05", 1.0, 2.0, 1.5, 0.7, "rules")
    out = vlm.verify(clip, [ev], 10.0)
    assert len(out) == 1 and out[0].confidence == 0.7 and out[0].extra["vlm"]["status"] == "error"


def test_verify_with_mock_env(clip):
    out = vlm.verify(clip, [Event("V05", 1.0, 2.0, 1.5, 0.7, "rules")], 30.0)
    assert out[0].extra["vlm"]["status"] == "ok" and out[0].source == "rules"


def test_discover_disabled_by_default(clip):
    vlm.set_backend(MockBackend(mode="phone"))
    assert vlm.discover(clip, 60.0) == []


def test_status_warmup_unload_are_technical_only():
    vlm.warmup()
    st = vlm.status()
    assert st["backend"] == "mock" and set(st) <= {"backend", "model", "reason", "url", "schema_mode", "quant",
                                                   "loaded", "load_s", "offline", "license"}
    vlm.unload()


def test_gpu_lock_is_used(clip):
    entered = []

    class Lock:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *a):
            return False

    vlm.set_gpu_lock(Lock())
    vlm.verify(clip, [Event("V05", 1.0, 2.0, 1.5, 0.7, "rules")], 30.0)
    assert entered  # generate шёл под общим GPU-lock от A03


def test_limits_kwarg(clip):
    be = MockBackend(mode="confirm")
    vlm.set_backend(be)
    evs = [Event("V05", t, t + 1, t + 0.5, 0.5, "rules") for t in (1.0, 4.0, 8.0)]
    out = vlm.verify(clip, evs, 60.0, limits={"max_calls": 1})
    assert len(be.calls) == 1 and sum(e.extra["vlm"]["status"] == "ok" for e in out) == 1


# ------------------------------------------------------------------ CLI (приватность вывода)


def test_try_cli_prints_aggregates_only(tmp_path, monkeypatch, capsys):
    video = _write_video(tmp_path / "synthetic.avi")
    monkeypatch.setenv("SENTINEL_VLM_MOCK_MODE", "confirm")
    from ml.vlm.cli_try import main
    rc = main([str(video), "--t", "3", "--type", "V05", "--mode", "both", "--dump-dir", str(tmp_path / "dump")])
    out = capsys.readouterr().out
    rep = json.loads(out)
    assert rc == 0 and rep["verify"]["status"] == "ok" and rep["verify"]["agreement"] == "supports"
    assert "texts" not in rep and "observed_facts" not in out and "Локальная VLM" not in out
    d = Path(rep["dump_dir"])
    assert (d / "records.jsonl").exists() and (d / "report.json").exists() and list(d.glob("*.jpg"))
    full = json.loads((d / "report.json").read_text(encoding="utf-8"))
    assert "texts" in full  # тексты — только в локальном дампе


def test_try_cli_show_text_and_no_dump(tmp_path, monkeypatch, capsys):
    video = _write_video(tmp_path / "synthetic.avi")
    monkeypatch.setenv("SENTINEL_VLM_MOCK_MODE", "confirm")
    from ml.vlm.cli_try import main
    main([str(video), "--t", "3", "--no-dump", "--show-text"])
    rep = json.loads(capsys.readouterr().out)
    assert rep["dump_dir"] is None and "texts" in rep


def test_try_module_entrypoint(tmp_path):
    video = _write_video(tmp_path / "synthetic.avi", seconds=3.0)
    env = dict(os.environ, SENTINEL_MOCK="1", PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-m", "ml.vlm.try", str(video), "--t", "1.5", "--no-dump"],
                       capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["verify"]["status"] == "ok"


def test_bench_synthetic(tmp_path, capsys):
    from ml.vlm.bench import main
    rc = main(["--synthetic", "6", "--n", "3", "--dump-dir", str(tmp_path / "d"), "--out", str(tmp_path / "r.json")])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and rep["calls"] == 3 and rep["json_valid_rate"] == 1.0
    assert rep["vram_peak_mb_torch"] is None  # без GPU не выдумываем цифры
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["calls"] == 3


def test_bench_frames_dir(tmp_path, capsys):
    from PIL import Image
    fr = tmp_path / "frames"
    fr.mkdir()
    for i in range(4):
        Image.new("RGB", (464, 848), (i * 20, 0, 0)).save(fr / f"{i:02d}.png")
    from ml.vlm.bench import main
    main(["--frames", str(fr), "--n", "2", "--images-per-call", "4", "--no-dump"])
    rep = json.loads(capsys.readouterr().out)
    assert rep["n_images_loaded"] == 4 and rep["visual_tokens_est_per_call"] > 0


def test_bench_without_input(capsys):
    from ml.vlm.bench import main
    assert main([]) == 2


def test_calibrate_aggregates_only(tmp_path, monkeypatch, capsys):
    import shutil
    vids = tmp_path / "videos"
    vids.mkdir()
    _write_video(vids / "X-1.avi", seconds=8.0)
    _write_video(vids / "Х-3.avi", seconds=7.0)          # кириллическая Х (другое содержимое)
    shutil.copy(vids / "X-1.avi", vids / "X-47.avi")      # дубликат по md5
    (tmp_path / "labels.csv").write_text("file;label;t\nx-1.mov;нарушение;22 секунда\nХ-3.MOV;без нарушения;\n"
                                         "X-47.mov;нарушение;5\n", encoding="utf-8")
    monkeypatch.setenv("SENTINEL_VLM_MOCK_MODE", "phone")
    from ml.vlm.calibrate import main, normalize_name
    assert normalize_name("Х-3.MOV") == normalize_name("x-3.mov") == "x-3"
    rc = main(["--videos", str(vids), "--labels", str(tmp_path / "labels.csv"), "--neg-per-video", "2",
               "--dump-dir", str(tmp_path / "dump")])
    out = capsys.readouterr().out
    rep = json.loads(out)
    assert rc == 0 and rep["videos_used"] == 2 and rep["duplicates_skipped"] == 1
    assert rep["discover_n_pos"] == 1 and rep["discover_n_neg"] == 2
    assert rep["discover_hit_rate_neg"] == 1.0  # mock «phone» находит везде — это и показывает FPR
    assert rep["verify_n_pos"] == 1 and rep["verify_json_ok_rate"] == 1.0
    assert "Кассир" not in out and "observed_facts" not in out  # никаких описаний сцен в stdout
