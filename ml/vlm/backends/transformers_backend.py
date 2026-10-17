"""transformers-бэкенд: Qwen3-VL-4B/8B-Instruct (bnb 4-bit) или Qwen2.5-VL-7B-Instruct-AWQ. Apache-2.0.

Тяжёлые импорты (torch/transformers) — только внутри методов. Модель грузится лениво и держится тёплой;
unload() освобождает VRAM (оркестратор может выгружать VLM между видео при нехватке памяти).
"""
from __future__ import annotations

import gc
import importlib.util
import logging
import os
import threading
import time
from typing import Any, Optional

from .base import BackendUnavailable, GenResult, VLMBackend

log = logging.getLogger(__name__)


def _hf_offline() -> bool:
    return str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}


def read_model_license(model_id: str) -> Optional[str]:
    """license из YAML front-matter README.md локальной папки или снапшота в HF-кэше (без сети)."""
    readme: Optional[str] = None
    if os.path.isdir(model_id):
        cand = os.path.join(model_id, "README.md")
        readme = cand if os.path.exists(cand) else None
    else:
        try:
            from huggingface_hub import try_to_load_from_cache

            r = try_to_load_from_cache(model_id, "README.md")
            readme = r if isinstance(r, str) and os.path.exists(r) else None
        except Exception:  # noqa: BLE001
            readme = None
    if not readme:
        return None
    with open(readme, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(8192)
    if not text.startswith("---"):
        return None
    fm = text.split("---", 2)[1] if text.count("---") >= 2 else ""
    for line in fm.splitlines():
        if line.strip().lower().startswith("license:"):
            return line.split(":", 1)[1].strip().strip("'\"").lower() or None
    return None


def check_license(model_id: str, mode: str = "warn", allowed: Optional[list[str]] = None) -> tuple[bool, str]:
    """strict: лицензия должна быть найдена и разрешена; warn: запрещённая — блок, не найдена — предупреждение."""
    mode = (mode or "warn").lower()
    if mode == "off":
        return True, "license check off"
    allowed_set = {a.lower() for a in (allowed or ["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"])}
    lic = read_model_license(model_id)
    if lic is None:
        msg = f"license of {model_id} not found locally (README.md front-matter)"
        return (mode != "strict"), msg
    if lic not in allowed_set:
        return False, f"license '{lic}' of {model_id} is not in allowed {sorted(allowed_set)}"
    return True, f"license {lic}"


class TransformersBackend(VLMBackend):
    name = "transformers"

    def __init__(self, model: str, quant: str = "bnb4", device: str = "cuda", allow_cpu: bool = False,
                 dtype: str = "float16", attn_implementation: str = "sdpa", max_pixels: int = 448 * 448,
                 min_pixels: int = 3136, skip_quant_modules: Optional[list[str]] = None,
                 local_files_only: Optional[bool] = None, license_guard: str = "warn",
                 allowed_licenses: Optional[list[str]] = None, min_free_vram_mb: int = 0, **_: Any) -> None:
        self.model_id = model
        self.min_free_vram_mb = int(min_free_vram_mb or 0)
        self.license_guard = license_guard
        self.allowed_licenses = list(allowed_licenses or ["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"])
        self.license_info = ""
        self.quant = (quant or "none").lower()
        self.device = device
        self.allow_cpu = allow_cpu
        self.dtype_name = dtype
        self.attn = attn_implementation
        self.max_pixels = int(max_pixels)
        self.min_pixels = int(min_pixels)
        self.skip_modules = list(skip_quant_modules or ["visual", "model.visual", "lm_head"])
        self.local_files_only = _hf_offline() if local_files_only is None else bool(local_files_only)
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()
        self.load_s: float = 0.0

    # ------------------------------------------------------------------ lifecycle
    def available(self) -> tuple[bool, str]:
        for mod in ("torch", "transformers", "PIL"):
            if importlib.util.find_spec(mod) is None:
                return False, f"python package '{mod}' not installed"
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available() and not self.allow_cpu:
            return False, "CUDA not available (transformers.allow_cpu=false)"
        if self.quant.startswith("bnb") and importlib.util.find_spec("bitsandbytes") is None:
            return False, "bitsandbytes not installed (quant=bnb4)"
        if self.quant == "awq" and importlib.util.find_spec("awq") is None and importlib.util.find_spec("gptqmodel") is None:
            return False, "autoawq/gptqmodel not installed (quant=awq)"
        return True, ""

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _device(self) -> str:
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            ok, why = self.available()
            if not ok:
                raise BackendUnavailable(why)
            lic_ok, self.license_info = check_license(self.model_id, self.license_guard, self.allowed_licenses)
            if not lic_ok:
                raise BackendUnavailable(self.license_info)
            if self.license_info.startswith("license of"):
                log.warning("%s", self.license_info)
            import torch
            import transformers
            from transformers import AutoProcessor

            t0 = time.perf_counter()
            dev = self._device()
            free_mb = _free_vram_mb() if dev.startswith("cuda") else None
            if free_mb is not None and self.min_free_vram_mb and free_mb < self.min_free_vram_mb:
                # не грузим, чтобы не выбить OOM у детектора/энкодеров: A03 может вызвать unload() у других
                raise BackendUnavailable(f"free VRAM {free_mb:.0f} MB < min_free_vram_mb {self.min_free_vram_mb}")
            dtype = getattr(torch, self.dtype_name) if dev != "cpu" else torch.float32
            kw: dict[str, Any] = {"device_map": dev, "attn_implementation": self.attn,
                                  "local_files_only": self.local_files_only}
            ver = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
            kw["dtype" if ver >= (4, 56) else "torch_dtype"] = dtype
            if self.quant in ("bnb4", "bnb8") and dev != "cpu":
                from transformers import BitsAndBytesConfig

                if self.quant == "bnb4":
                    kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=self.skip_modules)
                else:
                    kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,
                                                                    llm_int8_skip_modules=self.skip_modules)
            self._model = self._load_model(kw)
            self._model.eval()
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels,
                    local_files_only=self.local_files_only)
            except (TypeError, ValueError):
                self._processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=self.local_files_only)
            self.load_s = time.perf_counter() - t0
            log.info("VLM %s loaded in %.1fs (quant=%s, device=%s)", self.model_id, self.load_s, self.quant, dev)

    def _load_model(self, kw: dict[str, Any]) -> Any:
        import transformers

        errors = []
        for cls_name in ("AutoModelForImageTextToText", "Qwen3VLForConditionalGeneration",
                         "Qwen2_5_VLForConditionalGeneration", "AutoModelForVision2Seq"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(self.model_id, **kw)
            except (ValueError, KeyError) as e:  # класс не подходит к конфигу — пробуем следующий
                errors.append(f"{cls_name}: {e}")
        raise BackendUnavailable("cannot load %s: %s" % (self.model_id, " | ".join(errors) or "no VLM class in transformers"))

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    # ------------------------------------------------------------------ inference
    def generate(self, images: list[Any], system: str, prompt: str, max_new_tokens: int = 256,
                 timeout_s: float = 10.0, json_schema: Optional[dict[str, Any]] = None,
                 temperature: float = 0.0) -> GenResult:
        if self._model is None:
            self.load()
        import torch

        with self._lock:
            t0 = time.perf_counter()
            content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                        {"role": "user", "content": content}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=list(images) or None, return_tensors="pt", padding=True)
            inputs = inputs.to(self._model.device)
            gen_kw: dict[str, Any] = {"max_new_tokens": int(max_new_tokens),
                                      "max_time": max(0.5, float(timeout_s))}
            if temperature and temperature > 0:
                gen_kw.update(do_sample=True, temperature=float(temperature))
            else:
                gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            try:
                with torch.inference_mode():
                    out = self._model.generate(**inputs, **gen_kw)
            except torch.cuda.OutOfMemoryError:
                del inputs
                torch.cuda.empty_cache()
                log.warning("VLM generate: CUDA OOM (уменьшите frames.max_side / max_visual_tokens_per_call)")
                return GenResult(text="", latency_s=time.perf_counter() - t0, error="cuda_oom")
            n_in = int(inputs["input_ids"].shape[1])
            gen_ids = out[:, n_in:]
            txt = self._processor.batch_decode(gen_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)[0]
            lat = time.perf_counter() - t0
            n_out = int(gen_ids.shape[1])
            return GenResult(text=txt, latency_s=lat, prompt_tokens=n_in, completion_tokens=n_out,
                             timed_out=lat >= timeout_s and n_out < max_new_tokens,
                             meta={"vram_peak_mb": _vram_peak_mb()})

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model_id, "quant": self.quant, "loaded": self.loaded,
                "load_s": round(self.load_s, 2), "offline": self.local_files_only, "license": self.license_info}


def _free_vram_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 2**20
    except Exception:  # noqa: BLE001
        pass
    return None


def _vram_peak_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    return None
