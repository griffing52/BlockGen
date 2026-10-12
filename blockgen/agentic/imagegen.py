"""Image generators for dreams: OpenAI gpt-image, local diffusers, offline mocks.

A *dream* is an image of the house the model is asked to build; the suite's dream
loop builds towards it. Generators implement :class:`core.ImageGen` and are named
in a recipe like LLMs are (``image: openai:gpt-image-1-mini?quality=low``), so
swapping the paid API for local SDXL or for an offline mock is a one-line change.

Every generator is wrapped by :class:`CachingImageGen` (via :func:`get_image_gen`),
for the same reasons as the LLM response cache, plus one specific to dreams: the
cache is shared across arms and keyed on the request content, so holding the dream
spec and seed fixed while ablating the loop reuses the SAME images, which isolates
loop variance from dream variance. gpt-image ignores seeds, so for it the cache is
the only source of reproducibility.

Heavy dependencies (``openai``, ``diffusers``, ``torch``, the corpus, the renderer)
are imported inside the classes that need them: importing this module is cheap
and never touches a GPU (core.load_builtins imports it for registration).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import json
import math
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from blockgen.agentic.core import (CACHE_MODES, CacheMiss, ImageGen, ImageRequest,
                                   ImageResult, Spec, VIEWS, get_view, make, register,
                                   register_lazy, require_spend)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_CACHE = _REPO / "outputs" / "cache" / "imagegen"

# ===========================================================================
# Prices
# ===========================================================================
#: USD per OUTPUT image, (model, size, quality). OpenAI list prices as published
#: for gpt-image-1 / gpt-image-1-mini (re-check https://openai.com/api/pricing
#: before a paid battery). A combination missing here is reported UNPRICED
#: (priced=False), never as $0, like providers.PRICES.
IMAGE_PRICES: Dict[Tuple[str, str, str], float] = {
    ("gpt-image-1", "1024x1024", "low"): 0.011,
    ("gpt-image-1", "1024x1024", "medium"): 0.042,
    ("gpt-image-1", "1024x1024", "high"): 0.167,
    ("gpt-image-1", "1024x1536", "low"): 0.016,
    ("gpt-image-1", "1536x1024", "low"): 0.016,
    ("gpt-image-1", "1024x1536", "medium"): 0.063,
    ("gpt-image-1", "1536x1024", "medium"): 0.063,
    ("gpt-image-1", "1024x1536", "high"): 0.25,
    ("gpt-image-1", "1536x1024", "high"): 0.25,
    ("gpt-image-1-mini", "1024x1024", "low"): 0.005,
    ("gpt-image-1-mini", "1024x1024", "medium"): 0.011,
    ("gpt-image-1-mini", "1024x1024", "high"): 0.036,
    ("gpt-image-1-mini", "1024x1536", "low"): 0.006,
    ("gpt-image-1-mini", "1536x1024", "low"): 0.006,
    ("gpt-image-1-mini", "1024x1536", "medium"): 0.015,
    ("gpt-image-1-mini", "1536x1024", "medium"): 0.015,
}

#: ESTIMATE, USD per INPUT token (text, image) for gpt-image models (OpenAI list
#: rates as recalled; re-check https://openai.com/api/pricing before a paid
#: battery). Used to price the prompt and the reference images, which gpt-image
#: bills on every request on top of the output image.
INPUT_TOKEN_USD: Dict[str, Tuple[float, float]] = {
    "gpt-image-1": (5.0e-6, 10.0e-6),
    "gpt-image-1-mini": (2.0e-6, 2.5e-6),
}

#: ESTIMATE, USD added per reference image of an edit request, used only BEFORE a
#: call (budget checks) or when the response carries no ``usage``. A 1024px input
#: is about 765 image tokens; these constants round that UP at the image-input
#: rates above (input_fidelity=high costs more, which only the measured path sees).
REF_SURCHARGE_USD: Dict[str, float] = {"gpt-image-1": 0.008, "gpt-image-1-mini": 0.003}


def _prompt_tokens_upper(prompt: str) -> int:
    """Over-approximate token count of a prompt (about 4 chars/token; use 3)."""
    return math.ceil(len(prompt or "") / 3)


def image_price(model: str, size: str, quality: str, *, n: int = 1,
                n_refs: int = 0, prompt: str = "") -> Optional[float]:
    """Pre-call ESTIMATE of one request, or None if the (model, size, quality) is
    unknown: output images + reference-image surcharge + the prompt's text tokens
    (over-approximated, so a budget check never under-counts)."""
    per = IMAGE_PRICES.get((model, size, quality))
    if per is None:
        return None
    text_rate = INPUT_TOKEN_USD.get(model, (0.0, 0.0))[0]
    return (per * n + REF_SURCHARGE_USD.get(model, 0.0) * n_refs
            + text_rate * _prompt_tokens_upper(prompt))


def measured_image_price(model: str, size: str, quality: str, usage: Any, *,
                         n: int = 1) -> Optional[float]:
    """Actual cost of a finished request from its ``usage`` (input text and image
    tokens at INPUT_TOKEN_USD, output images at IMAGE_PRICES), or None when the
    usage or a rate is missing (the caller then falls back to the estimate)."""
    per = IMAGE_PRICES.get((model, size, quality))
    rates = INPUT_TOKEN_USD.get(model)
    if per is None or rates is None or usage is None:
        return None
    details = getattr(usage, "input_tokens_details", None)
    text = getattr(details, "text_tokens", None)
    img = getattr(details, "image_tokens", None)
    if text is None and img is None:
        total = getattr(usage, "input_tokens", None)
        if total is None:
            return None
        # No split: bill everything at the (higher) image-input rate.
        return per * n + float(total) * max(rates)
    return per * n + float(text or 0) * rates[0] + float(img or 0) * rates[1]


#: Parameters only one images endpoint accepts (openai 2.x). Constructor params
#: are passed to the API verbatim, so a generate-only key such as ``moderation``
#: must be dropped on edits (dreams with refs), and vice versa.
GENERATE_ONLY_PARAMS = frozenset({"moderation", "style"})
EDIT_ONLY_PARAMS = frozenset({"input_fidelity", "mask"})


def _endpoint_kwargs(fn: Any, kwargs: Dict[str, Any], drop: frozenset) -> Dict[str, Any]:
    """``kwargs`` fitted to one SDK endpoint: keys of the other endpoint dropped,
    keys the SDK method does not name routed through ``extra_body``."""
    import inspect
    out = {k: v for k, v in kwargs.items() if k not in drop}
    try:
        sig = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return out
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values()):
        return out
    extra = {k: out.pop(k) for k in list(out) if k not in sig}
    if extra:
        out["extra_body"] = {**out.get("extra_body", {}), **extra}
    return out


def _size_px(size: str) -> Tuple[int, int]:
    """'1024x768' -> (1024, 768); 'auto' or junk -> (1024, 1024)."""
    try:
        w, h = (int(v) for v in str(size).lower().split("x"))
        return w, h
    except ValueError:
        return 1024, 1024


def _png(arr: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _digest(*parts: Any) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode()).digest()


def _req_digest(req: ImageRequest) -> bytes:
    """Deterministic hash of what a mock should vary on (prompt, seed, refs)."""
    refs = [hashlib.sha256(r).hexdigest()[:16] for r in req.refs]
    return _digest(req.prompt, req.seed, refs)


# ===========================================================================
# OpenAI gpt-image
# ===========================================================================
@register("image", "openai", env="OPENAI_API_KEY")
class OpenAIImageGen:
    """OpenAI gpt-image (paid API): text-to-image, or edits when refs are given.

    ``quality`` / ``size`` set here OVERRIDE the request's; left as None the
    request's values are used. :meth:`effective` applies that before the cache
    keys the request, so the key always names what was actually generated.

    The ``quality`` / ``size`` ATTRIBUTES are always concrete strings (the
    override, else ``DEFAULT_QUALITY`` / ``DEFAULT_SIZE``): Episode.dream builds
    its ImageRequest from ``str(getattr(gen, 'quality', ...))``, so a None here
    would reach the API (and the cache key and price lookup) as the string 'None'.
    """

    name = "openai"
    spends = True
    billing = "api"
    DEFAULT_QUALITY = "low"
    DEFAULT_SIZE = "1024x1024"

    def __init__(self, model: str = "gpt-image-1-mini", *, quality: Optional[str] = None,
                 size: Optional[str] = None, background: str = "opaque", **params: Any) -> None:
        self.model = model
        self._quality = quality         # explicit overrides (None = take the request's)
        self._size = size
        self.background = background
        # Passed to the API verbatim (e.g. moderation); generate-only / edit-only
        # keys are dropped on the other endpoint (see _endpoint_kwargs).
        self.params = params
        self._client = None

    @property
    def quality(self) -> str:
        return self._quality or self.DEFAULT_QUALITY

    @property
    def size(self) -> str:
        return self._size or self.DEFAULT_SIZE

    def effective(self, req: ImageRequest) -> ImageRequest:
        return dataclasses.replace(req, quality=self._quality or req.quality,
                                   size=self._size or req.size)

    def key_params(self) -> Dict[str, Any]:
        """Output-affecting settings that are not in the ImageRequest (cache key)."""
        return {"background": self.background, **self.params}

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI

            from blockgen.agentic.providers import _require_key
            self._client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
        return self._client

    def generate(self, req: ImageRequest) -> ImageResult:
        from blockgen.agentic.providers import _retry
        req = self.effective(req)
        kwargs: Dict[str, Any] = {"model": self.model, "prompt": req.prompt, "n": req.n,
                                  "size": req.size, "quality": req.quality, **self.params}
        # The request's 'white' background is a prompt-level wish; the API takes
        # transparent | opaque | auto only.
        bg = req.background if req.background in ("transparent", "opaque", "auto") \
            else self.background
        if bg:
            kwargs["background"] = bg
        client = self._client_lazy()
        t0 = time.time()
        if req.refs:
            files = [(f"ref{i}.png", io.BytesIO(r), "image/png") for i, r in enumerate(req.refs)]
            edit_kw = _endpoint_kwargs(client.images.edit, kwargs, GENERATE_ONLY_PARAMS)
            resp = _retry(lambda: client.images.edit(image=files, **edit_kw))
        else:
            gen_kw = _endpoint_kwargs(client.images.generate, kwargs, EDIT_ONLY_PARAMS)
            resp = _retry(lambda: client.images.generate(**gen_kw))
        import base64
        images = [base64.b64decode(d.b64_json) for d in resp.data]
        revised = next((getattr(d, "revised_prompt", None) or "" for d in resp.data), "")
        # Prefer the measured cost (the response's token usage); the estimate is
        # the fallback when the API returns no usage.
        price = measured_image_price(self.model, req.size, req.quality,
                                     getattr(resp, "usage", None), n=req.n)
        if price is None:
            price = self.estimate_usd(req)
        return ImageResult(images=images, model=self.model, cost_usd=price or 0.0,
                           priced=price is not None, latency_s=time.time() - t0,
                           revised_prompt=revised)

    def estimate_usd(self, req: ImageRequest) -> Optional[float]:
        req = self.effective(req)
        return image_price(self.model, req.size, req.quality, n=req.n, n_refs=len(req.refs),
                           prompt=req.prompt)

    def supports_refs(self) -> bool:
        return True

    def describe(self) -> str:
        return f"openai:{self.model}"


# ===========================================================================
# Local diffusers (lazy)
# ===========================================================================
_PIPES: Dict[Tuple[str, str, bool], Any] = {}
_PIPES_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()


class DiffusersImageGen:
    """Local text-to-image / image-to-image with HF diffusers (free, GPU).

    One pipeline per (model, dtype, offload) per process, shared by every instance,
    because loading SDXL takes seconds and gigabytes. :meth:`unload` frees it; the
    suite calls it between a ``--dreams-first`` pass and scoring so diffusers, the
    scorer backbones and EGL do not contend for one 16 GB GPU.
    """

    name = "diffusers"
    spends = False
    billing = "local"
    ALIASES = {"sdxl-turbo": "stabilityai/sdxl-turbo",
               "flux-schnell": "black-forest-labs/FLUX.1-schnell"}
    #: Native resolution per model family (substring of the HF id, lower-cased).
    #: sdxl-turbo is distilled at 512px; asking it for 1024px gives tiled, incoherent
    #: images, so the default request size must follow the model, not the API's.
    NATIVE_SIZES = {"sdxl-turbo": "512x512", "sd-turbo": "512x512"}
    #: GiB of VRAM a family needs to load fully on the GPU (offload=False). The
    #: FLUX.1 transformer alone is ~12B params (~24 GB in bf16) plus T5-XXL, so it
    #: does not fit a 16 GB card even with model-level offload (which moves whole
    #: components). For these families offload=True means SEQUENTIAL offload
    #: (submodule by submodule: fits 16 GB, slow, needs ~35 GB of system RAM).
    MIN_VRAM_GB = {"flux": 34.0}
    SEQUENTIAL_OFFLOAD = ("flux",)

    def __init__(self, model: str = "sdxl-turbo", *, steps: int = 4, guidance: float = 0.0,
                 offload: bool = False, strength: float = 0.6, dtype: str = "auto",
                 size: Optional[str] = None, device: Optional[str] = None, **params: Any) -> None:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"image generator 'diffusers:{model}' needs the optional packages diffusers "
                f"and accelerate: pip install diffusers accelerate ({exc})") from exc
        self.model = model
        self.model_id = self.ALIASES.get(model, model)
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.offload = bool(offload)
        self.strength = float(strength)
        # 'auto': bf16 for FLUX (it overflows in fp16 and returns black/NaN images),
        # fp16 for everything else. Resolved here so the cache key names the real dtype.
        if dtype == "auto":
            dtype = "bf16" if "flux" in self.model_id.lower() else "fp16"
        if dtype not in ("fp16", "bf16", "fp32"):
            raise ValueError(f"diffusers dtype must be auto|fp16|bf16|fp32, got {dtype!r}")
        self.dtype = dtype
        self._size = size                 # explicit override (None = take the request's)
        self.device = device
        self.params = params              # extra pipeline-call kwargs

    @property
    def size(self) -> str:
        """Concrete size Episode.dream puts in its request: the override, else the
        model's native resolution (see NATIVE_SIZES), else 1024x1024."""
        if self._size:
            return self._size
        mid = self.model_id.lower()
        return next((v for k, v in self.NATIVE_SIZES.items() if k in mid), "1024x1024")

    def key_params(self) -> Dict[str, Any]:
        return {"steps": self.steps, "guidance": self.guidance, "strength": self.strength,
                "dtype": self.dtype, "size": self._size, **self.params}

    def effective(self, req: ImageRequest) -> ImageRequest:
        return dataclasses.replace(req, size=self._size or req.size)

    def _family(self, table: Any) -> Optional[str]:
        mid = self.model_id.lower()
        return next((k for k in table if k in mid), None)

    def check_vram(self, total_bytes: Optional[int]) -> None:
        """Fail early (before a multi-GB download) when the model cannot fit
        ``total_bytes`` of VRAM without offload. None = unknown/CPU: no check."""
        fam = self._family(self.MIN_VRAM_GB)
        if self.offload or fam is None or total_bytes is None:
            return
        need = self.MIN_VRAM_GB[fam]
        have = total_bytes / 2**30
        if have < need:
            raise RuntimeError(
                f"diffusers:{self.model} needs ~{need:.0f} GiB of VRAM to load on the GPU, "
                f"this GPU has {have:.1f} GiB; pass offload=true (sequential CPU offload: "
                f"fits, but slow) or use sdxl-turbo")

    def _pipe(self):
        import torch
        from diffusers import AutoPipelineForText2Image
        key = (self.model_id, self.dtype, self.offload)
        with _PIPES_LOCK:
            if key not in _PIPES:
                dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                if not self.offload and str(dev).startswith("cuda"):
                    self.check_vram(torch.cuda.get_device_properties(
                        torch.device(dev)).total_memory)
                torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                               "fp32": torch.float32}[self.dtype]
                kw: Dict[str, Any] = {"torch_dtype": torch_dtype}
                try:
                    # The fp16 files are half the download of the fp32 ones, and
                    # diffusers upcasts them to torch_dtype on load, so a CPU run
                    # in bf16/fp32 reuses the same ~7 GB of SDXL-Turbo weights.
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        self.model_id, variant="fp16", **kw)
                except (OSError, ValueError):   # repo without an fp16 variant (e.g. FLUX)
                    pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, **kw)
                if self.offload and self._family(self.SEQUENTIAL_OFFLOAD):
                    pipe.enable_sequential_cpu_offload()
                elif self.offload:
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(dev)
                pipe.set_progress_bar_config(disable=True)
                _PIPES[key] = pipe
            return _PIPES[key]

    def generate(self, req: ImageRequest) -> ImageResult:
        import torch
        from PIL import Image
        req = self.effective(req)
        w, h = _size_px(req.size)
        pipe = self._pipe()
        gen_device = "cpu" if self.offload else str(getattr(pipe, "device", "cpu"))
        generator = torch.Generator(gen_device).manual_seed(int(req.seed))
        kw: Dict[str, Any] = {"prompt": req.prompt, "num_inference_steps": self.steps,
                              "guidance_scale": self.guidance, "num_images_per_prompt": req.n,
                              "generator": generator, **self.params}
        t0 = time.time()
        # One generation at a time per process: a diffusers pipeline (its scheduler
        # state) is not thread-safe, and the suite runs prompts on worker threads.
        with _GEN_LOCK:
            if req.refs:
                from diffusers import AutoPipelineForImage2Image
                img2img = AutoPipelineForImage2Image.from_pipe(pipe)
                init = Image.open(io.BytesIO(req.refs[0])).convert("RGB").resize((w, h))
                # img2img runs steps*strength denoising steps; turbo models need >= 1.
                steps = max(self.steps, math.ceil(1.0 / max(self.strength, 1e-3)))
                out = img2img(image=init, strength=self.strength,
                              **{**kw, "num_inference_steps": steps})
            else:
                out = pipe(width=w, height=h, **kw)
        images = []
        for im in out.images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            images.append(buf.getvalue())
        return ImageResult(images=images, model=self.model, cost_usd=0.0, priced=False,
                           latency_s=time.time() - t0)

    def estimate_usd(self, req: ImageRequest) -> Optional[float]:
        return None   # local compute: unpriced, not "$0"

    def supports_refs(self) -> bool:
        return True   # first ref only (img2img takes one init image)

    def describe(self) -> str:
        return f"diffusers:{self.model}"

    def unload(self) -> None:
        """Free every cached pipeline (and its VRAM)."""
        with _PIPES_LOCK:
            _PIPES.clear()
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


register_lazy("image", "diffusers", "blockgen.agentic.imagegen:DiffusersImageGen",
              "local-image", doc="Local diffusers text/img2img (sdxl-turbo; flux-schnell needs "
                                 "offload=true on a 16 GB GPU); needs pip install "
                                 "diffusers accelerate")


# ===========================================================================
# Offline mocks
# ===========================================================================
MOCK_MODES = ("solid", "render", "dir")

#: ``ImageRequest.view`` prefix of a turnaround sheet hint (``Episode.dream(view_hint=)``):
#: ``grid2x2:iso,iso_left,iso_back,iso_right`` asks a mock for a 2x2 sheet of those views.
GRID_HINT = "grid2x2:"


def parse_grid_hint(view: str) -> Optional[List[str]]:
    """The panel views of a ``grid2x2:`` hint (None if ``view`` is not one).
    Names not in VIEWS stay as given; the renderer falls back to iso for them."""
    if not str(view or "").startswith(GRID_HINT):
        return None
    names = [v.strip() for v in str(view)[len(GRID_HINT):].split(",") if v.strip()]
    return (names + ["iso"] * 4)[:4]


@functools.lru_cache(maxsize=4)
def _train_pool(corpus: str, split: str) -> Tuple[Any, ...]:
    """The structures of one side of the pinned split (cached per process).

    Read straight from the split file's indices: a mock only needs WHICH builds
    are train, and splits.load_split would re-hash the whole corpus to verify it.
    """
    from blockgen.eval.bench.splits import SPLIT_ROOT, load_corpus, split_path
    path = _REPO / split_path(corpus, 0, (0.70, 0.15, 0.15), SPLIT_ROOT)
    blob = json.loads(path.read_text())
    structs, _ = load_corpus(corpus, _REPO / blob.get("cache_dir", "data/minecraft/cache"))
    return tuple(structs[i] for i in blob[split])


def _project(structure: Any, px: int) -> np.ndarray:
    """Pure-numpy stand-in render: front orthographic colour projection, shaded by depth.

    Used by ``mock:render`` only when neither the textured nor the flat renderer
    is available (render.py not landed, no EGL): a dream must still be a picture
    of a real build, not a placeholder.
    """
    s = structure.crop_to_non_air()
    ids, data = s.block_ids, s.block_data
    occ = ids != s.air_block_id
    if not occ.any():
        return np.full((px, px, 3), 255, np.uint8)
    from blockgen.utils.data import _color_for, _resource_location_for
    X, Y, Z = ids.shape
    first = np.where(occ.any(axis=2), occ.argmax(axis=2), -1)        # (X, Y) nearest z
    img = np.full((Y, X, 3), 255.0)
    lut: Dict[Tuple[int, int], np.ndarray] = {}
    for x in range(X):
        for y in range(Y):
            z = first[x, y]
            if z < 0:
                continue
            k = (int(ids[x, y, z]), int(data[x, y, z]))
            if k not in lut:
                lut[k] = np.asarray(_color_for(_resource_location_for(*k), k[1])[:3]) * 255.0
            img[Y - 1 - y, x] = lut[k] * (1.0 - 0.5 * z / max(1, Z))
    side = max(X, Y)
    canvas = np.full((side, side, 3), 255.0)
    oy, ox = (side - Y) // 2, (side - X) // 2
    canvas[oy:oy + Y, ox:ox + X] = img
    idx = (np.arange(px) * side // px).clip(0, side - 1)
    return canvas[idx][:, idx].astype(np.uint8)


@register("image", "mock")
class MockImageGen:
    """Offline, deterministic dreams: ``mock:solid`` | ``mock:render`` | ``mock:dir``.

    * ``solid``  a labelled placeholder: colour from hash(prompt, seed, refs), the
      prompt's first words drawn on it. No dependencies beyond PIL.
    * ``render`` a render of a REAL build from the houses_32 TRAIN split, chosen by
      hash(prompt, seed), at ``VIEWS[req.view or 'iso']``. This is the "oracle"
      dream: a target that is known to be buildable, used for the ceiling arm and
      scorer calibration. The test split is refused (leakage), and recipes using it
      are tagged ``train_split: oracle_train`` so they are never ranked.
    * ``dir``    replays PNGs from ``path`` (e.g. real dreams saved earlier), picked
      by hash(prompt, seed) modulo their count.

    A ``grid2x2:<v1>,<v2>,<v3>,<v4>`` view hint (a turnaround sheet) makes
    ``render`` draw ONE pool house at those four views as a 2x2 sheet, and
    ``solid`` four distinct coloured panels on white with gutters, so the sheet
    splitter and the per-panel assignment run offline. ``dir`` ignores it. A view
    name not in VIEWS renders as iso instead of raising (an API ignores the hint;
    a mock should not be stricter than the model it stands in for).

    Mocks are unpriced (``priced=False``): offline is not "free" in a cost table.
    """

    name = "mock"
    spends = False
    billing = "mock"

    def __init__(self, model: str = "solid", *, split: str = "train", corpus: str = "houses_32",
                 renderer: str = "textured", path: Optional[str] = None, px: Optional[int] = None,
                 structures: Optional[Sequence[Any]] = None) -> None:
        if model not in MOCK_MODES:
            raise ValueError(f"mock image mode must be one of {MOCK_MODES}, got {model!r}")
        if model == "render" and split not in ("train", "val"):
            raise ValueError(f"mock:render split={split!r} refused: dreams from the "
                             f"{split} split would leak evaluation targets; use train")
        if model == "dir" and not path:
            raise ValueError("mock:dir needs path=<directory of PNGs>")
        self.model = model
        self.split = split
        self.corpus = corpus
        self.renderer_name = renderer
        self.path = path
        self.px = px
        self._structures = list(structures) if structures is not None else None
        self._custom_pool = structures is not None
        self._renderer: Any = None
        self.renderer_used = ""           # which renderer actually drew (provenance)
        self._drawn: Dict[str, int] = {}  # output PNG sha256 -> pool index (render mode)

    def key_params(self) -> Dict[str, Any]:
        if self.model == "render":
            return {"split": self.split, "corpus": self.corpus, "renderer": self.renderer_name,
                    "px": self.px, "custom_pool": self._custom_pool}
        if self.model == "dir":
            return {"path": str(self.path)}
        return {"px": self.px}

    # --- modes ---------------------------------------------------------------
    def _solid_sheet(self, req: ImageRequest, i: int, views: Sequence[str]) -> bytes:
        """Four distinct coloured squares (one per panel view) on white, 2x2."""
        from blockgen.agentic.sheets import compose_grid
        px = int(self.px or 256)
        tiles = []
        for v in views:
            h = _digest(_req_digest(req).hex(), i, v)
            arr = np.full((px, px, 3), 255, np.uint8)
            lo, hi = px // 5, px - px // 5
            arr[lo:hi, lo:hi] = (h[0] // 2 + 32, h[1] // 2 + 32, h[2] // 2 + 32)
            tiles.append(_png(arr))
        return compose_grid(tiles, 2, 2, gutter=max(8, px // 16), px=2 * px)

    def _solid(self, req: ImageRequest, i: int) -> bytes:
        from PIL import Image, ImageDraw
        grid = parse_grid_hint(req.view)
        if grid is not None:
            return self._solid_sheet(req, i, grid)
        h = _digest(_req_digest(req).hex(), i)
        rgb = (h[0] // 2 + 64, h[1] // 2 + 64, h[2] // 2 + 64)      # never white
        px = int(self.px or 256)
        im = Image.new("RGB", (px, px), rgb)
        draw = ImageDraw.Draw(im)
        words = req.prompt.split()
        lines = [" ".join(words[j:j + 4]) for j in range(0, min(len(words), 16), 4)]
        draw.multiline_text((6, 6), "\n".join(lines + [f"seed {req.seed} #{i}"]),
                            fill=(255, 255, 255))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def _pool(self) -> Sequence[Any]:
        if self._structures is None:
            self._structures = list(_train_pool(self.corpus, self.split))
        if not self._structures:
            raise RuntimeError(f"mock:render: empty pool ({self.corpus} {self.split})")
        return self._structures

    def _render_one(self, structure: Any, view_name: str) -> np.ndarray:
        view = get_view(view_name if view_name in VIEWS else "iso")
        if self.px:
            view = dataclasses.replace(view, px=int(self.px))
        if self._renderer is None:
            # First call picks the renderer: the configured one, then 'flat', then
            # the numpy projection (render.py not landed, or no EGL on this box).
            chain = [self.renderer_name] + [r for r in ("flat",) if r != self.renderer_name]
            for name in chain:
                try:
                    r = make("renderer", name)
                    img = r.render(structure, [view])[0]
                except Exception as exc:  # noqa: BLE001 - KeyError/ImportError/RenderError
                    warnings.warn(f"mock:render: renderer {name!r} unavailable ({exc}); "
                                  f"falling back")
                    continue
                self._renderer, self.renderer_used = r, name
                return np.asarray(img, dtype=np.uint8)
            self._renderer, self.renderer_used = False, "numpy"
        if self._renderer is False:
            return _project(view.apply(structure), view.px)
        return np.asarray(self._renderer.render(structure, [view])[0], dtype=np.uint8)

    def _render(self, req: ImageRequest, i: int) -> bytes:
        pool = self._pool()
        h = _digest(_req_digest(req).hex(), i)
        idx = int.from_bytes(h[:8], "big") % len(pool)
        # A conditioned dream (back / interior / plan of 'the SAME building') must
        # show the same house as its reference, or the oracle multi-view target is
        # four different buildings. Remember which house each output drew.
        for ref in req.refs:
            known = self._drawn.get(hashlib.sha256(ref).hexdigest())
            if known is not None:
                idx = known
                break
        grid = parse_grid_hint(req.view)
        if grid is not None:
            from blockgen.agentic.sheets import compose_grid
            arrs = [self._render_one(pool[idx], v) for v in grid]
            side = int(np.asarray(arrs[0]).shape[0])
            png = compose_grid([_png(a) for a in arrs], 2, 2, gutter=max(8, side // 16),
                               px=2 * side)
        else:
            png = _png(self._render_one(pool[idx], req.view))
        self._drawn[hashlib.sha256(png).hexdigest()] = idx
        return png

    def _dir(self, req: ImageRequest, i: int) -> bytes:
        files = sorted(Path(str(self.path)).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"mock:dir: no PNGs in {self.path}")
        h = _digest(_req_digest(req).hex())
        return files[(int.from_bytes(h[:8], "big") + i) % len(files)].read_bytes()

    def generate(self, req: ImageRequest) -> ImageResult:
        t0 = time.time()
        fn = {"solid": self._solid, "render": self._render, "dir": self._dir}[self.model]
        images = [fn(req, i) for i in range(max(1, req.n))]
        return ImageResult(images=images, model=f"mock-{self.model}", cost_usd=0.0,
                           priced=False, latency_s=time.time() - t0)

    def estimate_usd(self, req: ImageRequest) -> Optional[float]:
        return None

    def supports_refs(self) -> bool:
        return True    # accepted (they change the hash), but not used to draw

    def describe(self) -> str:
        return f"mock:{self.model}"


# ===========================================================================
# Content-addressed cache
# ===========================================================================
def _atomic_write(path: Path, data: Union[bytes, str]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        if isinstance(data, bytes):
            tmp.write_bytes(data)
        else:
            tmp.write_text(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class CachingImageGen:
    """Content-addressed cache around any ImageGen (disk, or memory for mocks).

    Key = ``req.key(inner.name, model_key)`` where ``req`` has had the generator's
    overrides applied (``inner.effective``) and ``model_key`` is the model plus
    any output-affecting settings outside the request (``inner.key_params``), so
    two different images can never share a key. Files: ``<key>.png`` (``<key>_<i>.png``
    for n>1) and ``<key>.json``; the JSON is written LAST, so its presence means a
    complete entry. A hit returns ``cached=True`` with the STORED full price (the
    ledger books it as would_cost, paid 0).

    On a miss: ``read_only`` raises CacheMiss (except for the in-memory cache of a
    free mock, which just generates); with ``spend_gate`` and a paying
    inner generator, :func:`core.require_spend` must pass first. ``mode='off'`` is
    a pass-through that is still gated.
    """

    def __init__(self, inner: ImageGen, cache_dir: Union[str, Path] = DEFAULT_IMAGE_CACHE, *,
                 mode: str = "read_write", spend_gate: bool = True, memory: bool = False) -> None:
        if mode not in CACHE_MODES:
            raise ValueError(f"cache mode must be one of {CACHE_MODES}, got {mode!r}")
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.spend_gate = spend_gate
        self.memory = memory
        self._mem: Dict[str, Tuple[List[bytes], Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    # Delegated protocol attributes.
    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def spends(self) -> bool:
        return bool(getattr(self.inner, "spends", True))

    @property
    def billing(self) -> str:
        return str(getattr(self.inner, "billing", "api"))

    def estimate_usd(self, req: ImageRequest) -> Optional[float]:
        return self.inner.estimate_usd(req)

    def supports_refs(self) -> bool:
        return self.inner.supports_refs()

    def describe(self) -> str:
        return self.inner.describe()

    def unload(self) -> None:
        fn = getattr(self.inner, "unload", None)
        if fn is not None:
            fn()

    def __getattr__(self, item: str) -> Any:   # e.g. renderer_used on mocks
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)

    # --- keying ----------------------------------------------------------------
    def _effective(self, req: ImageRequest) -> ImageRequest:
        fn = getattr(self.inner, "effective", None)
        return fn(req) if fn is not None else req

    def model_key(self) -> str:
        kp = getattr(self.inner, "key_params", None)
        params = kp() if kp is not None else {}
        params = {k: v for k, v in sorted(params.items()) if v is not None}
        if not params:
            return self.inner.model
        return self.inner.model + "?" + json.dumps(params, sort_keys=True, default=str)

    def key(self, req: ImageRequest) -> str:
        return self._effective(req).key(self.inner.name, self.model_key())

    def _paths(self, key: str, n: int) -> Tuple[Path, List[Path]]:
        pngs = [self.cache_dir / (f"{key}.png" if i == 0 else f"{key}_{i}.png")
                for i in range(n)]
        return self.cache_dir / f"{key}.json", pngs

    def peek(self, req: ImageRequest) -> bool:
        """Whether ``req`` is cached (``estimate`` prices hits at $0)."""
        if self.mode == "off":
            return False
        key = self.key(req)
        if self.memory:
            return key in self._mem
        meta, pngs = self._paths(key, max(1, req.n))
        return meta.is_file() and all(p.is_file() for p in pngs)

    def _load(self, key: str, n: int) -> Optional[Tuple[List[bytes], Dict[str, Any]]]:
        if self.memory:
            return self._mem.get(key)
        meta, pngs = self._paths(key, n)
        if not meta.is_file():
            return None
        try:
            blob = json.loads(meta.read_text())
            return [p.read_bytes() for p in pngs], blob
        except (OSError, json.JSONDecodeError):
            return None   # damaged entry: regenerate

    def _store(self, key: str, req: ImageRequest, res: ImageResult) -> None:
        blob = {"model": res.model, "prompt": req.prompt, "cost_usd": res.cost_usd,
                "priced": res.priced, "revised_prompt": res.revised_prompt,
                "n": len(res.images), "backend": self.inner.name, "model_key": self.model_key(),
                "seed": req.seed, "view": req.view, "size": req.size, "quality": req.quality}
        if self.memory:
            self._mem[key] = (list(res.images), blob)
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta, pngs = self._paths(key, len(res.images))
        for p, img in zip(pngs, res.images):
            _atomic_write(p, img)
        _atomic_write(meta, json.dumps(blob, indent=1))

    def generate(self, req: ImageRequest) -> ImageResult:
        if self.mode != "off":
            key = self.key(req)
            hit = self._load(key, max(1, req.n))
            if hit is not None:
                images, blob = hit
                self.hits += 1
                return ImageResult(images=list(images), model=str(blob.get("model", self.model)),
                                   cost_usd=float(blob.get("cost_usd", 0.0)),
                                   priced=bool(blob.get("priced", True)), cached=True,
                                   revised_prompt=str(blob.get("revised_prompt", "")))
            # A memory-cached generator is a free offline mock: like mock LLMs
            # (never cached at all), it cannot spend, so read_only does not refuse it.
            # Otherwise `--mock --cache read_only` would fail on its first dream.
            if self.mode == "read_only" and not self.memory:
                raise CacheMiss(f"{self.describe()}: image not in cache {self.cache_dir} "
                                f"(cache mode read_only; prompt {req.prompt[:60]!r})")
        if self.spend_gate and self.spends:
            require_spend(self.describe())
        self.misses += 1
        res = self.inner.generate(req)
        if self.mode != "off":
            # The EFFECTIVE request: the provenance must name the size/quality used.
            self._store(key, self._effective(req), res)
        return res


def get_image_gen(spec: Spec, *, cache: str = "read_write",
                  cache_dir: Union[str, Path] = DEFAULT_IMAGE_CACHE,
                  spend_gate: bool = True) -> CachingImageGen:
    """``core.make('image', spec)`` wrapped in :class:`CachingImageGen`.

    Mock generators get an in-memory cache instead of the disk one: they are free
    and deterministic, and writing them to the shared dream cache would only fill
    it with placeholders.
    """
    inner = make("image", spec)
    return CachingImageGen(inner, cache_dir, mode=cache, spend_gate=spend_gate,
                           memory=getattr(inner, "billing", "") == "mock")


__all__ = ["CachingImageGen", "DEFAULT_IMAGE_CACHE", "DiffusersImageGen", "IMAGE_PRICES",
           "MockImageGen", "OpenAIImageGen", "REF_SURCHARGE_USD", "get_image_gen",
           "image_price"]
