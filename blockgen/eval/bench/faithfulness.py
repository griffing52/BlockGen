"""Does the build match the prompt?

**Retrieval leads, CLIPScore supports.** A raw multi-view CLIPScore has no
meaningful scale on Minecraft renders: the cosines compress into a narrow band,
so 0.31 versus 0.29 is unreadable and neither can be compared to zero. Asking
instead whether the sample's own caption outranks 31 distractors gives a
statistic with a known chance floor of 1/32, which is interpretable on its own
and robust to CLIP's absolute miscalibration on out-of-domain images. The
paired-vs-shuffled protocol already used in `scripts/eval_mv_conditioning.py`
and `scripts/sample_conditioned.py` is generalized here.

**Every number gets a real-data ceiling.** House captions are highly
substitutable -- "a two-storey suburban house with a pitched roof" describes
hundreds of builds in this corpus -- so perfect retrieval is not achievable even
in principle, and the reachable ceiling is well below 1.0. Running the identical
protocol on held-out *real* builds with their own captions is what makes an
observed 0.34 readable. Without that anchor the number means nothing.

Caption index is pinned to 0. `houses_32_captions.json` stores four captions per
build and the fourth is a template that embeds the scraped title ("American
Middle Class House 22, a Minecraft brick house build"), which would let a model
be rewarded for matching a name rather than a building.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from blockgen.eval.bench import features as ft
from blockgen.eval.bench import stats
from blockgen.labeling.templates import dominant_materials
from blockgen.utils.data import Structure

CAPTION_ROOT = Path("data/minecraft/labels")
DEFAULT_CAPTION_IDX = 0
DEFAULT_N_DISTRACTORS = 31


def load_captions(corpus: str = "houses_32", root: Path | str = CAPTION_ROOT
                  ) -> Dict[str, List[str]]:
    path = Path(root) / f"{corpus}_captions.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def captions_for_indices(captions: Dict[str, List[str]], indices: Sequence[int],
                         caption_idx: int = DEFAULT_CAPTION_IDX) -> List[str]:
    """Manifest row -> caption. Keys are `h{i:05d}` in manifest order."""
    out = []
    for i in indices:
        caps = captions.get(f"h{int(i):05d}") or []
        out.append(caps[caption_idx] if len(caps) > caption_idx else "")
    return out


def embed_texts(prompts: Sequence[str], backbone: str = "clipL",
                device: str = "cuda", batch: int = 64) -> np.ndarray:
    """L2-normalized text features aligned with the image tower."""
    import torch
    model, proc = ft._load_backbone(backbone, device)
    if not hasattr(model, "get_text_features"):
        raise ValueError(f"backbone {backbone!r} has no text tower")
    tok = getattr(proc, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ft.BACKBONES[backbone][0])
    feats = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            chunk = list(prompts[i:i + batch])
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=77).to(device)
            f = model.get_text_features(**enc)
            feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
    return np.concatenate(feats, 0) if feats else np.zeros((0, 0), np.float32)


def retrieval(img: np.ndarray, txt: np.ndarray, n_distractors: int,
              rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Rank each sample's own caption against `n_distractors` others.

    Distractors are drawn from the same caption bank for every arm and every
    sample, so the difficulty is held constant; the drawn ids are recorded by
    the caller for exact reruns.
    """
    n = len(img)
    if n < 2 or len(txt) != n:
        return {"correct": np.zeros(0), "rank": np.zeros(0)}
    k = min(n_distractors, n - 1)
    correct = np.zeros(n)
    ranks = np.zeros(n)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        pick = rng.choice(others, size=k, replace=False)
        cand = np.concatenate([[i], pick])
        scores = txt[cand] @ img[i]
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0])
        ranks[i] = rank + 1
        correct[i] = float(rank == 0)
    return {"correct": correct, "rank": ranks}


def material_agreement(structures: Sequence[Structure], prompts: Sequence[str],
                       k: int = 5) -> Dict[str, np.ndarray]:
    """Fraction of materials named in the caption that the build actually uses.

    Scored only over captions that name at least one recognizable material; the
    fraction of captions that qualify is reported as `coverage`, because an
    agreement score computed over three captions is not the same claim as one
    computed over sixty.
    """
    hits, scored = [], 0
    for s, prompt in zip(structures, prompts):
        # Tokenize on word boundaries. Substring matching is wrong here and
        # quietly so: "ice" occurs inside "a nice building", "sand" inside
        # "sandwich", "gold" inside "golden", each manufacturing a material the
        # caption never named.
        words = set(_WORD_RE.findall((prompt or "").lower()))
        mentioned = words & _MATERIAL_WORDS
        if not mentioned:
            continue
        # A build's materials contribute the words of their display names
        # ("Oak Wood Plank" -> oak, wood, plank), which is how captions name them.
        present: set = set()
        for m in dominant_materials(s, k=k):
            present.update(w for w in _WORD_RE.findall(m.lower()) if len(w) > 2)
        scored += 1
        hits.append(len(present & mentioned) / len(mentioned))
    return {"agreement": np.asarray(hits, dtype=float),
            "coverage": (scored / len(structures)) if len(structures) else 0.0}


_WORD_RE = re.compile(r"[a-z]+")

_MATERIAL_WORDS = {
    "oak", "spruce", "birch", "jungle", "acacia", "wood", "wooden", "plank",
    "log", "stone", "cobblestone", "brick", "bricks", "sandstone", "quartz",
    "glass", "wool", "clay", "terracotta", "concrete", "snow", "ice", "sand",
    "dirt", "nether", "prismarine", "obsidian", "iron", "gold", "diamond",
}


@dataclass
class FaithfulnessReport:
    retrieval_acc: float
    retrieval_ci: tuple
    chance: float
    retrieval_mrr: float
    mv_clipscore: float
    mv_clipscore_ci: tuple
    material_agreement: float
    material_agreement_ci: tuple
    material_coverage: float
    n: int
    n_distractors: int
    caption_idx: int
    backbone: str
    ceilings: Dict[str, float] = field(default_factory=dict)


def evaluate(
    structures: Sequence[Structure],
    prompts: Sequence[str],
    *,
    view: ft.ViewConfig = ft.ViewConfig(),
    backbone: str = "clipL",
    n_distractors: int = DEFAULT_N_DISTRACTORS,
    caption_idx: int = DEFAULT_CAPTION_IDX,
    device: str = "cuda",
    rng: np.random.Generator | None = None,
    n_boot: int = 1000,
    img_feats: np.ndarray | None = None,
) -> FaithfulnessReport:
    rng = rng or np.random.default_rng(0)
    keep = [i for i, p in enumerate(prompts) if p]
    structs = [structures[i] for i in keep]
    texts = [prompts[i] for i in keep]
    if len(structs) < 2:
        return FaithfulnessReport(
            float("nan"), (float("nan"),) * 2, stats.chance_floor(n_distractors),
            float("nan"), float("nan"), (float("nan"),) * 2, float("nan"),
            (float("nan"),) * 2, 0.0, len(structs), n_distractors, caption_idx,
            backbone)

    if img_feats is None:
        img_feats = ft.pooled(ft.embed_views(structs, view, backbone, device,
                                             verbose=False), "mean")
    else:
        img_feats = img_feats[keep]
    txt = embed_texts(texts, backbone, device)

    ret = retrieval(img_feats, txt, n_distractors, rng)
    clip_scores = 100.0 * np.sum(img_feats * txt, axis=1)
    mat = material_agreement(structs, texts)

    acc = stats.bootstrap_ci(ret["correct"], n_boot=n_boot, rng=rng)
    clip = stats.bootstrap_ci(clip_scores, n_boot=n_boot, rng=rng)
    agree = (stats.bootstrap_ci(mat["agreement"], n_boot=n_boot, rng=rng)
             if mat["agreement"].size else (float("nan"),) * 3)

    return FaithfulnessReport(
        retrieval_acc=acc[0], retrieval_ci=(acc[1], acc[2]),
        chance=stats.chance_floor(min(n_distractors, len(structs) - 1)),
        retrieval_mrr=float(np.mean(1.0 / ret["rank"])) if ret["rank"].size else float("nan"),
        mv_clipscore=clip[0], mv_clipscore_ci=(clip[1], clip[2]),
        material_agreement=agree[0], material_agreement_ci=(agree[1], agree[2]),
        material_coverage=float(mat["coverage"]), n=len(structs),
        n_distractors=n_distractors, caption_idx=caption_idx, backbone=backbone)
