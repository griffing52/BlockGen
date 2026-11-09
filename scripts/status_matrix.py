#!/usr/bin/env python
"""Generate docs/status.md: one view of every option the pipeline has today.

The page is built from the repo state (tier configs and their built JSON, shard
specs, the trainer's PRESETS and ARCHS, the train configs, the run registry, the
reward registry, the bench protocol) plus ``configs/status_notes.yaml``, which
holds what the code cannot know: legacy tracks, T-numbers, best numbers and
planned items. Where the two disagree the code wins, because the notes only add
text to rows the code produced.

    PYTHONPATH=. python scripts/status_matrix.py                 # writes docs/status.md
    PYTHONPATH=. python scripts/status_matrix.py --date 2026-10-31 --out -
    PYTHONPATH=. python scripts/status_matrix.py --check         # exit 1 if docs/status.md is stale

The date in the header comes from ``--date`` or from the newest registry line;
the script never reads the clock, so a test can regenerate the page byte for
byte. ``--root`` points every repo-state read at another tree (the tests build a
small one); the code registries are always the imported package's.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)
if DEFAULT_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_ROOT)

COMMAND = "PYTHONPATH=. python scripts/status_matrix.py"
EM_DASH = "\u2014"
STATUS_WORDS = ("used", "unused", "planned", "stale")

SECTIONS = [
    "1. Data sources",
    "2. Filters and curation",
    "3. Tiers: pretraining data and SFT sets",
    "4. Token kinds, layouts, augmentations",
    "5. Model architectures",
    "6. Train configs and stages",
    "7. SFT strategies and training mechanics",
    "8. Post-training strategies",
    "9. Controls, ceilings, protocol, metrics",
    "10. Trained runs",
    "11. Open decisions",
]

Row = Dict[str, Any]


# --- small helpers -------------------------------------------------------------------

def clean(s: Any) -> str:
    """Markdown-cell safe text: no em dashes, no pipes, no newlines."""
    t = "" if s is None else str(s)
    t = re.sub(r"\s*" + EM_DASH + r"\s*", ", ", t).replace("|", "/").replace("``", "`")
    return " ".join(t.split())


def fmt_count(n: Any) -> str:
    if n is None:
        return ""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{int(n):,}"


def fmt_num(x: Any, nd: int = 3) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def span(values: Sequence[float], nd: int = 1) -> str:
    lo, hi = min(values), max(values)
    return fmt_num(lo, nd) if fmt_num(lo, nd) == fmt_num(hi, nd) else f"{fmt_num(lo, nd)} to {fmt_num(hi, nd)}"


def check_status(s: str, where: str) -> str:
    s = clean(s)
    if not s.split(" ")[0].split(",")[0] in STATUS_WORDS:
        raise ValueError(f"{where}: status {s!r} must start with one of {STATUS_WORDS}")
    return s


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    n = 0
    for r in rows:
        cells = [clean(c) for c in r]
        cells += [""] * (len(headers) - len(cells))
        out.append("| " + " | ".join(cells) + " |")
        n += 1
    if n == 0:
        out.append("| (none) |" + " |" * (len(headers) - 1))
    return out


def option_table(rows: Iterable[Row]) -> List[str]:
    return table(["Option", "Status", "Where", "Current number"],
                 [(r.get("option"), r.get("status"), r.get("where"), r.get("number")) for r in rows])


def note_rows(notes: Dict[str, Any], key: str, facts: Optional[Dict[str, str]] = None) -> List[Row]:
    """Rows of the notes file under ``key``; a row with a ``key`` of its own gets the
    machine fact of that name appended to its number."""
    out: List[Row] = []
    for i, r in enumerate(notes.get(key) or []):
        row = dict(r)
        row["status"] = check_status(row.get("status", ""), f"status_notes.{key}[{i}]")
        fact = (facts or {}).get(row.get("key", ""))
        if fact:
            row["number"] = "; ".join(x for x in (clean(row.get("number")), fact) if x)
        out.append(row)
    return out


def read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


# --- repo state ----------------------------------------------------------------------

class Repo:
    """Everything the page reads from disk, resolved against one root."""

    def __init__(self, root: str, registry: Optional[str] = None):
        self.root = os.path.abspath(root)
        self.registry_path = registry or self.path("outputs", "registry.jsonl")

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def rel(self, p: str) -> str:
        p = os.path.abspath(p) if not os.path.isabs(p) else p
        return os.path.relpath(p, self.root) if p.startswith(self.root) else p

    # tiers: configs/corpus/<name>.yaml plus data/minecraft/corpus/tiers/<name>.json
    def tiers(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for p in sorted(glob.glob(self.path("configs", "corpus", "*.yaml"))):
            name = os.path.splitext(os.path.basename(p))[0]
            out[name] = {"name": name, "config": read_yaml(p), "config_path": self.rel(p), "json": None}
        for p in sorted(glob.glob(self.path("data", "minecraft", "corpus", "tiers", "*.json"))):
            name = os.path.splitext(os.path.basename(p))[0]
            out.setdefault(name, {"name": name, "config": None, "config_path": None})["json"] = read_json(p)
        return out

    def shards(self) -> List[dict]:
        out = []
        for p in sorted(glob.glob(self.path("data", "minecraft", "corpus", "shards", "*", "*", "spec.json"))):
            spec = read_json(p) or {}
            d = os.path.dirname(p)
            spec = dict(spec)
            spec.setdefault("kind", "voxels")
            spec.setdefault("layout", "per_face" if spec["kind"] == "facades" else "")
            spec.setdefault("augment", "")
            spec["dir"] = self.rel(d)
            spec["tier"] = spec.get("tier") or os.path.basename(os.path.dirname(d))
            spec["tokenizer_json"] = os.path.join(d, "tokenizer.json")
            out.append(spec)
        return out

    def tokenizers(self, shards: List[dict]) -> Dict[str, dict]:
        """hash -> {kind, layout, vocab_size} for every tokenizer.json a shard carries."""
        out: Dict[str, dict] = {}
        try:
            from blockgen.corpus.tokenizer import CorpusTokenizer
        except ImportError:
            return out
        for s in shards:
            if not os.path.exists(s["tokenizer_json"]):
                continue
            try:
                tok = CorpusTokenizer.load(s["tokenizer_json"])
            except Exception:                                   # a foreign or broken artifact
                continue
            out.setdefault(tok.hash, {"kind": tok.kind, "layout": tok.layout, "vocab_size": tok.vocab_size})
        return out

    def train_configs(self) -> Dict[str, dict]:
        """stem -> flattened config (``extends`` resolved, ``description`` kept aside)."""
        d = self.path("configs", "train")
        stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(d, "*.yaml")))
        cache: Dict[str, dict] = {}

        def flat(stem: str, seen: Tuple[str, ...] = ()) -> dict:
            if stem in cache:
                return dict(cache[stem])
            if stem in seen:
                raise ValueError(f"configs/train/{stem}.yaml: extends cycle {seen + (stem,)}")
            raw = read_yaml(os.path.join(d, f"{stem}.yaml"))
            merged: dict = {}
            parents = raw.pop("extends", None) or []
            for parent in ([parents] if isinstance(parents, str) else parents):
                merged.update(flat(parent, seen + (stem,)))
            merged.pop("description", None)
            merged.update(raw)
            cache[stem] = merged
            return dict(merged)

        out = {}
        for stem in stems:
            cfg = flat(stem)
            cfg["_path"] = f"configs/train/{stem}.yaml"
            cfg["_description"] = cfg.pop("description", "")
            out[stem] = cfg
        return out

    def registry(self) -> List[dict]:
        if not os.path.exists(self.registry_path):
            return []
        rows = []
        with open(self.registry_path) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        return [r for r in rows if "run" in r]   # other writers (facade_cascade) append rows in their own schema

    def run_config(self, run: str) -> dict:
        rj = read_json(self.path(run, "run.json")) or {}
        cfg = rj.get("config") if isinstance(rj.get("config"), dict) else {}
        return cfg or {}

    def card(self, run: str, card_field: Optional[str]) -> Optional[dict]:
        for p in ([card_field] if card_field else []) + [os.path.join(run, "bench", "scorecard.json")]:
            full = p if os.path.isabs(p) else self.path(p)
            c = read_json(full)
            if c:
                c["_path"] = self.rel(full)
                return c
        return None

    def bon_reports(self) -> List[dict]:
        out = []
        for p in sorted(glob.glob(self.path("outputs", "run_*", "*", "report.json"))):
            r = read_json(p)
            if isinstance(r, dict) and r.get("method"):
                r = dict(r)
                r["_dir"] = self.rel(os.path.dirname(p))
                r["_run"] = os.path.basename(os.path.dirname(os.path.dirname(p)))
                out.append(r)
        return out

    def worldcrop_months(self) -> Tuple[int, int]:
        """(months with crops.jsonl, months with a manifest parquet)."""
        ingested = len(glob.glob(self.path("data", "minecraft", "worldcrop", "*", "crops.jsonl")))
        manifested = len(glob.glob(self.path("data", "minecraft", "corpus", "manifest", "worldcrop_*.parquet")))
        return ingested, manifested

    def stale_shards(self, shards: List[dict], tiers: Dict[str, dict]) -> List[str]:
        """Shard dirs whose spec.tier_sha256 differs from the tier JSON they were built from."""
        out = []
        for s in shards:
            j = (tiers.get(s["tier"]) or {}).get("json") or {}
            if j.get("sha256") and s.get("tier_sha256") and j["sha256"] != s["tier_sha256"]:
                out.append(s["dir"])
        return out

    def split_sizes(self, split_path: Optional[str]) -> Optional[dict]:
        if not split_path:
            return None
        d = read_json(self.path(split_path))
        if not isinstance(d, dict):
            return None
        return {k: (len(d[k]) if isinstance(d.get(k), (list, dict)) else d.get(k)) for k in ("train", "val", "test")}

    def lab_counts(self) -> Dict[str, Any]:
        db = self.path("outputs", "lab", "lab.db")
        if not os.path.exists(db):
            return {}
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            labels = dict(con.execute("select label, count(*) from labels group by label").fetchall())
            captions = con.execute("select count(*) from captions").fetchone()[0]
            compares = con.execute("select count(*) from compares").fetchone()[0]
            by_tag = dict(con.execute("select tag, count(*) from compares group by tag").fetchall())
            con.close()
        except sqlite3.Error:
            return {}
        return {"labels": labels, "captions": captions, "compares": compares, "compares_by_tag": by_tag}


# --- code registries (imported, read-only) -------------------------------------------

def code_registries() -> Dict[str, Any]:
    from blockgen.train.config import PRESETS, TrainConfig
    from blockgen.train import archs
    from blockgen.corpus import tokenizer as tk
    from blockgen.corpus import facade as fc
    from blockgen.corpus import reps
    from blockgen.eval.bench import protocol
    from blockgen.posttrain import rewards
    from blockgen.train import trainer
    pe_choices: Tuple[str, ...] = ()
    try:
        from blockgen.models.voxel_transformer_ar2 import PE_CHOICES
        pe_choices = tuple(PE_CHOICES)
    except ImportError:
        pass
    return {
        "PRESETS": PRESETS, "TrainConfig": TrainConfig, "ARCHS": archs.ARCHS,
        "KINDS": tk.KINDS, "LAYOUTS": tk.LAYOUTS, "TAG_GROUPS": tk.TAG_GROUPS,
        "FACADE_TAG_GROUPS": tk.FACADE_TAG_GROUPS, "CorpusTokenizer": tk.CorpusTokenizer,
        "face_sets": {"sides": fc.SIDES, "sides_top": fc.SIDES_TOP, "all": fc.ALL_FACES},
        "reps": reps.names(), "protocol": protocol, "rewards": rewards,
        "SELECT_FNS": dict(trainer.SELECT_FNS), "PE_CHOICES": pe_choices,
    }


# --- sections ------------------------------------------------------------------------

def tier_usage(registry: List[dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    trained: Dict[str, int] = {}
    val: Dict[str, int] = {}
    for r in registry:
        for t in r.get("train") or []:
            trained[str(t).partition(":")[0]] = trained.get(str(t).partition(":")[0], 0) + 1
        for t in r.get("val") or []:
            val[str(t)] = val.get(str(t), 0) + 1
    return trained, val


def section_sources(notes: dict, repo: Repo, tiers: Dict[str, dict]) -> List[Row]:
    counts: Dict[str, List[str]] = {}
    for name in ("pretrain-all", "pretrain-clean", "anneal", "sft-houses32", "eval-holdout"):
        j = (tiers.get(name) or {}).get("json") or {}
        for src, n in (j.get("by_source") or {}).items():
            counts.setdefault(src, []).append(f"{name} {fmt_count(n)}")
    facts = {src: "rows: " + ", ".join(v) for src, v in counts.items()}
    ingested, manifested = repo.worldcrop_months()
    if ingested:
        facts["worldcrop"] = f"{manifested} of {ingested} ingested months manifested; " + facts.get("worldcrop", "")
    rows = note_rows(notes, "data_sources", facts)
    known = {r.get("key") for r in rows}
    for src in sorted(counts):
        if src not in known:
            rows.append({"option": src, "status": "used (manifest source without a notes row)",
                         "where": "configs/corpus/*.yaml sources", "number": facts[src]})
    return rows


def section_curation(notes: dict, repo: Repo, tiers: Dict[str, dict], shards: List[dict],
                     configs: Dict[str, dict]) -> List[Row]:
    facts: Dict[str, str] = {}
    split_path = None
    for t in tiers.values():
        c = t.get("config") or {}
        if c.get("holdout_split"):
            split_path = c["holdout_split"]
            break
    sizes = repo.split_sizes(split_path)
    if sizes:
        facts["split"] = f"{split_path}: train {fmt_count(sizes['train'])} / val {fmt_count(sizes['val'])} / test {fmt_count(sizes['test'])}"
    if shards:
        mb = sorted({int(s["max_blocks"]) for s in shards if s.get("max_blocks") is not None})   # derived (rft) shards carry none
        mw = sorted({int(s.get("max_windows", 0)) for s in shards})
        facts["windows"] = f"shards on disk use max_blocks {', '.join(map(str, mb))} and max_windows {', '.join(map(str, mw))}"
    lens = sorted({int(c.get("max_seq_len", 0)) for c in configs.values() if c.get("max_seq_len")})
    sft = [s for s in shards if s["tier"] == "sft-houses32" and s["kind"] == "voxels"]
    q99 = sft[0].get("seq_len", {}).get("0.99") if sft else None
    if lens:
        facts["seq_len"] = "configs use max_seq_len " + ", ".join(map(str, lens)) + \
            (f"; sft-houses32 voxel shard p99 length {fmt_count(q99)}" if q99 else "")
    lab = repo.lab_counts()
    if lab:
        labels = ", ".join(f"{k} {v}" for k, v in sorted(lab["labels"].items()))
        facts["lab_labels"] = f"{sum(lab['labels'].values())} labels ({labels}), {lab['captions']} captions"
    stale = repo.stale_shards(shards, tiers)
    if shards:
        facts["stale_shards"] = (f"{len(shards)} shards on disk, {len(stale)} stale: " + ", ".join(stale)) if stale \
            else f"all {len(shards)} shards on disk match their tier JSON"
    return note_rows(notes, "curation", facts)


def section_tiers(notes: dict, tiers: Dict[str, dict], shards: List[dict], registry: List[dict]) -> List[Row]:
    trained, val = tier_usage(registry)
    tier_notes = notes.get("tiers") or {}
    rows: List[Row] = []
    for name in sorted(tiers, key=lambda n: (tiers[n].get("config") is None, n)):
        t = tiers[name]
        c = t.get("config") or {}
        j = t.get("json") or {}
        mine = [s for s in shards if s["tier"] == name]
        if name in trained:
            status = f"used (train, {plural(trained[name], 'run')})"
        elif name in val:
            status = f"used (val, {plural(val[name], 'run')})"
        elif mine:
            status = "unused (sharded, never trained on)"
        elif j:
            status = "unused (no shard)"
        else:
            status = "planned (config without a built tier)"
        where_bits = []
        if t.get("config_path"):
            where_bits.append(t["config_path"])
        if c.get("holdout_split"):
            where_bits.append(f"split {os.path.basename(c['holdout_split'])} parts {','.join(c.get('holdout_parts') or [])}")
        if c.get("filter"):
            where_bits.append(f"filter `{c['filter']}`")
        if c.get("dedup"):
            where_bits.append("dedup")
        if c.get("exclude"):
            where_bits.append("exclude " + ",".join(c["exclude"]))
        if j.get("kind") == "derived":
            where_bits.append(f"derived from {j.get('source')}")
        num_bits = []
        if j:
            num_bits.append(f"{fmt_count(j.get('n'))} builds")
            if j.get("built"):
                num_bits.append(f"built {j['built']}")
        for s in mine:
            label = s["kind"] + (f"/{s['layout']}" if s["kind"] == "facades" else "") + (f" {s['augment']}" if s["augment"] else "")
            num_bits.append(f"{label} [{s.get('tokenizer', '?')}]: {fmt_count(s.get('n_seqs'))} seqs, {fmt_count(s.get('n_tokens'))} tokens")
        note = (tier_notes.get(name) or {}).get("note")
        if note:
            num_bits.append(note)
        rows.append({"option": name, "status": status, "where": "; ".join(where_bits), "number": "; ".join(num_bits)})
    mixed = [r for r in registry if len(r.get("train") or []) > 1 or any(":" in str(t) for t in r.get("train") or [])]
    rows.append({"option": "Tier mixtures tier:weight", "status": f"used ({plural(len(mixed), 'run')})" if mixed else "unused",
                 "where": "blockgen/train/config.py train_specs; blockgen/train/data.py MixedBatches",
                 "number": (tier_notes.get("mixtures") or {}).get("note", "")})
    return rows


def section_tokens(notes: dict, code: dict, shards: List[dict], toks: Dict[str, dict], registry: List[dict],
                   run_cfgs: Dict[str, dict]) -> List[Row]:
    rows: List[Row] = []
    runs_by_tok: Dict[Tuple[str, str], int] = {}
    for r in registry:
        # rows written before 2026-10-31 carry no augment field; the run's own config has it
        aug = r.get("augment") if r.get("augment") is not None else run_cfgs.get(r["run"], {}).get("augment") or ""
        key = (str(r.get("tokenizer") or ""), str(aug or ""))
        runs_by_tok[key] = runs_by_tok.get(key, 0) + 1
    seen_hashes = set()
    for s in shards:
        h = s.get("tokenizer", "")
        if (h, s["augment"]) in seen_hashes:
            continue
        seen_hashes.add((h, s["augment"]))
        info = toks.get(h, {"kind": s["kind"], "layout": s["layout"], "vocab_size": None})
        label = f"{info['kind']}" + (f" {info['layout']}" if info["kind"] == "facades" else "") + \
                (f", augment {s['augment']}" if s["augment"] else "")
        n_runs = runs_by_tok.get((h, s["augment"]), 0)
        tiers_with = sorted({x["tier"] for x in shards if x.get("tokenizer") == h and x["augment"] == s["augment"]})
        vocab = f"{info['vocab_size']} tokens, " if info.get("vocab_size") else ""
        tnotes = notes.get("tokens_notes") or {}
        rows.append({"option": f"{label} ({vocab}hash {h})",
                     "status": f"used ({plural(n_runs, 'run')})" if n_runs else "unused (sharded)",
                     "where": "blockgen/corpus/tokenizer.py; shards " + ", ".join(tiers_with),
                     "number": tnotes.get(h + ("-" + s["augment"] if s["augment"] else ""), "")})
    for kind in code["KINDS"]:
        layouts = code["LAYOUTS"] if kind == "facades" else ("per_face",)
        for layout in layouts:
            if not any(t["kind"] == kind and (kind != "facades" or t["layout"] == layout) for t in toks.values()):
                rows.append({"option": f"{kind}" + (f" {layout}" if kind == "facades" else "") + " (registered kind, no shard)",
                             "status": "unused (no shard)", "where": "blockgen/corpus/tokenizer.py KINDS, LAYOUTS", "number": ""})
    try:
        ot = code["CorpusTokenizer"].default(oriented=True)
        rows.append({"option": f"voxels oriented ({ot.vocab_size} tokens, hash {ot.hash})",
                     "status": "used" if ot.hash in toks else "unused (no shard)",
                     "where": "blockgen/corpus/tokenizer.py; shards build --oriented", "number": ""})
    except Exception:
        pass
    tags = "; ".join(f"{g}:{{{','.join(v)}}}" for g, v in code["TAG_GROUPS"].items())
    ftags = "; ".join(f"{g}:{{{','.join(v)}}}" for g, v in code["FACADE_TAG_GROUPS"].items())
    rows.append({"option": "Prefix tags", "status": "used", "where": "blockgen/corpus/tokenizer.py TAG_GROUPS",
                 "number": f"{tags}; facades add {ftags}; sample_tags default " +
                           json.dumps(code["TrainConfig"]().sample_tags).replace('"', "")})
    faces_used = sorted({",".join(s.get("faces") or []) for s in shards if s["kind"] == "facades"})
    face_sets = "; ".join(f"{k} = {','.join(v)}" for k, v in code["face_sets"].items())
    used_names = [k for k, v in code["face_sets"].items() if ",".join(v) in faces_used]
    rows.append({"option": "Facade face sets", "status": f"used ({', '.join(used_names) or 'none'} on disk)",
                 "where": "blockgen/corpus/facade.py; shards build --faces", "number": face_sets})
    augs = sorted({(s["kind"], s["layout"], s["augment"]) for s in shards if s["augment"]})
    rows.append({"option": "D4 augmentation (shard sibling <hash>-d4)",
                 "status": "used" if any(runs_by_tok.get((h, "d4")) for h in toks) else ("unused (sharded)" if augs else "unused (no shard)"),
                 "where": "blockgen/corpus/shards.py; TrainConfig.augment",
                 "number": "shards: " + ", ".join(f"{k}/{l}" for k, l, _ in augs) if augs else "no augmented shard"})
    rows.append({"option": "Representation registry", "status": "used", "where": "blockgen/corpus/reps.py",
                 "number": ", ".join(code["reps"]) + "; not linked to KINDS"})
    rows.extend(note_rows(notes, "tokens_legacy"))
    return rows


def preset_of(cfg: dict, presets: Dict[str, dict]) -> Optional[str]:
    if cfg.get("preset"):
        return cfg["preset"]
    for name, p in presets.items():
        if all(cfg.get(k) == v for k, v in (("d_model", p["d_model"]), ("layers", p["layers"]), ("ff", p["ff"]))):
            return name
    return None


def section_archs(notes: dict, code: dict, registry: List[dict], run_cfgs: Dict[str, dict]) -> List[Row]:
    rows: List[Row] = []
    arch_notes = notes.get("arch_notes") or {}
    arch_runs: Dict[str, int] = {}
    for r in registry:
        a = r.get("arch") or "ar2"
        arch_runs[a] = arch_runs.get(a, 0) + 1
    for name, arch in sorted(code["ARCHS"].items()):
        mod = getattr(arch.build, "__module__", "")
        rows.append({"option": f"{name} (trainer arch, track {arch.track})",
                     "status": f"used ({plural(arch_runs.get(name, 0), 'registry run')})" if arch_runs.get(name) else "unused",
                     "where": f"blockgen/train/archs.py register; build in {mod}",
                     "number": arch_notes.get(name, "")})
    pe_used: Dict[str, int] = {}
    for r in registry:
        pe = r.get("pe") or run_cfgs.get(r["run"], {}).get("pe")
        if pe:
            pe_used[pe] = pe_used.get(pe, 0) + 1
    for pe in code["PE_CHOICES"]:
        rows.append({"option": f"ar2 pe={pe}", "status": f"used ({plural(pe_used[pe], 'run')})" if pe in pe_used else "unused",
                     "where": "blockgen/models/voxel_transformer_ar2.py PE_CHOICES; TrainConfig.pe",
                     "number": arch_notes.get(f"pe_{pe}", "")})
    for name, p in code["PRESETS"].items():
        mine = [r for r in registry if preset_of({**run_cfgs.get(r["run"], {}), **({"preset": r["preset"]} if r.get("preset") else {})}, code["PRESETS"]) == name]
        params = sorted({r.get("params") for r in mine if r.get("params")})
        kinds = sorted({r.get("kind") or "" for r in mine})
        num = f"d_model {p['d_model']}, {p['layers']} layers, ff {p['ff']}"
        if params:
            num += "; params " + " / ".join(sorted({fmt_count(x) for x in params}))
        if arch_notes.get(f"preset_{name}"):
            num += "; " + arch_notes[f"preset_{name}"]
        rows.append({"option": f"preset {name}", "status": f"used ({plural(len(mine), 'run')}{', ' + '/'.join(k for k in kinds if k) if any(kinds) else ''})" if mine else "unused",
                     "where": "blockgen/train/config.py PRESETS", "number": num})
    rows.append({"option": "Extra constructor knobs through model: {...}", "status": "unused",
                 "where": "blockgen/train/archs.py constructor_kwargs; TrainConfig.model",
                 "number": "ar2 accepts geo_bias, piece_factors, semantic_embedding; no registry row sets model options "
                 f"(model {{}} recorded on {plural(sum(1 for r in registry if 'model' in r), 'row')}, "
                 f"field absent on {plural(sum(1 for r in registry if 'model' not in r), 'older row')})"
                 if not any(r.get("model") for r in registry) else
                 f"{plural(sum(1 for r in registry if r.get('model')), 'run')} set model options"})
    rows.extend(note_rows(notes, "architectures_legacy"))
    return rows


def match_runs(registry: List[dict], stem: str, cfg: dict, patterns: Sequence[str]) -> List[dict]:
    out = []
    for r in registry:
        if r.get("config") and os.path.normpath(r["config"]) == os.path.normpath(cfg["_path"]):
            out.append(r)
        elif r.get("name") == cfg.get("name") or any(fnmatch.fnmatch(str(r.get("name")), p) for p in patterns):
            out.append(r)
    return out


def first_val(r: dict, key: str = "val_nll") -> Optional[float]:
    v = r.get(key)
    if isinstance(v, dict) and v:
        return list(v.values())[0]
    return v if isinstance(v, (int, float)) else None


def section_configs(notes: dict, code: dict, configs: Dict[str, dict], registry: List[dict]) -> List[Row]:
    rows: List[Row] = []
    cnotes = notes.get("configs") or {}
    for stem in sorted(configs, key=lambda s: (s != "base", s)):
        cfg = configs[stem]
        n = cnotes.get(stem) or {}
        if stem == "base":
            rows.append({"option": "base (defaults every config extends)", "status": "used",
                         "where": cfg["_path"], "number": f"preset {cfg.get('preset')}, lr {cfg.get('lr')}, bench_n {cfg.get('bench_n')}, eval_seqs {cfg.get('eval_seqs')}"})
            continue
        mine = match_runs(registry, stem, cfg, n.get("patterns") or [])
        nlls = [first_val(r) for r in mine if first_val(r) is not None]
        scores = [(r.get("bench") or {}).get("blockscore") for r in mine if (r.get("bench") or {}).get("blockscore") is not None]
        bits = [f"stage {cfg.get('stage', 'pretrain')}", f"train {','.join(cfg.get('train') or code['TrainConfig']().train)}",
                f"kind {cfg.get('kind', 'voxels')}" + (f" {cfg['augment']}" if cfg.get("augment") else ""),
                f"preset {cfg.get('preset')}", f"{fmt_count(cfg.get('steps'))} steps", f"dropout {cfg.get('dropout', 0.0)}",
                f"select {cfg.get('select', 'best')}", f"bench_n {cfg.get('bench_n')}"]
        if cfg.get("init_from"):
            bits.append(f"init_from {cfg['init_from']}")
        num = []
        if mine:
            num.append(plural(len(mine), "run"))
        if nlls:
            num.append(f"best val NLL {fmt_num(min(nlls))}")
        if scores:
            num.append(f"in-run BlockScore {span(scores)}")
        if n.get("t"):
            num.append(n["t"])
        if n.get("note"):
            num.append(n["note"])
        status = f"used ({plural(len(mine), 'run')})" if mine else "unused"
        rows.append({"option": stem, "status": status, "where": f"{cfg['_path']}: " + ", ".join(bits), "number": "; ".join(num)})
    rows.extend(note_rows(notes, "configs_planned"))
    stages_used: Dict[str, int] = {}
    for r in registry:
        stages_used[r.get("stage") or "?"] = stages_used.get(r.get("stage") or "?", 0) + 1
    for stage in ("pretrain", "anneal", "sft", "posttrain"):
        in_cfgs = sorted(s for s, c in configs.items() if c.get("stage", "pretrain") == stage and s != "base")
        rows.append({"option": f"stage: {stage}",
                     "status": f"used ({plural(stages_used[stage], 'run')})" if stage in stages_used else ("unused (configs only)" if in_cfgs else "planned (label only)"),
                     "where": "blockgen/train/config.py TrainConfig.stage (a registry label)",
                     "number": ("configs " + ", ".join(in_cfgs)) if in_cfgs else "no config"})
    rows.extend(note_rows(notes, "stages_extra"))
    return rows


def section_mechanics(notes: dict, code: dict, registry: List[dict], run_cfgs: Dict[str, dict]) -> List[Row]:
    tc = code["TrainConfig"]()
    facts: Dict[str, str] = {}
    n_init = sum(1 for r in registry if r.get("init_from"))
    facts["init_from"] = f"{plural(n_init, 'registry run')} start from a checkpoint"
    partial = sum(1 for r in registry if r.get("init_missing") or r.get("init_unexpected"))
    facts["partial_init"] = f"{plural(partial, 'run')} record a partial match" if partial else "no run has a partial match"
    selects: Dict[str, int] = {}
    for r in registry:
        s = r.get("select") or run_cfgs.get(r["run"], {}).get("select")
        if s:
            selects[s] = selects.get(s, 0) + 1
    facts["select"] = "built in: best, last; hooks: " + (", ".join(sorted(code["SELECT_FNS"])) or "none") + \
        "; runs by select " + (", ".join(f"{k} {v}" for k, v in sorted(selects.items())) or "unrecorded before 2026-10-31") + \
        (f"; no select recorded on {plural(len(registry) - sum(selects.values()), 'older run')}"
         if len(registry) > sum(selects.values()) else "")
    mixed = sum(1 for r in registry if len(r.get("train") or []) > 1 or any(":" in str(t) for t in r.get("train") or []))
    facts["mixtures"] = f"{plural(mixed, 'run')} mix tiers" if mixed else "no registry run mixes tiers"
    finals = sorted({v for r in registry for v in ((r.get("eval_seqs") or {}).get("final") or {}).values()} if registry else set())
    facts["eval"] = f"TrainConfig eval_seqs {tc.eval_seqs}, final_eval_all {tc.final_eval_all}" + \
        (f"; final pass sizes recorded: {', '.join(map(str, finals))}" if finals else "")
    compiled = sum(1 for c in run_cfgs.values() if c.get("compile"))
    facts["compile"] = f"{plural(compiled, 'run')} set compile" if compiled else "no run sets compile"
    facts["sampling"] = f"defaults temperature {tc.temperature}, top_k {tc.top_k}, n_samples {tc.n_samples}, bench_tier {tc.bench_tier}, bench_n {tc.bench_n}"
    return note_rows(notes, "mechanics", facts)


def section_posttrain(notes: dict, code: dict, repo: Repo, registry: List[dict], bon: List[dict]) -> Tuple[List[Row], List[Row]]:
    rw = code["rewards"]
    smoke = set((notes.get("runs") or {}).get("smoke_names") or [])
    real_bon = [b for b in bon if not any(b["_run"].endswith("_" + s) for s in smoke)]
    posttrain_runs = [r for r in registry if r.get("stage") == "posttrain"]
    rows: List[Row] = []
    for i, rung in enumerate(notes.get("posttrain_rungs") or []):
        module = rung.get("module")
        exists = bool(module) and os.path.exists(repo.path(module))
        if not exists:
            status = "planned (no code)"
        elif rung.get("evidence") == "bon" and real_bon:
            status = f"used ({plural(len(real_bon), 'BoN arm')} outside smoke runs)"
        elif rung.get("evidence") == "posttrain_runs" and posttrain_runs:
            status = f"used ({plural(len(posttrain_runs), 'posttrain run')})"
        elif rung.get("evidence") == "bon" and bon:
            status = "unused (smoke BoN only)"
        else:
            status = "unused (implemented, never run)"
        rows.append({"option": rung["option"], "status": status,
                     "where": ((module if exists else "proposed: " + module) + "; " if module else "") + clean(rung.get("where")),
                     "number": rung.get("number", "")})
    lab = repo.lab_counts()
    facts = {}
    if lab:
        tags = ", ".join(f"{k or 'untagged'} {v}" for k, v in sorted(lab["compares_by_tag"].items(), key=lambda kv: str(kv[0])))
        facts["compares"] = f"{lab['compares']} compares in outputs/lab/lab.db ({tags})"
    rows.extend(note_rows(notes, "posttrain_extra", facts))

    reward_rows: List[Row] = []
    trained_on: Dict[str, int] = {}
    for r in registry:
        for s in r.get("trained_on_signals") or []:
            name = rw.parse_signal(s)[0] if hasattr(rw, "parse_signal") else str(s)
            trained_on[name] = trained_on.get(name, 0) + 1
    bon_by_reward: Dict[str, List[dict]] = {}
    for b in bon:
        bon_by_reward.setdefault(str(b.get("reward")), []).append(b)
    for name in sorted(rw.REWARDS):
        if name in getattr(rw, "PILLAR_SIGNALS", ()):
            pillar, note = True, "pillar input: rewards.get refuses it and the placeholder cannot score"
        elif name == "composite":
            pillar, note = False, "inherits pillar status from its parts; reached through --weights"
        else:
            cls = rw.REWARDS[name]
            pillar = bool(getattr(cls, "pillar", False))
            note = ((getattr(cls, "__doc__", "") or "").strip().splitlines()[0] if getattr(cls, "__doc__", None) else "").rstrip(".")
        uses = []
        if name in bon_by_reward:
            real = [b for b in bon_by_reward[name] if not any(b["_run"].endswith("_" + s) for s in smoke)]
            uses.append(plural(len(bon_by_reward[name]), "BoN arm") + (f" ({len(real)} outside smoke runs)" if real != bon_by_reward[name] else ""))
        if name in trained_on:
            uses.append(f"trained on in {plural(trained_on[name], 'run')}")
        if pillar:
            status = "unused (pillar, refused)"
        elif any(not any(b["_run"].endswith("_" + s) for s in smoke) for b in bon_by_reward.get(name, [])) or name in trained_on:
            status = "used"
        else:
            status = "unused" + (" (smoke BoN only)" if name in bon_by_reward else "")
        cli = "CLI" if name in getattr(rw, "CLI_REWARDS", ()) else "python only"
        reward_rows.append({"option": f"{name} (pillar={'yes' if pillar else 'no'}, {cli})", "status": status,
                            "where": "blockgen/posttrain/rewards.py REWARDS", "number": "; ".join([note] + uses) if note or uses else ""})
    for b in bon:
        reward_rows.append({"option": f"BoN arm {b['_dir']}", "status": "used" if not any(b["_run"].endswith("_" + s) for s in smoke) else "used (smoke run)",
                            "where": b.get("arms") or b["_dir"],
                            "number": f"k {b.get('k')}, drawn {b.get('n_drawn', b.get('n_requested'))}, kept {b.get('n_kept', b.get('n'))}, "
                                      f"{b.get('reward')} mean all {fmt_num(b.get('score_all_mean'))}, kept {fmt_num(b.get('score_kept_mean'))}, "
                                      f"train_split {b.get('train_split')}, signals {','.join(b.get('trained_on_signals') or [])}"})
    return rows, reward_rows


def section_controls(notes: dict, code: dict, registry: List[dict]) -> List[Row]:
    pr = code["protocol"]
    rows: List[Row] = []
    for p in pr.OFFICIAL:
        rows.append({"option": f"Protocol {p.id}", "status": "used",
                     "where": "blockgen/eval/bench/protocol.py OFFICIAL",
                     "number": f"corpus {p.corpus}, split {p.split_key}, tier {p.tier}, min_n {p.min_n}, min_ref {p.min_ref}, seed {p.seed}, descriptor {p.descriptor}"})
    rows.append({"option": "Legacy descriptor (cards before 2026-10-23)", "status": "stale",
                 "where": "blockgen/eval/bench/protocol.py LEGACY_DESCRIPTOR",
                 "number": f"{pr.LEGACY_DESCRIPTOR}; those cards are in a different BlockScore unit and unranked until re-scored (T64)"})
    rows.append({"option": "Clean training provenance", "status": "used",
                 "where": "blockgen/eval/bench/protocol.py CLEAN_TRAIN_SPLITS, match_arm",
                 "number": f"report.train_split in {list(pr.CLEAN_TRAIN_SPLITS)}; control arm {pr.CONTROL_ARM}"})
    splits: Dict[str, int] = {}
    for r in registry:
        s = r.get("train_split") or ("canonical" if (r.get("train") or []) == ["sft-houses32"] else ",".join(r.get("train") or []))
        splits[s] = splits.get(s, 0) + 1
    facts = {"board": "registry runs by train_split: " + ", ".join(f"{k} {v}" for k, v in sorted(splits.items()))}
    rows.extend(note_rows(notes, "controls", facts))
    return rows


def run_rows(notes: dict, code: dict, repo: Repo, registry: List[dict], run_cfgs: Dict[str, dict],
             toks: Dict[str, dict]) -> Tuple[List[Sequence[Any]], List[Sequence[Any]]]:
    rn = notes.get("runs") or {}
    smoke = set(rn.get("smoke_names") or [])
    stale = set(rn.get("stale_names") or [])
    run_notes = rn.get("notes") or {}
    latest: Dict[str, str] = {}
    for r in registry:
        latest[r.get("name")] = r["run"]         # later lines win
    main_rows, smoke_rows = [], []
    for r in sorted(registry, key=lambda x: x.get("time") or x["run"], reverse=True):
        base = os.path.basename(r["run"])
        cfg = run_cfgs.get(r["run"], {})
        name = r.get("name")
        preset = r.get("preset") or preset_of(cfg, code["PRESETS"]) or "?"
        tok = str(r.get("tokenizer") or "")
        kind = r.get("kind") or toks.get(tok, {}).get("kind") or cfg.get("kind") or "?"
        layout = r.get("layout") or (toks.get(tok, {}).get("layout") if kind == "facades" else "") or ""
        augment = r.get("augment") if r.get("augment") is not None else cfg.get("augment") or ""
        model = f"{r.get('arch') or 'ar2'} {preset}" + (f" ({fmt_count(r.get('params'))})" if r.get("params") else "")
        data = f"{kind}" + (f"/{layout}" if layout and kind == "facades" else "") + (f" {augment}" if augment else "") + \
               f"; train {','.join(r.get('train') or [])}; {fmt_count(r.get('steps'))} steps" + \
               (f"; dropout {cfg.get('dropout')}" if cfg.get("dropout") is not None else "") + \
               (f"; lr {cfg.get('lr')}" if cfg.get("lr") is not None else "")
        src = r.get("init_from")
        src = src[-1] if isinstance(src, list) and src else src     # an adapter stack lists every layer; the last is the run's own
        init = os.path.basename(os.path.dirname(src)) if isinstance(src, str) and src else ""
        nll = fmt_num(first_val(r))
        if first_val(r, "val_nll_all") is not None:
            nll += f" (all {fmt_num(first_val(r, 'val_nll_all'))})"
        sel = r.get("selected_step")
        if sel is not None:
            nll += f" @{sel}"
        bench = r.get("bench") or {}
        card = repo.card(r["run"], r.get("card"))
        n = None
        tier = bench.get("tier")
        if card:
            arm = (card.get("arms") or {}).get(name) or {}
            n = (arm.get("meta") or {}).get("n")
            tier = tier or (card.get("run") or {}).get("tier")
        n = n or cfg.get("bench_n")
        if bench.get("blockscore") is not None:
            score = f"{fmt_num(bench['blockscore'], 1)} ({tier or 'fast'}, n={n})"
        elif bench.get("status"):
            score = f"{bench['status']} ({tier or 'fast'}, n={n})"
        else:
            score = "none"
        card_cell = card["_path"].replace(r["run"].rstrip("/") + "/", "") if card else "none"
        if name in smoke:
            status = "smoke"
        elif latest.get(name) != r["run"]:
            status = "superseded"
        elif name in stale:
            status = "stale"
        else:
            status = "current"
        if not os.path.isdir(repo.path(r["run"])):
            status += " (dir missing)"
        row = (base, f"{name} / {r.get('stage')}", model, data, init, nll, score, card_cell, status, run_notes.get(base, ""))
        (smoke_rows if name in smoke else main_rows).append(row)
    return main_rows, smoke_rows


# --- page ----------------------------------------------------------------------------

def build_page(repo: Repo, notes: dict, date: str) -> str:
    code = code_registries()
    tiers = repo.tiers()
    shards = repo.shards()
    toks = repo.tokenizers(shards)
    configs = repo.train_configs()
    registry = repo.registry()
    run_cfgs = {r["run"]: repo.run_config(r["run"]) for r in registry}
    bon = repo.bon_reports()

    L: List[str] = []
    L += ["# Status matrix", "",
          f"Generated by `{COMMAND}` on {date} from the repo state and `configs/status_notes.yaml`. "
          "Do not edit this file; edit the notes file for facts the code cannot know and rerun the command. "
          f"`{COMMAND} --check` exits 1 when this file no longer matches the repo.", "",
          "Status words: **used** = implemented and exercised in a recorded run; **unused** = implemented, never run; "
          "**planned** = no code; **stale** = exists but superseded or broken. A parenthesis after the word says how the "
          "generator decided. Numbers carry their T-number (results.md) or run directory.", "",
          "Machine-read sources: `configs/corpus/*.yaml`, `data/minecraft/corpus/tiers/*.json`, "
          "`data/minecraft/corpus/shards/*/*/spec.json` and `tokenizer.json`, `blockgen/train/config.py` PRESETS, "
          "`blockgen/train/archs.py` ARCHS, `configs/train/*.yaml`, `outputs/registry.jsonl` plus each run's "
          "`run.json` and `bench/scorecard.json`, `outputs/run_*/*/report.json` (best-of-N arms), "
          "`blockgen/posttrain/rewards.py` REWARDS, `blockgen/eval/bench/protocol.py`, `outputs/lab/lab.db` (read-only).", ""]
    n_runs = len(registry)
    n_cards = sum(1 for r in registry if repo.card(r["run"], r.get("card")))
    L += ["## At a glance", "",
          f"- {len([t for t in tiers.values() if t.get('json')])} built tiers, {len(shards)} shards, {len(toks)} tokenizer hashes on disk.",
          f"- {len(code['ARCHS'])} trainer arch(s) ({', '.join(sorted(code['ARCHS']))}), {len(code['PRESETS'])} presets, {len(configs) - (1 if 'base' in configs else 0)} train configs.",
          f"- {n_runs} registry runs, {n_cards} with an in-run card; {len(bon)} best-of-N arm(s) on disk.",
          f"- {len(code['rewards'].REWARDS)} rewards registered, {len(code['rewards'].PILLAR_SIGNALS)} of them pillar placeholders that are refused; "
          f"{plural(len(code['rewards'].CLI_REWARDS), 'reward')} reachable from the command line ({', '.join(code['rewards'].CLI_REWARDS)}).",
          ""]

    L += [f"## {SECTIONS[0]}", ""] + option_table(section_sources(notes, repo, tiers)) + [""]
    L += [f"## {SECTIONS[1]}", ""] + option_table(section_curation(notes, repo, tiers, shards, configs)) + [""]
    L += [f"## {SECTIONS[2]}", "",
          "Status is read from the registry: a tier is used when a run trained or validated on it. "
          "Counts are the tier JSON's builds and each shard's sequences and tokens.", ""]
    L += option_table(section_tiers(notes, tiers, shards, registry)) + [""]
    L += [f"## {SECTIONS[3]}", ""] + option_table(section_tokens(notes, code, shards, toks, registry, run_cfgs)) + [""]
    L += [f"## {SECTIONS[4]}", "",
          "The first rows are the trainer's registry and its knobs; the rest are the legacy tracks, which train "
          "through their own scripts and reach the bench as sample caches.", ""]
    L += option_table(section_archs(notes, code, registry, run_cfgs)) + [""]
    L += [f"## {SECTIONS[5]}", "",
          "One row per `configs/train/*.yaml` with its resolved settings and the registry runs it produced "
          "(matched by `config`, `name` or the name patterns in the notes file). Stage rows count registry runs by label.", ""]
    L += option_table(section_configs(notes, code, configs, registry)) + [""]
    L += [f"## {SECTIONS[6]}", ""] + option_table(section_mechanics(notes, code, registry, run_cfgs)) + [""]
    rungs, reward_rows = section_posttrain(notes, code, repo, registry, bon)
    L += [f"## {SECTIONS[7]}", "",
          "The ladder, cheapest rung first. A rung is planned while its module is missing (its path is then marked "
          "`proposed:`), unused once the module exists, used once a best-of-N arm outside a smoke run or a "
          "`stage: posttrain` registry run exists. BoN arms report the draw and the kept count: `mean all` is the reward "
          "over every drawn sample, `kept` over the survivors.", ""]
    L += option_table(rungs) + ["", "### Rewards", "",
                                "`pillar=yes` marks a BlockScore pillar input; `rewards.get` refuses it and the placeholder cannot score. "
                                "Best-of-N arms on disk are listed after the rewards.", ""]
    L += option_table(reward_rows) + [""]
    L += [f"## {SECTIONS[8]}", ""] + option_table(section_controls(notes, code, registry)) + [""]
    main_rows, smoke_rows = run_rows(notes, code, repo, registry, run_cfgs, toks)
    headers = ["Run", "Name / stage", "Model", "Data", "init_from", "val NLL (selected; all)", "BlockScore (tier, n)", "Card", "Status", "Note"]
    L += [f"## {SECTIONS[9]}", "",
          "Every `outputs/registry.jsonl` line, newest first. `current` is the latest run of its name, `superseded` an "
          "earlier one, `stale` a run the notes file marks as superseded by a later protocol. val NLL is the selected "
          "checkpoint on the periodic holdout subset, `all` the whole-tier pass where recorded, `@step` the selected step. "
          + clean((notes.get("runs") or {}).get("blockscore_note", "")), ""]
    L += table(headers, main_rows) + ["", "### Wiring checks", "",
                                       "Smoke and replay runs (names listed under `runs.smoke_names` in the notes file).", ""]
    L += table(headers, smoke_rows) + [""]
    L += [f"## {SECTIONS[10]}", ""] + option_table(note_rows(notes, "open_decisions")) + [""]
    text = "\n".join(L)
    if EM_DASH in text:
        raise ValueError("an em dash reached the page")
    return text


def registry_date(repo: Repo) -> Optional[str]:
    rows = repo.registry()
    if not rows:
        return None
    t = str(rows[-1].get("time") or "")
    return t[:10] or None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="repo root to read state from")
    ap.add_argument("--notes", default=None, help="hand-maintained facts (default <root>/configs/status_notes.yaml)")
    ap.add_argument("--registry", default=None, help="registry jsonl (default <root>/outputs/registry.jsonl)")
    ap.add_argument("--out", default=None, help="output path (default <root>/docs/status.md; '-' prints)")
    ap.add_argument("--date", default=None, help="date for the header; default = the newest registry line's date")
    ap.add_argument("--check", action="store_true", help="exit 1 when the output differs from the file at --out")
    a = ap.parse_args(argv)
    repo = Repo(a.root, a.registry)
    notes = read_yaml(a.notes or repo.path("configs", "status_notes.yaml"))
    date = a.date or registry_date(repo) or "unknown date"
    page = build_page(repo, notes, date)
    out = a.out or repo.path("docs", "status.md")
    if out == "-":
        sys.stdout.write(page)
        return 0
    if a.check:
        have = open(out).read() if os.path.exists(out) else ""
        if have != page:
            print(f"{out} is stale; rerun {COMMAND}", file=sys.stderr)
            return 1
        print(f"{out} is current")
        return 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(page)
    print(f"wrote {out} ({len(page.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
