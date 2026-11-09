"""Category labels for the class-conditioned tiers: one prompt word per build.

The ``cat`` tag group of the ``voxels-cat`` tokenizer (``blockgen.corpus.tokenizer``)
conditions a model on what a person would type after ``/gen``: house, castle,
ship, ... This module decides that word for every row the category tiers can use
and writes it, with a split, to a sidecar the tiers merge like dedup and kinds::

    data/minecraft/corpus/labels.parquet   id, cat, cat_src, cat_raw, build_key, cat_split
    data/minecraft/corpus/labels.json      counts per class, source and split; leftovers

Label sources, in the order they are tried for a row (first one that maps wins):

* ``title``    the build's own title through ``RULES`` (GrabCraft titles, world titles)
* ``subcat``   the GrabCraft subcategory (``route``) through ``SUBCAT``, else the
               route as free text through ``RULES`` (text2mc tags, 3D-Craft "house")
* ``tags``     GrabCraft tags (``gc_small_48_meta.json``) through ``RULES``
* ``vlm``      the VLM ``category`` of all_32 / houses_48 (``data/minecraft/labels``)
* ``cache``    a houses cache row nothing else named is a ``house``

``RULES`` is an ordered keyword list (first match wins), so a "tree house" is a
house, a "lighthouse" a tower and an "airship" an aircraft, never a ship. A build
whose smallest edge is at most 2 blocks is flat and becomes ``pixelart`` whatever
its words say; a curated build nothing maps stays ``none`` and still trains, as
``cat:none``.

Rows. Curated builds (all_32, houses_32, houses_48 and the auxiliary GrabCraft 48
cache ``gc_small_48``) appear up to four times, once per cache; ``build_key`` (the
GrabCraft url, or the basename of a 3D-Craft / text2mc path, which moved between
cache builds) and ``canon_hash`` collapse them to one representative, preferring
houses_32 (so the pinned split keeps its own rows), then all_32, houses_48,
gc_small_48. World crops are labelled only from their world's title, only when the
world has at most ``WC_MAX_CROPS`` crops in the whole manifest (a single-build
world) and the title is not a map, server or city (``WC_STOP``).

Split (``cat_split``):

* ``holdout``  in eval-holdout (houses_32 val + test) or a copy of one of its builds
               (same ``build_key`` or ``canon_hash``); never trained on
* ``eval``     the class-balanced eval set: per class ``eval_quota(n)`` builds,
               eval-holdout members first (they are already untrained), then the
               rest by a hash of ``build_key``; never a sft-houses32 row nor a build
               sharing a duplicate cluster with sft-houses32 or the anneal tiers, so
               no existing run trained on an eval build; world crops are never eval
* ``train``    everything else in a class or ``none``
* ``extra``    a house outside houses_32: kept out so the house class is exactly
               the pinned houses_32 train split (the unconditional baseline's data)
* ``dup``      a second copy of a build already represented
* ``drop``     too small (``n_blocks < MIN_BLOCKS``), too big for 48^3, or in a class
               below ``MIN_TRAIN`` that is folded into ``none`` (kept as train)

CLI::

    python -m blockgen.corpus.labels build      # writes labels.parquet + labels.json
    python -m blockgen.corpus.labels show
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from .manifest import CORPUS_DIR, ROOT

LABELS = os.path.join(CORPUS_DIR, "labels.parquet")
LABELS_JSON = os.path.join(CORPUS_DIR, "labels.json")
LABEL_DIR = os.path.join(ROOT, "data", "minecraft", "labels")
CACHE_DIR = os.path.join(ROOT, "data", "minecraft", "cache")
AUX = ["gc_small_48"]
HOLDOUT_TIER = "eval-holdout"
SFT_TIER = "sft-houses32"
#: Tiers existing runs trained on; an eval build may not share a duplicate cluster with them.
SEEN_TIERS = ("anneal-worldcrop", "anneal")

#: The prompt classes, in tag order. ``none`` is the unlabeled class and is always last.
CLASSES: Tuple[str, ...] = ("house", "castle", "tower", "church", "shop", "farm", "ship", "vehicle",
                            "aircraft", "statue", "tree", "pixelart", "none")
MIN_TRAIN = 120            # a class needs this many curated train builds (after its eval draw), else it folds into none
MIN_BLOCKS = 30
WC_MAX_CROPS = 3           # a world with at most this many crops is treated as one build
WC_MIN_BLOCKS = 300
#: A world title with one of these words names a map, a server or a place, not a build.
WC_STOP = re.compile(r"\b(?:map|maps|parkour|adventure|pvp|server|survival|city|town|village|world|spawn|hub|lobby|"
                     r"minigame|mini game|game|games|redstone|skyblock|island|pack|kingdom|land|puzzle|challenge|"
                     r"escape|hunger|arena|factions|creative|seed|base|mod|resource|texture|download|"
                     r"pvp|ctf|survive|empire|realm|series|let'?s play|ep(?:isode)?\s*\d+)\b")

# Ordered: the first rule whose pattern matches the lowercase text wins.
RULES: List[Tuple[str, str]] = [
    ("pixelart", r"pixel ?art|\blogos?\b|\bemoji|\bpixelart"),
    ("house", r"tree ?house|boat ?house|farm ?house|light ?house keeper"),
    ("tower", r"\blight ?houses?\b"),
    ("aircraft", r"\bair ?ships?\b|\bair ?planes?\b|\bplanes?\b|\bjets?\b|helicopter|\bspace ?ships?\b|star ?ships?\b|"
                 r"\bshuttle\b|\brockets?\b|\bufo\b|fighter|\bbomber\b|zeppelin|blimp|air ?balloon|"
                 r"x-wing|tie fighter|star destroyer|\bgunship\b|\bboeing\b|\bairbus\b|\bcessna\b"),
    ("ship", r"\bships?\b|\bboats?\b|\byachts?\b|galleon|submarine|sail ?boat|warship|frigate|\bcanoe\b|"
             r"\bferry\b|\bcaravel\b|\bschooner\b|\bdestroyer\b|\bcruiser\b|\bcarrier\b|battleship|\bbarge\b|"
             r"\btitanic\b|\bkayak\b|\braft\b"),
    ("vehicle", r"\bcars?\b|\btrucks?\b|\bbus\b|\bbuses\b|\btanks?\b(?! farm)|\btrains?\b|locomotive|\bwagons?\b|"
                r"tractor|\bjeep\b|\bvan\b|motor ?bike|motorcycle|bulldozer|forklift|humvee|\bcarriage\b|"
                r"\btrailer\b|railway|\btram\b|excavator|ambulance|limousine|\bsuv\b|\bpickup\b|\bsedan\b|"
                r"\bvehicles?\b|\bfire ?truck\b|\bbattering ram\b|siege engine|\bcatapult\b|\btrebuchet\b|"
                r"\bminecart\b|\bscooter\b|\bbike\b"),
    ("statue", r"statue|\bbust\b|sculpture|\bcharacters?\b|superhero|super hero|creature|\banimals?\b|"
               r"\bgolem\b|pokemon|\bfigure\b|\bdragon\b|\bskull\b|\bhead\b|\bmonument\b|\bsuperman\b|"
               r"\bbatman\b|\bmario\b|\bcreeper\b|\bunicorn\b|\bdinosaur\b|\bt-rex\b|\bmermaid\b"),
    ("tree", r"\btrees?\b|\bents?\b|\bbonsai\b|\bpalm\b|\bwillow\b|\bsakura\b|\bmushroom\b(?! house)|"
             r"giant mushroom"),
    ("church", r"church|cathedral|chapel|\btemples?\b|pagoda|shrine|mosque|monastery|abbey|basilica|"
               r"\bshrines?\b|synagogue|\bstupa\b"),
    ("castle", r"castle|fortress|\bforts?\b|\bkeep\b|citadel|palace|\bwalls?\b|\bgates?\b|gatehouse|"
               r"barracks|outpost|bastion|rampart|stronghold|\bbunker\b|\bgarrison\b"),
    ("tower", r"\btowers?\b|watch ?tower|\bspire\b|minaret|\bturret\b|\bbelfry\b"),
    ("farm", r"\bbarn\b|\bstables?\b|\bfarms?\b|windmill|water ?mill|\bsilo\b|granary|\bcoop\b|\bpens?\b|"
             r"greenhouse|\bmill\b|\bfarming\b"),
    ("shop", r"\bshops?\b|\bstores?\b|\bmarkets?\b|\bstalls?\b|tavern|\binn\b|bakery|restaurant|\bcafe\b|"
             r"\bbar\b|\bpub\b|blacksmith|\bsmith\b|\bforge\b|\bbank\b|\bmall\b|workshop|carpenter|brewery|"
             r"butcher|\bhotel\b|marketplace|\bbistro\b|\bdiner\b|supermarket|\bboutique\b|\bmerchant\b|"
             r"\btrader\b|\bgeneral store\b|\balchemist\b|\btailor\b"),
    ("house", r"\bhouses?\b|\bhomes?\b|cottage|\bcabin\b|\bhuts?\b|\bvilla\b|mansion|\bmanor\b|bungalow|"
              r"apartment|igloo|longhouse|hobbit|residence|\blodge\b|\bshack\b|\bdwelling\b|\bchalet\b|"
              r"\bstarter\b|\bsuburban\b|\btownhouse\b"),
]
_RULES = [(c, re.compile(p)) for c, p in RULES]

#: GrabCraft subcategories that name a class on their own (the url's last segment).
SUBCAT: Dict[str, str] = {
    "planes": "aircraft", "airships": "aircraft", "spaceships": "aircraft", "helicopters": "aircraft",
    "boats": "ship", "historical-ships": "ship", "sailing-ships": "ship", "warships": "ship", "ships": "ship",
    "modern-ships": "ship", "submarines": "ship",
    "other-transportation": "vehicle", "working-vehicles": "vehicle", "emergency-vehicles": "vehicle",
    "family-cars": "vehicle", "sports-cars": "vehicle", "buses": "vehicle", "cars": "vehicle", "trains": "vehicle",
    "trucks": "vehicle", "military-vehicles": "vehicle", "motorcycles": "vehicle",
    "statues": "statue", "superheroes-176": "statue", "superheroes": "statue", "supervillains": "statue",
    "youtubers": "statue", "cartoon-characters-183": "statue", "cartoon-characters": "statue", "cartoons": "statue",
    "video-game-characters-180": "statue", "movie-characters": "statue", "movie-characters-185": "statue",
    "fictional-characters": "statue", "comic-book-characters": "statue", "comic-book-characters-184": "statue",
    "tv-series-characters-179": "statue", "famous-characters-181": "statue", "minecraft-creatures": "statue",
    "wild-animals-1": "statue", "farm-animals-1": "statue", "fictional-animals-1": "statue", "pets": "statue",
    "birds-1": "statue", "insects": "statue", "sea-animals-1": "statue",
    "logos-172": "pixelart", "logos": "pixelart", "emoji-175": "pixelart",
    "medieval-castles": "castle", "castles": "castle",   # military-buildings: by title only (radars, flag posts)
    "towers": "tower", "churches": "church", "temples": "church",
    "farm-buildings": "farm", "restaurants": "shop", "cafes": "shop", "malls": "shop", "hotels": "shop", "shops": "shop",
    "medieval-houses": "house", "modern-houses": "house", "wooden-houses": "house", "stone-houses": "house",
    "brick-houses": "house", "survival-houses": "house", "tree-houses": "house", "quartz-houses": "house",
    "starter-houses": "house",
}
#: Sources whose rows are curated builds (one build per row).
CURATED = ("grabcraft", "3dcraft", "text2mc", "text2mc_schem")
PREFER = {"houses_32": 0, "all_32": 1, "houses_48": 2, "gc_small_48": 3}


def classify_text(text: str) -> Optional[Tuple[str, str]]:
    """``(class, matched words)`` for a title, tag string or VLM category, or None."""
    t = (text or "").lower().replace("_", " ")
    for c, rx in _RULES:
        m = rx.search(t)
        if m:
            return c, m.group(0)
    return None


def classify(title: str = "", route: str = "", tags: Optional[List[str]] = None, vlm: str = "",
             house_cache: bool = False, min_edge: Optional[int] = None) -> Tuple[str, str, str]:
    """``(class, source, raw label)`` for one curated build; ``none`` when nothing maps."""
    out = None
    hit = classify_text(title)
    if hit:
        out = (hit[0], "title", hit[1])
    elif route in SUBCAT:
        out = (SUBCAT[route], "subcat", route)
    elif classify_text(route):                  # a free-text category (text2mc tags, 3D-Craft "house")
        out = (classify_text(route)[0], "subcat", route)
    else:
        hit = classify_text(" | ".join(tags or []))
        if hit:
            out = (hit[0], "tags", hit[1])
        else:
            hit = classify_text(vlm)
            if hit:
                out = (hit[0], "vlm", vlm)
            elif house_cache:
                out = ("house", "cache", route or "house")
    if min_edge is not None and min_edge <= 2 and (out is None or out[0] in ("statue", "pixelart", "tree")):
        return "pixelart", "flat", (out[2] if out else f"min edge {min_edge}")
    return out or ("none", "", route or vlm or "")


def build_key(source: str, world_id: str) -> str:
    """One key per build across caches: the GrabCraft url, else the file's basename (the 3D-Craft
    and text2mc paths moved from data/ to data/minecraft/ between cache builds)."""
    w = str(world_id or "")
    if source == "grabcraft" or w.startswith("http"):
        return w.rstrip("/")
    return f"{source}:{os.path.basename(w.rstrip('/'))}"


def _h(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def eval_quota(n: int) -> int:
    """Eval builds for a class of ``n`` labeled builds: 15%, at least 32, at most 128."""
    return int(min(128, max(32, round(0.15 * n))))


def _vlm_labels() -> Dict[str, str]:
    out = {}
    for cache in ("all_32", "houses_48"):
        p = os.path.join(LABEL_DIR, f"{cache}_labels.json")
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                out[f"cache:{cache}:{int(k[1:])}"] = (v.get("category") or "") if v.get("is_build", True) else ""
    return out


def _gc_tags() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name in AUX:
        p = os.path.join(CACHE_DIR, f"{name}_meta.json")
        if os.path.exists(p):
            for url, m in json.load(open(p)).items():
                out[url.rstrip("/")] = [str(t) for t in m.get("tags") or []]
    return out


def label_frame(df):
    """Label and split a ``tiers._frame(aux=AUX)`` frame; returns the sidecar rows."""
    import numpy as np
    import pandas as pd
    vlm = _vlm_labels()
    tags = _gc_tags()
    hold_ids = set(open(os.path.join(CORPUS_DIR, "tiers", f"{HOLDOUT_TIER}.ids.txt")).read().split())
    sft_ids = set(open(os.path.join(CORPUS_DIR, "tiers", f"{SFT_TIER}.ids.txt")).read().split())
    # duplicate clusters an existing arm trained on (the pinned SFT split, the world-crop pretraining)
    seen = set(sft_ids)
    for t in SEEN_TIERS:
        p = os.path.join(CORPUS_DIR, "tiers", f"{t}.ids.txt")
        if os.path.exists(p):
            seen |= set(open(p).read().split())
    seen_clusters = set(df.cluster[df.id.isin(seen) & (df.cluster >= 0)])

    cur = df[df.source.isin(CURATED) & df.id.str.startswith("cache:")].copy()
    cur["cache"] = cur.id.str.split(":").str[1]
    cur = cur[cur.cache.isin(PREFER)]
    cur["build_key"] = [build_key(s, w) for s, w in zip(cur.source, cur.world_id)]
    rows = []
    for r in cur.itertuples(index=False):
        c, src, raw = classify(r.title, r.route, tags.get(r.build_key), vlm.get(r.id, ""),
                               house_cache=r.cache in ("houses_32", "houses_48"), min_edge=min(r.dx, r.dy, r.dz))
        rows.append((c, src, raw))
    cur["cat"], cur["cat_src"], cur["cat_raw"] = zip(*rows) if rows else ([], [], [])

    # eval-holdout and every copy of its builds, by key or by content
    hold = cur[cur.id.isin(hold_ids)]
    hold_keys, hold_hash = set(hold.build_key), set(hold.canon_hash)
    cur["is_hold"] = cur.id.isin(hold_ids) | cur.build_key.isin(hold_keys) | cur.canon_hash.isin(hold_hash)

    # one representative per build: by key, then by content
    cur["pref"] = cur.cache.map(PREFER)
    cur = cur.sort_values(["pref", "id"]).reset_index(drop=True)
    first_key = ~cur.duplicated("build_key")
    first_hash = ~cur.duplicated("canon_hash")
    cur["cat_split"] = np.where(first_key & first_hash, "", "dup")
    # a houses_32 holdout row is the representative of its build, so copies of it are dups (excluded) too
    rep = cur.cat_split == ""
    too_small = cur.n_blocks < MIN_BLOCKS
    too_big = cur.max_dim > 48
    cur.loc[rep & (too_small | too_big), "cat_split"] = "drop"
    rep = cur.cat_split == ""
    cur.loc[rep & cur.is_hold & ~cur.id.isin(hold_ids), "cat_split"] = "holdout"   # a copy that is not the pinned row
    cur.loc[rep & (cur.cat == "house") & (cur.cache != "houses_32"), "cat_split"] = "extra"

    # classes with too few builds fold into none (before the eval draw)
    live = cur[cur.cat_split == ""]
    n_by = live.groupby("cat").size()
    small = {c for c in CLASSES if c != "none" and (n_by.get(c, 0) - eval_quota(n_by.get(c, 0))) < MIN_TRAIN}
    cur["cat_folded"] = cur.cat.where(~cur.cat.isin(small), "none")

    # the eval draw: per class, holdout members first, then by hash; none is never eval
    live = cur.cat_split == ""
    cur["_h"] = [_h(k) for k in cur.build_key]
    for c in CLASSES:
        if c == "none":
            continue
        m = live & (cur.cat_folded == c)
        n = int(m.sum())
        if n == 0:
            continue
        q = eval_quota(n)
        # never a build the pinned split trains on: sft-houses32 rows stay train
        cand = cur[m & ~cur.id.isin(sft_ids) & ~cur.cluster.isin(seen_clusters)]
        cand = cand.assign(_first=~cand.id.isin(hold_ids)).sort_values(["_first", "_h"])
        cur.loc[cand.index[:q], "cat_split"] = "eval"
    cur.loc[(cur.cat_split == "") & cur.id.isin(hold_ids), "cat_split"] = "holdout"
    cur.loc[cur.cat_split == "", "cat_split"] = "train"
    cur["cat_orig"] = cur.cat
    cur["cat"] = cur.cat_folded

    # world crops: titled single-build worlds, train only
    wc = df[df.source == "worldcrop"]
    per_world = wc.groupby("world_id").size()
    q = wc[(wc.p_built >= 0.9) & ~wc.cut & (wc.kind == "build") & wc.is_rep & (wc.n_blocks >= WC_MIN_BLOCKS)
           & (wc.max_dim <= 48)]
    q = q[q.world_id.map(per_world) <= WC_MAX_CROPS]
    t = q.title.fillna("").str.lower()
    q = q[(t != "") & ~t.str.contains(WC_STOP)]
    lab = [classify_text(x) for x in q.title]
    keep = [h is not None and h[0] not in small for h in lab]
    q = q[keep].copy()
    lab = [h for h, k in zip(lab, keep) if k]
    q["cat"] = [h[0] for h in lab]; q["cat_orig"] = q["cat"]
    q["cat_raw"] = [h[1] for h in lab]
    q["cat_src"] = "wc_title"; q["build_key"] = q.id; q["cat_split"] = "wc_train"

    cols = ["id", "cat", "cat_orig", "cat_src", "cat_raw", "build_key", "cat_split"]
    out = pd.concat([cur[cols], q[cols]], ignore_index=True)
    return out, sorted(small)


def summarize(lab, small: List[str]) -> dict:
    s = {"classes": [c for c in CLASSES if c not in small], "folded_into_none": small, "min_train": MIN_TRAIN,
         "eval_quota": "15% of a class, at least 32, at most 128", "wc_max_crops": WC_MAX_CROPS}
    cur = lab[lab.cat_src != "wc_title"]
    s["by_split"] = cur.groupby("cat_split").size().to_dict()
    for split in ("train", "eval", "holdout", "extra"):
        s[f"{split}_by_class"] = cur[cur.cat_split == split].groupby("cat").size().to_dict()
    s["train_by_class_src"] = {f"{a}/{b}": int(v) for (a, b), v in
                               cur[cur.cat_split == "train"].groupby(["cat", "cat_src"]).size().items()}
    wc = lab[lab.cat_src == "wc_title"]
    s["wc_by_class"] = wc.groupby("cat").size().to_dict()
    lo = cur[(cur.cat == "none") & cur.cat_split.isin(["train"])]
    s["none_raw_top"] = lo.cat_raw.fillna("").value_counts().head(40).to_dict()
    s["folded_raw"] = cur[cur.cat_orig.isin(small)].groupby("cat_orig").size().to_dict()
    return s


def build() -> dict:
    from .tiers import _frame
    df = _frame(aux=AUX, labels=False)
    lab, small = label_frame(df)
    lab.to_parquet(LABELS, index=False)
    s = summarize(lab, small)
    json.dump(s, open(LABELS_JSON, "w"), indent=1)
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build"); sub.add_parser("show")
    a = ap.parse_args(argv)
    s = build() if a.cmd == "build" else json.load(open(LABELS_JSON))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
