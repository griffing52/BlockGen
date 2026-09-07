"""BlockLab — a local instrument for looking at builds, not just scoring them.

Why this exists
---------------
Every number this project produces comes from a batch script that runs once and
prints a table. That is fine for a result and useless for the two things that
actually consume a researcher's time: *looking* at builds, and *deciding* about
them. T21 left a standing lesson -- "look at `samples_*.png` BEFORE writing any
number down" -- and the only tool for that today is rendering a contact sheet and
squinting. There is no way to mark a build bad, no way to leave a note on one, no
way to see which arm produced it, and no way to try a curation threshold without
rebuilding a dataset.

BlockLab is that missing half: a small always-on local app over the artifacts the
repo already writes. It reads; the batch pipeline still owns writing.

Design constraints, and why each one
------------------------------------
**Standard library only.** `http.server` + `sqlite3` + what `blockgen` already
imports. FastAPI lives in `deploy/inference/requirements.txt` and is not in the
main environment; a tool meant to be re-run for months should not acquire a
dependency that can rot. No build step, no npm, no bundler: the pages are plain
HTML that a browser opens as-is.

**It lives outside `blockgen/`.** This is an instrument, not part of the library
under test. Nothing in `blockgen/` may import it. It sits in `tools/` rather than
`deploy/` because `deploy/` is the shipped Minecraft demo -- a mod plus an
inference server with its own dependency file and its own release cadence --
while this never leaves the workstation.

**Renders are serialized and cached.** `renderer.textured` holds a single EGL
context and is not safe to call from two threads; `notes.md` records a context
clash from mixing pyrender backends in one process. So every render goes through
one lock and lands in a content-addressed disk cache. A thumbnail is computed
once, ever.

**Decisions are additive and exportable.** Labels and notes go in a SQLite file
under `outputs/lab/lab.db` and are never written back into a corpus cache. A
curation decision made here is a *proposal*; turning it into a dataset stays the
batch pipeline's job, and the export endpoint is the seam.

Run it
------
    python -m tools.lab                    # http://127.0.0.1:8765
    python -m tools.lab --port 9000 --open

The HTTP contract
-----------------
Frozen here so the pages and the server can be built against one description.
All responses are JSON unless noted. `build_id` is stable across restarts and is
`"<dataset_id>:<index>"`.

    GET  /api/datasets
         -> [{id, name, kind, n, source, note}]     the flat list, sorted by kind
            kind in {"corpus", "raw", "subset", "split", "arm"} (`catalog._KIND_ORDER`,
            which is also the sort order); `source` is a path or a split key.

    GET  /api/tree
         -> {roots: [node], totals: {datasets, roots, builds, collections}}
            node = {id, name, kind, n, source, note, rule, children: [node],
                    sources?, drops?}. The same datasets, nested by DERIVATION,
            with `builds` counting each build once (see `tree.totals`).

    GET  /api/subsets
         -> [{id, dataset_id, name, parent, mode, rule, note, created_at, describe}]

    GET  /api/builds?dataset=<id>&offset=0&limit=60&label=<any|good|bad|unsure|unlabeled>
         -> {total, items: [{build_id, dataset, index, dims: [x,y,z], n_blocks,
                             label, has_note, title, category}]}

    GET  /api/thumb/<build_id>?px=256           -> image/jpeg (cached on disk)
    GET  /api/view/<build_id>/<k>?px=320        -> image/jpeg, view k of four

    GET  /api/build/<build_id>
         -> {build_id, dataset, index, dims, n_blocks, meta,
             features: {...}, geometry: {...}, label, note, views: [url, ...]}

    POST /api/label      {build_id, label}            -> {ok, label}
         label in {"good", "bad", "unsure", null}; null clears.
    POST /api/note       {build_id, text}             -> {ok}
    POST /api/compare    {a, b, winner, ms, tag}      -> {ok}
         `winner` is a build_id or null for "can't tell".

    POST /api/subset/preview  {parent, rule, mode, name?, id?, note?}
         -> {n, parent_n, describe, indices: [int], sample: [row]}
         Resolves a rule WITHOUT saving it. The full index list, not a page of
         it: freezing a branch sends these back as an `indices` rule, and a
         truncated list would freeze a smaller set than the one previewed.
    POST /api/subset          {parent, rule, mode, name?, id?, note?, overwrite?}
         -> the saved subset + {describe, n}
    POST /api/subset/delete   {id}                    -> {deleted}
         Children are left in place and report a missing parent.

    GET  /api/labels?dataset=<id>   -> {counts: {good, bad, unsure, unlabeled},
                                        by_build: {build_id: label}}
    GET  /api/export?what=labels|notes|compares       -> JSON download

    GET  /api/scorecards
         -> [{run, path, when, n_arms, corpus,        <- the original five
              label, note, started, tier, n_submissions, n_controls,
              n_baselines, git_dirty, has_examples}]
            `run` is the directory name and is the only identity; `label` is
            derived, display-only, and deliberately not unique. `git_dirty` is
            null (not false) on a card that predates the flag.

    GET  /api/scorecard/<run>
         -> the parsed scorecard, plus {run_id, path, when, leaderboard,
            head_to_head, arm_names, metrics, label, note, started,
            n_submissions, n_controls, n_baselines,
            compat: {version, read_as, sections, missing, unsupported},
            arms_index: [{name, track, n, n_empty, source, has_prompts, score,
                          status, disqualified, kind, origin, provenance, recipe,
                          source_run_id, structures_sha, examples: [build_id],
                          dataset, model_card}]}
            EVERY per-arm join lands in `arms_index`, never in `arms`: `arms` is
            the object in the process-wide scorecard memo, so writing into it
            would poison every later reader in the process. `arms_index` is a
            fresh list rebuilt per request. `examples` are build_ids the
            /api/thumb route already serves -- that is why a run's example builds
            are an npz beside the scorecard and not a PNG inside it: no route
            serves a file out of a run directory.

    GET  /api/curation/preview?<gate params>
         -> {kept, dropped, reasons: {name: count}, examples: {reason: [build_id]}}

    GET  /api/ontology/catalogs      -> [{id, path, name, domain, n, corpus,
                                          n_builds, n_placements, when}]
    GET  /api/ontology?catalog=<id>&variant=mined|shuffled|stats
         -> {catalog, schema, prompt_fields, parts: [{id, category, support,
             notes, attrs}], prompt: {chars, tokens}, sources}
    GET  /api/ontology/part/<part_id>?catalog=&variant=
         -> {id, category, support, notes, nested, by_source, neighbors, swatch}
    GET  /api/ontology/prompt?catalog=&variant=&fields=a,b
         -> {text, chars, tokens, fields}   the exact system-prompt block
    GET  /api/ontology/swatch/<part_id>               -> image/png (texture tile)

    The ontology routes are read-only and derive `shuffled`/`stats` on the fly;
    catalogs are built by `python -m blockgen.ontology`, never by the lab.

Errors are `{"error": "...", "detail": "..."}` with a 4xx/5xx status. Every
endpoint must work when the underlying artifact is missing -- an empty
`outputs/` is the normal state on a fresh clone, and the pages must render an
empty state rather than a stack trace.
"""

from __future__ import annotations

__all__ = ["DEFAULT_PORT", "LAB_ROOT", "DB_PATH", "THUMB_CACHE"]

from pathlib import Path

DEFAULT_PORT = 8765
#: Everything the lab writes lives here, under the repo's gitignored outputs/.
LAB_ROOT = Path("outputs/lab")
DB_PATH = LAB_ROOT / "lab.db"
THUMB_CACHE = LAB_ROOT / "thumbs"
