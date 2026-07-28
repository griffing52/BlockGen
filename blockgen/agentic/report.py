"""Run artifacts for the agentic track: files, metrics, sample sheets.

One place decides what an agentic run leaves on disk, so the single-prompt CLI
(``scripts/run_agentic.py``) and the battery (``blockgen/experiments_agentic.py``)
produce the same layout::

    outputs/run_<stamp>_<name>/
      config.json                 the AgentConfig + arm definitions
      metrics.json                per-build metrics + per-arm aggregates
      summary.md                  the table you actually read
      samples.png                 TEXTURED render of every build, titled by prompt
      <arm>_32.npz + _manifest    builds in the standard structure-cache format
      builds/<arm>/<i>_*.txt|json program, plan, transcript, execution report

The ``.npz`` is deliberately the *same* format ``blockgen.curation.houses``
writes, so agentic builds load with ``load_structures_from_cache`` and drop
straight into the existing novelty / perceptual / render tooling — an agentic
sample and an AR sample are then comparable without a converter.

Rendering follows the repo rule that every run renders its samples
(``blockgen/eval/cond_render.textured_prompt_grid``), and degrades to a note in
the summary when no GPU/EGL is available rather than failing the run.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from blockgen.agentic.agent import BuildResult

# Metrics averaged into the per-arm row of summary.md, in display order.
AGGREGATE_KEYS = ("blocks", "n_components", "largest_component_frac", "volume_fill",
                  "n_block_types", "n_commands", "command_success_rate",
                  "blocks_per_command", "n_failed_commands", "clipped_writes",
                  "completion_tokens", "cost_usd", "elapsed_s")


def save_build(result: BuildResult, out_dir: Path, index: int) -> Path:
    """Write one build's program, plan, transcript and report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:03d}"
    (out_dir / f"{stem}_program.txt").write_text(result.program_text)
    if result.plan_text:
        (out_dir / f"{stem}_plan.txt").write_text(result.plan_text)
    (out_dir / f"{stem}_build.json").write_text(json.dumps(result.to_dict(), indent=1))
    (out_dir / f"{stem}_transcript.json").write_text(
        json.dumps(result.transcript, indent=1))
    return out_dir / f"{stem}_build.json"


def save_structures(results: Sequence[BuildResult], run_dir: Path, arm: str,
                    max_dim: int = 64) -> Optional[str]:
    """Save builds in the standard structure-cache format (``load_structures_from_cache``).

    The prompt travels in the manifest's ``title`` field, so a later re-render or
    eval pass can title each build with what it was asked to build.
    """
    from blockgen.curation.houses import save_house_cache
    keep = [r for r in results if r.blocks > 0]
    if not keep:
        return None
    for r in keep:
        r.structure.metadata.setdefault("corpus", "agentic")
        r.structure.metadata["title"] = r.description
        r.structure.metadata.setdefault("category", arm)
    return save_house_cache(
        [r.structure for r in keep], max_dim, cache_dir=str(run_dir), name=arm,
        report={"arm": arm, "n_requested": len(results), "n_nonempty": len(keep)})


def render_samples(results: Sequence[BuildResult], out_path: Path, *,
                   suptitle: str = "", cols: int = 4, px: int = 384) -> Optional[Path]:
    """Textured sample sheet titled with each prompt; ``None`` if rendering fails."""
    keep = [r for r in results if r.blocks > 0]
    if not keep:
        return None
    try:
        from blockgen.eval.cond_render import textured_prompt_grid
        return textured_prompt_grid([r.structure for r in keep],
                                    [r.description for r in keep], out_path,
                                    cols=max(1, min(cols, len(keep))), px=px,
                                    suptitle=suptitle)
    except Exception as exc:  # noqa: BLE001 - a headless box must still finish the run
        print(f"[render] sample sheet unavailable: {type(exc).__name__}: {exc}",
              flush=True)
        return None


def aggregate(results: Sequence[BuildResult]) -> Dict[str, Any]:
    """Mean of each aggregate metric over non-failed builds, plus totals."""
    rows = [r.metrics() for r in results]
    out: Dict[str, Any] = {"n_builds": len(rows),
                           "n_empty": sum(1 for r in rows if r["blocks"] == 0)}
    for key in AGGREGATE_KEYS:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        out[key] = round(statistics.fmean(vals), 4) if vals else None
    out["total_cost_usd"] = round(sum(r["cost_usd"] for r in rows), 4)
    out["total_elapsed_s"] = round(sum(r["elapsed_s"] for r in rows), 1)
    # "Coherent" = one connected component holding >=90% of the blocks. Same spirit
    # as the validity gate the neural tracks report, so the numbers are comparable.
    coherent = [r for r in rows if r.get("largest_component_frac") is not None]
    out["coherence_rate"] = round(
        sum(1 for r in coherent if r["largest_component_frac"] >= 0.9)
        / max(1, len(coherent)), 4) if coherent else None
    return out


def summary_table(arms: Dict[str, Dict[str, Any]],
                  keys: Sequence[str] = ("n_builds", "n_empty", "blocks",
                                         "n_components", "coherence_rate",
                                         "command_success_rate", "n_commands",
                                         "blocks_per_command", "completion_tokens",
                                         "total_cost_usd")) -> str:
    """Markdown table, one row per arm."""
    header = "| arm | " + " | ".join(keys) + " |"
    rule = "|" + "---|" * (len(keys) + 1)
    lines = [header, rule]
    for name, agg in arms.items():
        cells = []
        for k in keys:
            v = agg.get(k)
            cells.append("—" if v is None else (f"{v:.3f}" if isinstance(v, float)
                                                else str(v)))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_run(run_dir: Path, *, arms: Dict[str, List[BuildResult]],
              config: Dict[str, Any], title: str = "agentic run",
              max_dim: int = 64, render: bool = True) -> Dict[str, Any]:
    """Write every artifact for a run and return the metrics dict."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1, default=str))

    metrics: Dict[str, Any] = {"arms": {}, "builds": {}}
    for arm, results in arms.items():
        for i, r in enumerate(results):
            save_build(r, run_dir / "builds" / arm, i)
        save_structures(results, run_dir, arm, max_dim=max_dim)
        metrics["arms"][arm] = aggregate(results)
        metrics["builds"][arm] = [r.metrics() for r in results]
        if render:
            render_samples(results, run_dir / f"samples_{arm}.png",
                           suptitle=f"{title} — {arm}")

    if render and len(arms) > 1:
        # One combined sheet so arms are comparable side by side at a glance.
        flat = [r for results in arms.values() for r in results]
        render_samples(flat, run_dir / "samples.png", suptitle=title)
    elif render and len(arms) == 1:
        only = next(iter(arms))
        src = run_dir / f"samples_{only}.png"
        if src.exists():
            (run_dir / "samples.png").write_bytes(src.read_bytes())

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    summary = [f"# {title}", "", summary_table(metrics["arms"]), "",
               "## Config", "", "```json", json.dumps(config, indent=1, default=str),
               "```", "", "## Prompts", ""]
    for arm, results in arms.items():
        summary.append(f"### {arm}")
        for i, r in enumerate(results):
            m = r.metrics()
            summary.append(
                f"- `{i:03d}` **{r.description}** — {m['blocks']} blocks, "
                f"{m['n_commands']} commands, {m['n_failed_commands']} failed"
                + (f", ERROR: {m['error']}" if m["error"] else ""))
        summary.append("")
    (run_dir / "summary.md").write_text("\n".join(summary))
    return metrics


__all__ = ["aggregate", "render_samples", "save_build", "save_structures",
           "summary_table", "write_run"]
