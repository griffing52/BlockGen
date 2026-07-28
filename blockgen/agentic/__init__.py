"""Track E — agentic build generation: an LLM writes a build *program*.

Where tracks A-D make a model emit one token per voxel, this track has a frontier
LLM emit a short program in a WorldEdit-flavored command language, which is then
executed onto a voxel canvas. That buys three things the per-voxel tracks cannot
get: builds one to two orders of magnitude larger for the same context, arbitrary
canvas sizes, and text/image conditioning that comes free with the base model.

Layout::

    blockstate.py  modern block names (+ states) -> legacy (id, data)
    canvas.py      the bounded voxel buffer commands write into
    dsl.py         the command language: registry, parser, executor
    providers.py   provider-agnostic LLM access (OpenAI / Gemini / Anthropic / mock)
    prompts.py     system / plan / build / repair / critique prompts
    examples.py    in-context example programs + retrieval hook
    tasks.py       prompt sets (short, detailed, large, real corpus captions)
    agent.py       the loop: plan -> generate -> execute -> repair -> critique
    report.py      run artifacts: metrics, sample sheets, structure caches

Quick start::

    from blockgen.agentic import AgentConfig, build_one
    result = build_one("a small oak cottage with a stone chimney",
                       config=AgentConfig(provider="openai:gpt-5-mini", plan=True))
    result.structure   # a blockgen.utils.data.Structure, like any other track's output

See ``docs/agentic.md`` for the full operator's guide.
"""

from blockgen.agentic.agent import AgentConfig, BuildAgent, BuildResult, build_one
from blockgen.agentic.canvas import Canvas, structure_to_canvas
from blockgen.agentic.dsl import (COMMANDS, ExecutionReport, Program,
                                  command_reference, parse_program, run_program)
from blockgen.agentic.providers import LLMProvider, get_provider
from blockgen.agentic.tasks import load_prompts

__all__ = ["COMMANDS", "AgentConfig", "BuildAgent", "BuildResult", "Canvas",
           "ExecutionReport", "LLMProvider", "Program", "build_one",
           "command_reference", "get_provider", "load_prompts", "parse_program",
           "run_program", "structure_to_canvas"]
