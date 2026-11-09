"""Built-in method functions. Importing this package registers them.

    legacy      BuildAgent itself (byte-identical requests), with suite traces
    classic     BuildAgent's loop on Episode verbs, any action space
    inspect     classic + inspect-and-repair rounds (geometry checks as text feedback)
    dream       imagine -> build -> compare renders with the dream -> revise
    best_of_n   n independent runs of an inner method, keep the best
    dream_vote  pick the most buildable of n dreams, then the dream loop
    blueprint   plan (typed JSON) -> deterministic shell -> rooms one by one -> roof
    floors      plan -> storeys bottom up, each in its own region -> roof
    turnaround  one 2x2 multi-view sheet, split into four dreams, then the dream loop
    interior    (scorer, staged_scoring) offline rooms + furnishing metric

A new method is one decorated function in a module imported here, or in any file
passed with ``--plugin`` (see docs/agentic-suite.md, section 6.1 "A method" and
section 5.4 "Writing a staged variant").
"""

from blockgen.agentic.recipes import classic, dream, inspect, legacy, steps, variants  # noqa: F401
from blockgen.agentic.recipes import (  # noqa: F401
    blueprint, floors, staged_scoring, steps_staged, turnaround,
)

__all__ = ["blueprint", "classic", "dream", "floors", "inspect", "legacy", "staged_scoring", "steps",
           "steps_staged", "turnaround", "variants"]
