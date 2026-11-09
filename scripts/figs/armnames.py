"""One display name per arm, matching the macros the paper's text uses.

Figures were labelling arms with the identifiers they carry in the code:
`real_test`, `native_oriented`, `llm_control`. Those are historical, they leak
implementation detail a reader has no use for, and they disagree with the names
the prose uses, so a reader matching a figure to a sentence has to translate.

The names here are the ones `main.tex` defines as macros. Keep the two in step:
if a macro changes, change it here too.
"""

from __future__ import annotations

DISPLAY = {
    # real data and the probes applied to it
    "real_test":                "Real",
    "real@solidify":            "Fill-Interior",
    "real@hollow":              "Hollow-Shell",
    "real@grow":                "Grow-Shell",
    "real_shuffled_materials":  "Material-Shuffle",
    "real@monochrome":          "Monochrome",
    "real@single_mode":         "Single-Mode",
    "train_verbatim":           "Train-Copy",
    "train_copy_noise":         "Train-Copy+Noise",
    # cutaway conditions, shown only against each other
    "real#cut":                 "Real (cutaway)",
    "real@solidify#cut":        "Fill-Interior (cutaway)",
    # procedural baselines and the attack
    "uniform_random":           "Random-Voxels",
    "shell_box":                "Hollow-Box",
    "gabled_house":             "Gabled-House",
    "patchwork":                "Patch-Stitch",
    "pick_n_place":             "Pick-and-Place",
    # submissions
    "canon_oriented":           "Voxel-AR",
    "canon_unoriented":         "Voxel-AR-U",
    "native_oriented":          "Voxel-AR (contaminated)",
    "llm_control":              "LLM-FT",
    "llm_bias":                 "LLM-FT-Geo",
    # external, published generator (notes §66): the extracted largest
    # constructed component is the ranked arm; the whole chunk is a DQ control
    "dreamcubed":               "Dream-Cubed",
    "dreamcubed_raw":           "Dream-Cubed (raw chunk)",
    # decimated arms: present in old scorecards, never shown (see fast.py)
    "real@canon16":             "Decimated 16",
    "real@canon8":              "Decimated 8",
}


def display(arm: str) -> str:
    """Display name for an arm id, or the id itself if it has no entry.

    Falling back to the raw id is deliberate: a new arm should look wrong in a
    figure until someone names it, rather than silently acquiring a prettified
    version of its identifier.
    """
    return DISPLAY.get(arm, arm)


#: Metric and gate identifiers -> the names the paper prints. Same fallback rule
#: as `display`: an unnamed identifier shows up raw so someone notices.
METRIC = {
    "geom_kid":            "V-KID",
    "mv_dino_kid":         "MV-KID",
    "voxel_diversity":     "voxel diversity",
    "dino_nn_percentile":  "retrieval percentile",
    "voxel_twin_rate":     "near-twin rate",
    "solidify":            "Fill-Interior",
}


def metric(key: str) -> str:
    """Display name for a metric / gate identifier, or the id itself."""
    return METRIC.get(key, key)
