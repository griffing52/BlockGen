from __future__ import annotations

import importlib
from typing import Optional

from blockgen.utils.data import Structure


def _to_structure(schem) -> Structure:
    if isinstance(schem, Structure):
        return schem
    return Structure.from_schematic(schem)


def render_schem(
    schem,
    *,
    ax=None,
    crop_non_air: bool = True,
    max_dim: Optional[int] = 96,
    figsize=(8, 8),
    elev: float = 28,
    azim: float = 42,
    show: bool = True,
):
    """Render a schematic/Structure as a 3D voxel plot.

    Parameters
    ----------
    schem:
        Either a raw schematic object loaded by `nbtschematic` or a `Structure`.
    ax:
        Existing matplotlib 3D axis. If omitted, a new figure is created.
    crop_non_air:
        If True, trims surrounding air before plotting.
    max_dim:
        Optional maximum dimension for display downsampling to keep rendering fast.
    """

    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for render_schem. Install it with `pip install matplotlib`."
        ) from exc

    structure = _to_structure(schem)
    if crop_non_air:
        structure = structure.crop_to_non_air()
    if max_dim is not None:
        structure = structure.downsample(max_dim=max_dim)

    occupancy = structure.occupied_mask
    facecolors = structure.facecolors()

    created_axis = ax is None
    if created_axis:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    ax.voxels(
        occupancy,
        facecolors=facecolors,
        edgecolor=(0.02, 0.02, 0.02, 0.08),
        shade=True,
    )

    sx, sy, sz = structure.shape
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((sx, sy, sz))
    ax.view_init(elev=elev, azim=azim)
    title = "Minecraft Structure"
    if structure.source_path:
        title = f"Minecraft Structure: {structure.source_path.split('/')[-1]}"
    ax.set_title(title)

    if created_axis and show:
        plt.tight_layout()
        plt.show()

    return ax