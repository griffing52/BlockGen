from __future__ import annotations

import importlib
from typing import Optional
import numpy as np

from blockgen.utils.data import Structure

try:
    plt = importlib.import_module("matplotlib.pyplot")
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required for rendering. Install it with `pip install matplotlib`."
    ) from exc


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

    structure = _to_structure(schem)
    if crop_non_air:
        structure = structure.crop_to_non_air()
    if max_dim is not None:
        structure = structure.downsample(max_dim=max_dim)

    # Structure uses XYZ (Minecraft: X horizontal, Y up, Z horizontal).
    # Matplotlib voxels interprets dimensions as plot X/Y/Z, so remap to XZY
    # to place Minecraft Y on rendered Z (vertical) axis.
    occupancy = structure.occupied_mask.transpose(0, 2, 1)
    facecolors = structure.facecolors().transpose(0, 2, 1, 3)

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
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_box_aspect((sx, sz, sy))
    ax.view_init(elev=elev, azim=azim)
    title = "Minecraft Structure"
    if structure.source_path:
        title = f"Minecraft Structure: {structure.source_path.split('/')[-1]}"
    ax.set_title(title)

    if created_axis and show:
        plt.tight_layout()
        plt.show()

    return ax

def render_schem_to_array(
    schem,
    crop_non_air: bool = True,
    max_dim: int | None = 96,
    elev: float = 28,
    azim: float = 42,
    figsize=(4, 4),
):
    """
    Render schematic to a NumPy RGB image array instead of displaying it.
    """

    # Create off-screen figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Reuse your existing renderer
    render_schem(
        schem,
        ax=ax,
        crop_non_air=crop_non_air,
        max_dim=max_dim,
        elev=elev,
        azim=azim,
        show=False,
    )

    # Remove extra padding for tight image
    plt.tight_layout(pad=0)

    # Draw canvas
    fig.canvas.draw()

    # Convert to numpy array
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape((height, width, 3))

    # Clean up (VERY important for large grids)
    plt.close(fig)

    return img