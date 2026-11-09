# Faces paper: open experiments (2026-11-09)

Story: six (depth, block) faces are a cheap representation of block exteriors; free decoding is
almost never consistent; cross-face constrained decoding makes consistency exact without training.
Interiors are out of scope (one sentence in Problem and Limitations), not a ceiling or a headline.

Every `\UNMEASURED` / `\TODO` in sec/*.tex maps to a row here. `make tldr` prints the outline.

| # | experiment | why | cost | status |
|---|---|---|---|---|
| 1 | paired card n=256: constrained best-of-4 vs Voxel-AR (canon_oriented and recanon20m) vs Real-Shell | headline "ties voxels on geometry, beats on appearance" currently rests on free best-of-4 (T104) | ~1-2 h CPU/GPU, existing npz | todo |
| 2 | exterior-only evaluation: score every arm (voxel arms projected to shells) against Real-Shell as reference | removes interiors from the comparison by design | ~1 card | decide |
| 3 | dimensional study: id-only single facade, single facade + depth, six faces; Voxel-AR projected down; Face-KID with real-vs-real floor | Sec. 6.4 claim | ~2 days code, ~3 GPU-h | todo |
| 4 | constrained decoding on Indep-Face 5M | how much of the joint gain is consistency alone | ~1 GPU-h | todo |
| 5 | seeds 1-2 for grammar / constrained / constrained best-of-4 | everything is seed 0 | ~3 GPU-h | todo |
| 6 | face order + temperature sweep under constraint | ablation | ~2 GPU-h | optional |
| 7 | entrance-first ordering: rotate each house so a side face that shows a door is decoded first | enterability is the main weakness (0.32 vs 0.73) | retrain 20M D4 (~2-3 GPU-h) | idea |
| 8 | faces at 48^3 (houses_48, 3,088 builds) | quadratic cost argument; largest house 13,442 face cells vs 106,032 voxels | new shard + 20M train | idea, likely post-deadline |

Door statistic (job tmp `doors.py`, D = ids 64, 71, 193-197, first hit along x/z):
houses_32 door visible on >= 1 side 86.1% (train 85.1%), door anywhere 89.9%;
houses_48 84.6% / 88.9%. Requiring a door face drops about 14% of the corpus; reordering
(item 7) drops nothing.

Cell counts (2*(XY+YZ+XZ) vs XYZ over cache boxes): houses_32 median 1,238 vs 2,783, max 5,632 vs
28,672; houses_48 median 1,402 vs 3,360, max 13,442 vs 106,032.
