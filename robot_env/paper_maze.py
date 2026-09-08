"""The 2D point maze of Trott et al. (NeurIPS 2019), recovered from the paper's Figure 3.

WHY THIS IS TRANSCRIBED AND NOT IMPORTED
----------------------------------------
Experiment 4's brief asks to reuse "the existing Point Maze implementation". There isn't
one. `~/projects/ppo-nav/ppo_sr/mazes.py` carries three MuJoCo `PointMaze` layouts
(EASY / MEDIUM / HARD, 5x8 and 7x8 cells) written deliberately NOT to be the stock maps,
for a different study. None of them is the paper's maze, which is 10x10.

The paper's own description (NeurIPS supplement, "2D point maze navigation"; the arXiv v1
has no appendix) is:

    "The 2D point maze is implemented in a 10x10 environment (arbitrary units) consisting
     of an array of pseudo-randomly connected 1x1 squares. The construction of the maze
     ensures that all squares are connected to one another by exactly one path. This is a
     continuous environment. The agent sees as input its 2D coordinates and well as the 2D
     goal coordinates, which are always somewhere near the top right corner of the maze.
     The agent takes an action in a 2D space that controls the direction and magnitude of
     the step it takes"

The layout itself is one sample from their generator and the seed is not published, so the
only way to get THE maze rather than a lookalike is to read it off Figure 3. That is what
the arrays below are: the figure rendered at 600 dpi, thresholded, and each of the 180
interior grid edges tested for a wall.

HOW WE KNOW THE TRANSCRIPTION IS RIGHT
--------------------------------------
The paper states the maze is perfect -- "all squares are connected to one another by
exactly one path". That is a spanning tree over the 100 cells, which pins THREE numbers
simultaneously: exactly 99 open passages, exactly 1 connected component, and 0 cycles.

`assert_perfect_maze()` below re-checks all three at import. A transcription with even one
wall wrong fails at least one of them: a missing wall creates a cycle, an extra wall
disconnects a cell. Passing all three is therefore strong evidence the read is exact, not
merely plausible -- which matters because a subtly wrong maze would silently change the
difficulty of the task the whole experiment measures.

The first attempt sampled each edge at its nominal pixel position and found 118 passages
with 19 cycles -- 19 walls missed to sub-pixel grid drift. Searching a +-6px window
perpendicular to each edge fixed it exactly.

GEOMETRY AND ORIENTATION
------------------------
Row 0 of these arrays is the TOP of the figure. World coordinates put y=0 at the BOTTOM,
so cell (row, col) spans x in [col, col+1] and y in [N-1-row, N-row]. That makes:

    start  cell (9, 0) -> world (0.5, 0.5)   bottom-left, the paper's blue square
    goal   cell (0, 9) -> world (9.5, 9.5)   top-right,   the paper's red square

Measured on this layout: the unique start->goal route is 22 moves against a 12.73-unit
straight line, a 1.73x detour. At the paper's 50-step horizon that forces a mean step of
0.44 world units, which is what sizes `MAX_STEP` in `point_maze.py`.
"""
from __future__ import annotations

import numpy as np

#: Cells per side. The paper's "10x10 environment ... of 1x1 squares".
N_CELLS = 10

#: Interior VERTICAL walls. `VWALL[r][c]` is the wall between cell (r, c) and (r, c+1).
#: Row 0 is the top of the figure. Shape (10, 9).
VWALL_ROWS = [
    "100000010",
    "011010101",
    "101010110",
    "011111101",
    "101010101",
    "000001100",
    "111111011",
    "001101000",
    "110001010",
    "001010001",
]

#: Interior HORIZONTAL walls. `HWALL[r][c]` is the wall between cell (r, c) and (r+1, c).
#: Shape (9, 10).
HWALL_ROWS = [
    "0101111100",
    "1000101011",
    "0100110001",
    "1000000010",
    "0010110010",
    "1011100010",
    "0000010101",
    "0010011011",
    "0100100100",
]

VWALL = np.array([[ch == "1" for ch in row] for row in VWALL_ROWS], dtype=bool)
HWALL = np.array([[ch == "1" for ch in row] for row in HWALL_ROWS], dtype=bool)

#: The start and goal CELLS, and their centres. The environment samples a point inside each
#: cell per episode (the paper's own setup); `START_XY` / `GOAL_XY` are the exact centres,
#: used only by `start_goal_mode="fixed"` and by tests that need exact geometry.
START_CELL = (N_CELLS - 1, 0)          # bottom-left
GOAL_CELL = (0, N_CELLS - 1)           # top-right
START_XY = (0.5, 0.5)
GOAL_XY = (N_CELLS - 0.5, N_CELLS - 0.5)


def cell_bounds(row, col, margin=0.0):
    """Axis-aligned bounds of cell (row, col) as (x_low, x_high, y_low, y_high).

    `margin` insets the box so a sampled point cannot land exactly on a wall, where the
    swept-collision test would immediately register a contact on step one.
    """
    x_low, x_high = col + margin, col + 1.0 - margin
    y_bottom = (N_CELLS - 1 - row)
    y_low, y_high = y_bottom + margin, y_bottom + 1.0 - margin
    return x_low, x_high, y_low, y_high


def cell_to_xy(row, col):
    """Centre of cell (row, col) in world coordinates, y up."""
    return (col + 0.5, (N_CELLS - 1 - row) + 0.5)


def open_neighbours(row, col):
    """Cells reachable from (row, col) in one move, i.e. not separated by a wall."""
    if col < N_CELLS - 1 and not VWALL[row, col]:
        yield row, col + 1
    if col > 0 and not VWALL[row, col - 1]:
        yield row, col - 1
    if row < N_CELLS - 1 and not HWALL[row, col]:
        yield row + 1, col
    if row > 0 and not HWALL[row - 1, col]:
        yield row - 1, col


def maze_statistics():
    """Passage count, component count and cycle count -- the perfect-maze invariants."""
    parent = list(range(N_CELLS * N_CELLS))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    passages = cycles = 0
    for row in range(N_CELLS):
        for col in range(N_CELLS):
            for nrow, ncol in open_neighbours(row, col):
                if (nrow, ncol) <= (row, col):      # count each passage once
                    continue
                passages += 1
                if not union(row * N_CELLS + col, nrow * N_CELLS + ncol):
                    cycles += 1
    components = len({find(i) for i in range(N_CELLS * N_CELLS)})
    return {"passages": passages, "components": components, "cycles": cycles}


def assert_perfect_maze():
    """Re-derive the paper's own invariant. Raises rather than warning.

    A silently-wrong maze changes the difficulty of the task every Experiment 4 number is
    measured against, and would be invisible in the results. Better to fail at import.
    """
    stats = maze_statistics()
    expected = {"passages": N_CELLS * N_CELLS - 1, "components": 1, "cycles": 0}
    if stats != expected:
        raise ValueError(
            f"the transcribed maze is not a perfect maze: got {stats}, expected {expected}. "
            "The paper states every pair of squares is joined by exactly one path, so this "
            "means the wall arrays are wrong."
        )
    return stats


def wall_segments():
    """Every wall as an axis-aligned segment ((x0, y0), (x1, y1)), including the border.

    Used for swept collision. `wall_rects` below is the same geometry as thin rectangles,
    which is what the LiDAR ray caster consumes.
    """
    segments = []
    n = N_CELLS
    segments += [((0, 0), (n, 0)), ((0, n), (n, n)), ((0, 0), (0, n)), ((n, 0), (n, n))]
    for row in range(n):
        y_top, y_bottom = n - row, n - row - 1
        for col in range(n - 1):
            if VWALL[row, col]:
                segments.append(((col + 1, y_bottom), (col + 1, y_top)))
    for row in range(n - 1):
        y = n - row - 1
        for col in range(n):
            if HWALL[row, col]:
                segments.append(((col, y), (col + 1, y)))
    return segments


def wall_rects(thickness=1e-3):
    """Walls as (centre_x, centre_y, half_width, half_height) rectangles.

    This is the format `robot_env.lidar_core.cast_rays` already consumes, so the LiDAR
    implementation is reused verbatim rather than reimplemented for the maze. The walls are
    given a small but non-zero thickness because a true zero-width rectangle has no
    interior for a ray to enter; `thickness` is far below the 0.5-unit success radius and
    the 1.0-unit cell, so it cannot change which cell anything is in.
    """
    rects = []
    half = thickness / 2.0
    for (x0, y0), (x1, y1) in wall_segments():
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw = max(abs(x1 - x0) / 2.0, half)
        hh = max(abs(y1 - y0) / 2.0, half)
        rects.append((cx, cy, hw, hh))
    return rects


# Checked once at import: the arrays above are the paper's maze or the module refuses to load.
MAZE_STATS = assert_perfect_maze()
