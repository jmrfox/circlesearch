"""Estimate a(n) — the number of distinct circle arrangements — via
random sampling, and plot the discovered arrangements.

Usage:
    uv run scripts/solve_an_random_sampling.py [N] [--samples S] [--seed SEED]

Arguments:
    N          Number of circles (default: 3)
    --samples  Number of random circle configurations to try (default: 100_000)
    --seed     Random seed for reproducibility (default: 42)

The script generates random circle configurations, builds their
topological arrangements, and deduplicates them using the
is_isomorphic_to equivalence check.  It then produces a gallery
of all discovered arrangements, each drawn with its constituent
circles.
"""

from __future__ import annotations

import argparse
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle

from circlesearch.circle import Circle
from circlesearch.arrangement import Arrangement

# Known values from OEIS A250001
KNOWN = {0: 1, 1: 1, 2: 3, 3: 14, 4: 173, 5: 16951}

RADII = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]


def random_circles(n: int) -> list[Circle]:
    """Generate n random circles using varied strategies.

    Uses multiple generation modes to cover both common and
    rare topological configurations:
      - wide:  circles spread across a large area
      - tight: circles clustered near the origin
      - mixed: one large circle with smaller ones nearby
      - chain: circles placed along a line with overlap
    """
    mode = random.choice(["wide", "tight", "mixed", "chain"])

    if mode == "wide":
        return [
            Circle(
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.choice(RADII),
            )
            for _ in range(n)
        ]

    if mode == "tight":
        # Small area, varied radii — good for finding
        # alternating cyclic orders and nested overlaps
        return [
            Circle(
                random.uniform(-1.5, 1.5),
                random.uniform(-1.5, 1.5),
                random.uniform(0.3, 2.0),
            )
            for _ in range(n)
        ]

    if mode == "mixed":
        # One big circle + smaller ones inside/around it
        big_r = random.uniform(3, 8)
        circles = [Circle(0, 0, big_r)]
        for _ in range(n - 1):
            r = random.uniform(0.1, big_r * 0.8)
            d = random.uniform(0, big_r + r)
            a = random.uniform(0, 6.283)
            circles.append(Circle(d * math.cos(a), d * math.sin(a), r))
        return circles

    # chain: circles along a line with controlled spacing
    circles = []
    x = 0.0
    for _ in range(n):
        r = random.uniform(0.5, 2.0)
        y = random.uniform(-0.5, 0.5)
        circles.append(Circle(x, y, r))
        x += random.uniform(0.3, 2.5)
    return circles


def solve(
    n: int,
    n_samples: int,
    seed: int,
) -> tuple[
    list[Arrangement],
    list[list[Circle]],
]:
    """Run the random search and return unique arrangements.

    Returns:
        unique_arrs: list of unique Arrangement objects
        unique_circles: circle configs for each unique arr
    """
    random.seed(seed)

    unique_arrs: list[Arrangement] = []
    unique_circles: list[list[Circle]] = []

    t0 = time.perf_counter()
    last_report = t0

    for i in range(n_samples):
        cs = random_circles(n)
        arr = Arrangement.from_circles(cs)

        is_dup = False
        for ua in unique_arrs:
            if arr.is_isomorphic_to(ua):
                is_dup = True
                break

        if not is_dup:
            unique_arrs.append(arr)
            unique_circles.append(cs)

        # Progress reporting
        now = time.perf_counter()
        if now - last_report >= 2.0 or i == n_samples - 1:
            elapsed = now - t0
            rate = (i + 1) / elapsed
            print(
                f"  [{i + 1:>{len(str(n_samples))}}/{n_samples}] "
                f"unique={len(unique_arrs)}  "
                f"({rate:.0f} samples/s, {elapsed:.1f}s elapsed)",
                flush=True,
            )
            last_report = now

    return unique_arrs, unique_circles


def plot_arrangement_gallery(
    n: int,
    unique_circles: list[list[Circle]],
    unique_arrs: list[Arrangement],
) -> plt.Figure:
    """Plot a gallery of all discovered arrangements."""
    n_arrs = len(unique_arrs)
    if n_arrs == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No arrangements found", ha="center", va="center")
        return fig

    # Determine grid layout
    cols = min(n_arrs, 5)
    rows = (n_arrs + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 3.2 * rows),
        squeeze=False,
    )
    fig.suptitle(
        f"All {n_arrs} discovered arrangements of {n} circles",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))

    for idx, (cs, arr) in enumerate(zip(unique_circles, unique_arrs)):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        # Draw circles
        for ci, c in enumerate(cs):
            patch = MplCircle(
                (c.x, c.y),
                c.r,
                fill=False,
                edgecolor=colors[ci % len(colors)],
                linewidth=1.8,
            )
            ax.add_patch(patch)

        # Draw intersection points
        for node in arr.graph.nodes:
            pos = arr.graph.nodes[node]["pos"]
            ax.plot(pos[0], pos[1], "k.", markersize=4)

        # Compute bounding box
        if cs:
            x_min = min(c.x - c.r for c in cs) - 0.5
            x_max = max(c.x + c.r for c in cs) + 0.5
            y_min = min(c.y - c.r for c in cs) - 0.5
            y_max = max(c.y + c.r for c in cs) + 0.5
            # Make square
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            half = max(x_max - x_min, y_max - y_min) / 2
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

        ax.set_aspect("equal")
        ax.set_title(
            f"#{idx + 1}  V={arr.n_vertices} E={arr.n_edges} F={arr.n_faces}",
            fontsize=9,
        )
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for idx in range(n_arrs, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate a(n) for circle arrangements (OEIS A250001)"
    )
    parser.add_argument(
        "n",
        type=int,
        nargs="?",
        default=3,
        help="Number of circles (default: 3)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Number of random configurations to try (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    n = args.n
    n_samples = args.samples
    seed = args.seed

    print(f"Searching for a({n}) with {n_samples:,}" f" random samples (seed={seed})")
    if n in KNOWN:
        print(f"Known value: a({n}) = {KNOWN[n]}")
    print()

    unique_arrs, unique_circles = solve(n, n_samples, seed)

    print(f"\nResult: found {len(unique_arrs)}" f" unique arrangements of {n} circles")
    if n in KNOWN:
        if len(unique_arrs) == KNOWN[n]:
            print(f"✓ Matches known a({n}) = {KNOWN[n]}")
        elif len(unique_arrs) < KNOWN[n]:
            print(
                f"✗ Below known a({n}) = {KNOWN[n]} — "
                f"missing {KNOWN[n] - len(unique_arrs)}, try more samples"
            )
        else:
            print(
                f"\u2717 Above known a({n}) = {KNOWN[n]}"
                f" \u2014 possible equivalence checker bug"
            )

    # Print summary table
    print(f"\n{'#':>3}  {'V':>3}  {'E':>3}" f"  {'F':>3}  {'Intersecting pairs':>18}")
    print("-" * 36)
    for i, arr in enumerate(unique_arrs):
        mat = arr.intersection_matrix()
        n_pairs = sum(sum(row) for row in mat) // 2
        print(
            f"{i + 1:>3}  {arr.n_vertices:>3}  {arr.n_edges:>3}  "
            f"{arr.n_faces:>3}  {n_pairs:>18}"
        )

    # Plot
    plot_arrangement_gallery(n, unique_circles, unique_arrs)

    plt.show()


if __name__ == "__main__":
    main()
