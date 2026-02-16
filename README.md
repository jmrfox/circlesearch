# circlesearch
Python code for studying Jonathan (Jon) Wild's circle arrangement problem.

> For $n$ circles, find the number $a(n)$ of unique arrangements of the circles in the affine plane. 

The present state of the problem is described in [OEIS A250001](https://oeis.org/A250001).

Known values of $a(n)$:

| n | a(n) |
|---|------|
| 0 | 1 |
| 1 | 1 |
| 2 | 3 |
| 3 | 14 |
| 4 | 173 |
| 5 | 16951 |

Comments from the OEIS entry: 

* We count only configurations where any two circles are either disjoint or meet in two points: We do not allow tangential contacts. A point on a circle must belong to only one or at most two circles: Three circles must not meet at a point. The circles may have different radii.
* This is in the affine plane, rather than the projective plane.
* Two arrangements are considered the same if one can be continuously changed to the other while keeping all circles circular (although the radii may be continuously changed), without changing the multiplicity of intersection points, and without a circle passing through an intersection point. Turning the whole configuration over is allowed.

## Initial search strategy: random placement

The first strategy is to estimate $a(n)$ by generating many random circle configurations, extracting their topological arrangement, and deduplicating via an equivalence check. The strategy has three layers: **representation**, **equivalence checking**, and **sampling**.

### Representation

Each arrangement of $n$ circles is represented by an `Arrangement` object containing:

1. **Arrangement graph** — a `networkx.MultiGraph` where:
   - **Nodes** are intersection points, keyed by `(i, j, k)` where `i < j` are the two circle indices and `k ∈ {0, 1}` distinguishes the two intersection points of circles `i` and `j`.
   - **Edges** are arcs of circles between consecutive intersection points. Each edge carries a `circle` attribute indicating which circle it belongs to.
   - A `MultiGraph` is required because two circles that intersect produce two nodes connected by four edges (two arcs per circle).

2. **Cyclic orderings** (`circle_orders`) — for each circle, the cyclically ordered list of intersection-point nodes around that circle, sorted by angle. Together with the graph, this fully determines the combinatorial embedding of the arrangement.

3. **Nesting tree** (`nesting`) — for each circle, the index of the smallest non-intersecting circle that fully contains it (using the geometric test $d + r_\text{inner} < r_\text{outer}$), or `None` if it sits in the unbounded face. This forms a forest of rooted trees.

4. **Region vectors** (`region_vectors`) — for each circle that has no intersection points (i.e. doesn't participate in any pairwise intersection), a `frozenset` of the indices of all circles that fully contain it. This identifies which face of the arrangement the circle sits in, distinguishing cases like "inside the lens of two overlapping circles" from "inside only one of them."

5. **Derived quantities** — vertex/edge/face counts. Face count uses Euler's formula for planar graphs: $F = E - V + C + 1 + n_\text{isolated}$, where $C$ is the number of connected components of the arrangement graph and $n_\text{isolated}$ is the number of circles with no intersections that don't appear in the graph.

### Equivalence checking

Two arrangements are equivalent (`is_isomorphic_to`) if there exists a graph isomorphism between their arrangement graphs that preserves cyclic orderings (up to reflection) and placement of non-intersecting circles. The check proceeds in stages, from cheap to expensive:

1. **Quick filters** — reject immediately if the arrangements differ in number of circles, vertices, edges, sorted degree sequence, or sorted intersection-matrix row sums.

2. **Graph isomorphism** — enumerate candidate node bijections using the VF2 algorithm (`networkx.MultiGraphMatcher`). Edge labels (circle indices) are not compared directly since circles may be permuted.

3. **Cycle consistency** — for each candidate isomorphism, map every circle's cyclic node ordering through the bijection, compute a canonical form (lexicographically smallest rotation, considering both orientations for reflection), and check that the multiset of canonical cycle keys matches between the two arrangements. This verifies that the combinatorial embedding is preserved up to circle relabeling and reflection.

4. **Placement signature** — for non-intersecting circles, compute a permutation-invariant canonical signature that combines:
   - The **nesting tree** canonical form (standard canonical form for unordered rooted forests: each node's signature is the sorted tuple of its children's signatures).
   - A sorted list of **descriptors** `(nesting_depth, region_vector_size)` for each non-intersecting circle, capturing both how deep it is in the nesting tree and which face it occupies.

   This distinguishes arrangements where non-intersecting circles sit in different faces (e.g. inside the lens vs. inside only one circle of an overlapping pair).

### Sampling

The search script (`scripts/solve_an_random_sampling.py`) generates random circle configurations and tests each against the growing list of known-unique arrangements. To cover both common and rare topological configurations, it uses four generation modes chosen uniformly at random:

| Mode | Description | What it finds well |
|------|-------------|--------------------|
| **wide** | Circles with random centers in $[-5, 5]^2$ and radii from a discrete set $\{0.1, 0.2, \ldots, 8.0\}$ | Disjoint, simple overlaps, basic nesting |
| **tight** | Circles clustered in $[-1.5, 1.5]^2$ with continuous radii in $[0.3, 2.0]$ | Multiple overlaps, alternating cyclic orders |
| **mixed** | One large circle (radius $[3, 8]$) at the origin, with $n-1$ smaller circles placed at random distances and angles | Nesting variants, lens configurations, contains-both |
| **chain** | Circles placed along the $x$-axis with controlled spacing $[0.3, 2.5]$ and radii $[0.5, 2.0]$ | Linear chains, grouped cyclic orders |

**Limitations.** Some arrangements require very specific geometric relationships (e.g. the "alternating" 2-edge pattern or the "mixed" 3-edge pattern for $n = 3$) and are rarely produced by random sampling. For $n = 3$, the current strategy reliably finds 12 of the 14 arrangements with 50k samples; finding all 14 requires either more samples or targeted generation strategies.

## Usage

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Run the search for n=3 with 100k samples
uv run scripts/solve_an_random_sampling.py 3 --samples 100000 --seed 42

# Run tests
uv run pytest -v
```

## Project structure

```
circlesearch/          Core library
  circle.py            Circle dataclass with geometric methods
  arrangement.py       Arrangement graph, equivalence checking
scripts/
  solve_an_random_sampling.py          Random search script with plotting
tests/
  test_circle.py       Tests for Circle class
  test_arrangement.py  Tests for Arrangement class (117 tests)
notebooks/
  demo.ipynb           Interactive demo notebook
```
