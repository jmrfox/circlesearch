"""Graph-based representation of circle arrangements.

An arrangement of n circles partitions the plane into vertices
(intersection points), edges (arcs of circles between intersection
points), and faces (connected regions). We represent this as a
planar graph using networkx, with additional structure to capture
the combinatorial embedding (cyclic arc orderings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx

from circlesearch.circle import Circle


@dataclass
class Arrangement:
    """Topological representation of an arrangement of circles.

    The arrangement graph G has:
      - Nodes: intersection points, keyed by (i, j, k) where i < j
        are the circle indices and k in {0, 1} distinguishes the two
        intersection points of circles i and j.
      - Edges: arcs of circles between consecutive intersection points.
        Each edge has a 'circle' attribute indicating which circle it
        belongs to.

    Additionally, for each circle we store the cyclic ordering of
    intersection points around it, which (together with the graph)
    fully determines the combinatorial embedding.

    Attributes:
        n_circles: Number of circles in the arrangement.
        graph: The arrangement graph.
        circle_orders: For each circle index, the cyclically ordered
            list of node keys around that circle.
        nesting: For each circle index, the index of the smallest
            circle that contains it (without intersecting it), or
            None if it is not contained in any other circle. This
            forms a forest (collection of rooted trees).
        region_vectors: For each non-intersecting circle, a
            frozenset of circle indices whose interior contains
            that circle's center. This identifies which face of
            the arrangement the circle sits in.
    """

    n_circles: int
    graph: nx.MultiGraph = field(default_factory=nx.MultiGraph)
    circle_orders: dict[int, list] = field(default_factory=dict)
    nesting: dict[int, int | None] = field(default_factory=dict)
    region_vectors: dict[int, frozenset[int]] = field(default_factory=dict)

    @classmethod
    def from_circles(cls, circles: list[Circle]) -> Arrangement:
        """Build an arrangement from a concrete list of circles.

        Computes all pairwise intersections and derives the
        arrangement graph with its combinatorial embedding.
        """
        n = len(circles)
        arr = cls(n_circles=n)

        if n == 0:
            return arr

        # Initialize nesting: for each circle, find the smallest
        # non-intersecting circle that fully contains it.
        # Circle ci is inside cj iff d(ci, cj) + ri < rj,
        # where d is the distance between centers.
        for ci in range(n):
            arr.nesting[ci] = None
        for ci in range(n):
            best_parent = None
            best_radius = float("inf")
            for cj in range(n):
                if ci == cj:
                    continue
                if circles[ci].intersects(circles[cj]):
                    continue
                d = circles[ci].distance_to(circles[cj])
                if d + circles[ci].r < circles[cj].r:
                    if circles[cj].r < best_radius:
                        best_parent = cj
                        best_radius = circles[cj].r
            arr.nesting[ci] = best_parent

        # Compute all intersection points
        # node key: (i, j, k) where i < j, k in {0, 1}
        node_positions: dict[tuple, tuple[float, float]] = {}

        for i, j in combinations(range(n), 2):
            pts = circles[i].intersection_points(circles[j])
            if len(pts) == 2:
                for k, pt in enumerate(pts):
                    key = (i, j, k)
                    node_positions[key] = pt
                    arr.graph.add_node(key, pos=pt)

        # For each circle, collect its intersection points and sort
        # them by angle around the circle to get the cyclic ordering.
        for ci in range(n):
            nodes_on_circle = []
            for key, pos in node_positions.items():
                i, j, _ = key
                if ci == i or ci == j:
                    angle = circles[ci].angle_of(*pos)
                    nodes_on_circle.append((angle, key))

            nodes_on_circle.sort()
            ordered = [key for _, key in nodes_on_circle]
            arr.circle_orders[ci] = ordered

            # Add edges (arcs) between consecutive nodes on this circle
            for idx in range(len(ordered)):
                u = ordered[idx]
                v = ordered[(idx + 1) % len(ordered)]
                arr.graph.add_edge(u, v, circle=ci)

        # Compute region vectors for non-intersecting circles.
        # For each circle that doesn't participate in any
        # intersection, record which other circles fully contain
        # it (d + r_inner < r_outer). This identifies the face
        # of the arrangement it sits in.
        for ci in range(n):
            if len(arr.circle_orders[ci]) == 0:
                containing = frozenset(
                    cj
                    for cj in range(n)
                    if cj != ci
                    and not circles[ci].intersects(circles[cj])
                    and (
                        circles[ci].distance_to(circles[cj]) + circles[ci].r
                        < circles[cj].r
                    )
                )
                arr.region_vectors[ci] = containing

        return arr

    @property
    def n_vertices(self) -> int:
        """Number of intersection points."""
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """Number of arcs."""
        return self.graph.number_of_edges()

    @property
    def n_faces(self) -> int:
        """Number of faces via Euler's formula: V - E + F = 1 + C.

        For a connected planar graph: F = E - V + 2.
        For a disconnected one: F = E - V + C + 1,
        where C is the number of connected components of the graph.

        Note: isolated circles (no intersections) each contribute
        one additional face (inside the circle). We must account
        for circles not present in the graph.
        """
        if self.n_vertices == 0:
            # No intersections: each circle is either isolated or
            # nested. The number of faces is n + 1 for n non-
            # intersecting circles in general position (each adds
            # one bounded face), but nesting complicates this.
            # For the empty arrangement (0 circles), there is 1 face.
            return self.n_circles + 1

        v = self.n_vertices
        e = self.n_edges
        c = nx.number_connected_components(self.graph)
        # Isolated circles (those with no intersections)
        circles_in_graph = set()
        for _, _, _, data in self.graph.edges(data=True, keys=True):
            circles_in_graph.add(data["circle"])
        n_isolated = self.n_circles - len(circles_in_graph)

        # Euler: F = E - V + C + 1 for the graph components,
        # plus 1 face per isolated circle
        return e - v + c + 1 + n_isolated

    def intersection_matrix(self) -> list[list[bool]]:
        """Return the n x n boolean matrix of which circles intersect."""
        n = self.n_circles
        mat = [[False] * n for _ in range(n)]
        for key in self.graph.nodes:
            i, j, _ = key
            mat[i][j] = True
            mat[j][i] = True
        return mat

    def is_isomorphic_to(self, other: Arrangement) -> bool:
        """Check if two arrangements are topologically equivalent.

        Two arrangements are equivalent if there is a graph isomorphism
        between their arrangement graphs that:
          1. Preserves the circle-labeling of edges (up to a
             permutation of circle indices).
          2. Preserves the cyclic ordering of nodes around each circle
             (up to reversal, since reflection is allowed).
        """
        if self.n_circles != other.n_circles:
            return False
        if self.n_vertices != other.n_vertices:
            return False
        if self.n_edges != other.n_edges:
            return False

        # Quick check: sorted degree sequences must match
        d1 = sorted(dict(self.graph.degree()).values())
        d2 = sorted(dict(other.graph.degree()).values())
        if d1 != d2:
            return False

        # Quick check: intersection matrices (up to permutation)
        # must have the same sorted row-sum signature
        m1 = self.intersection_matrix()
        m2 = other.intersection_matrix()
        s1 = sorted(sum(row) for row in m1)
        s2 = sorted(sum(row) for row in m2)
        if s1 != s2:
            return False

        # Full isomorphism check using networkx VF2 with edge matching
        def edge_match(e1_data, e2_data):
            # We don't compare circle labels directly since circles
            # can be permuted; instead we rely on the structural
            # isomorphism plus the cycle consistency check below.
            return True

        gm = nx.algorithms.isomorphism.MultiGraphMatcher(
            self.graph, other.graph, edge_match=edge_match
        )

        for iso in gm.isomorphisms_iter():
            if self._check_cycle_consistency(other, iso):
                return True

        return False

    def _check_cycle_consistency(self, other: Arrangement, node_map: dict) -> bool:
        """Verify that a node isomorphism preserves cyclic orderings.

        The node_map maps self's nodes to other's nodes. We need to
        find a permutation of circle indices such that the cyclic
        order of nodes around each circle in self maps (under
        node_map) to the cyclic order around the corresponding
        circle in other (allowing reversal for reflection).
        """
        # For each circle in self, compute the canonical form of
        # its mapped cyclic order (considering both orientations).
        mapped_keys: list[tuple] = []
        for ci in range(self.n_circles):
            order = self.circle_orders.get(ci, [])
            if len(order) == 0:
                mapped_keys.append(())
                continue
            mapped = [node_map[n] for n in order]
            key_fwd = _canonical_cycle(mapped)
            key_rev = _canonical_cycle(mapped[::-1])
            mapped_keys.append(min(key_fwd, key_rev))

        # For each circle in other, compute its canonical cycle key.
        other_keys: list[tuple] = []
        for ci in range(other.n_circles):
            order = other.circle_orders.get(ci, [])
            if len(order) == 0:
                other_keys.append(())
                continue
            key_fwd = _canonical_cycle(order)
            key_rev = _canonical_cycle(order[::-1])
            other_keys.append(min(key_fwd, key_rev))

        # The multisets of keys must match for a valid assignment.
        if sorted(mapped_keys) != sorted(other_keys):
            return False

        # Check placement of non-intersecting circles.
        # Build a canonical signature that encodes, for each
        # non-intersecting circle, the face it sits in (via its
        # region vector) and the nesting structure, all in a
        # permutation-invariant way.
        self_place = _placement_signature(
            self.region_vectors, self.nesting, self.n_circles
        )
        other_place = _placement_signature(
            other.region_vectors, other.nesting, other.n_circles
        )
        return self_place == other_place

    def __repr__(self) -> str:
        return (
            f"Arrangement(n_circles={self.n_circles}, "
            f"vertices={self.n_vertices}, "
            f"edges={self.n_edges}, "
            f"faces={self.n_faces})"
        )


def _nesting_signature(nesting: dict[int, int | None], n: int) -> tuple:
    """Compute a canonical signature for the nesting forest.

    Returns a sorted tuple of canonical tree signatures, one per
    root. Each tree signature is built recursively: a node's
    signature is a sorted tuple of its children's signatures.
    This is the standard canonical form for unordered rooted trees.
    """
    # Build children map
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    roots = []
    for ci in range(n):
        parent = nesting.get(ci)
        if parent is None:
            roots.append(ci)
        else:
            children[parent].append(ci)

    def _tree_sig(node: int) -> tuple:
        child_sigs = sorted(_tree_sig(c) for c in children[node])
        return tuple(child_sigs)

    root_sigs = sorted(_tree_sig(r) for r in roots)
    return tuple(root_sigs)


def _placement_signature(
    region_vectors: dict[int, frozenset[int]],
    nesting: dict[int, int | None],
    n: int,
) -> tuple:
    """Compute a canonical placement signature.

    Combines the nesting tree structure with region vector
    information to produce a permutation-invariant signature.

    For each non-intersecting circle, we compute a descriptor:
      (nesting_depth, region_vector_size)
    where nesting_depth is how deep it is in the nesting tree,
    and region_vector_size is how many circles contain its center.

    The nesting tree canonical form plus the sorted list of these
    descriptors uniquely identifies the placement up to relabeling.
    """

    # Compute nesting depth for each circle
    def _depth(ci: int) -> int:
        d = 0
        cur: int | None = ci
        while cur is not None and nesting.get(cur) is not None:
            cur = nesting[cur]
            d += 1
        return d

    # Build descriptor for each non-intersecting circle
    descriptors = []
    for ci, rv in sorted(region_vectors.items()):
        descriptors.append((_depth(ci), len(rv)))

    # Combine nesting tree signature with descriptors
    nesting_sig = _nesting_signature(nesting, n)
    return (nesting_sig, tuple(sorted(descriptors)))


def _canonical_cycle(seq: list) -> tuple:
    """Return the lexicographically smallest rotation of seq as a tuple."""
    if not seq:
        return ()
    n = len(seq)
    doubled = seq + seq
    best = min(tuple(doubled[i : i + n]) for i in range(n))
    return best


def _is_cyclic_rotation(a: list, b: list) -> bool:
    """Check if list a is a cyclic rotation of list b."""
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    # Standard doubling trick
    doubled = b + b
    n = len(a)
    for start in range(n):
        if doubled[start : start + n] == a:
            return True
    return False
