"""Tests for circlesearch.arrangement.Arrangement."""

import pytest
import networkx as nx

from circlesearch.circle import Circle
from circlesearch.arrangement import (
    Arrangement,
    _canonical_cycle,
    _is_cyclic_rotation,
    _nesting_signature,
    _placement_signature,
)


# ── Helper function tests ─────────────────────────────────────────


class TestCanonicalCycle:
    def test_empty(self):
        assert _canonical_cycle([]) == ()

    def test_single(self):
        assert _canonical_cycle([5]) == (5,)

    def test_already_canonical(self):
        assert _canonical_cycle([1, 2, 3]) == (1, 2, 3)

    def test_rotation_needed(self):
        assert _canonical_cycle([3, 1, 2]) == (1, 2, 3)

    def test_all_same(self):
        assert _canonical_cycle([2, 2, 2]) == (2, 2, 2)

    def test_two_elements(self):
        assert _canonical_cycle([5, 3]) == (3, 5)

    def test_alternating_vs_grouped(self):
        alt = _canonical_cycle([0, 2, 0, 2])
        grp = _canonical_cycle([0, 0, 2, 2])
        assert alt != grp


class TestIsCyclicRotation:
    def test_empty(self):
        assert _is_cyclic_rotation([], [])

    def test_identity(self):
        assert _is_cyclic_rotation([1, 2, 3], [1, 2, 3])

    def test_rotation(self):
        assert _is_cyclic_rotation([2, 3, 1], [1, 2, 3])

    def test_not_rotation(self):
        assert not _is_cyclic_rotation([1, 3, 2], [1, 2, 3])

    def test_different_lengths(self):
        assert not _is_cyclic_rotation([1, 2], [1, 2, 3])

    def test_single_element(self):
        assert _is_cyclic_rotation([1], [1])
        assert not _is_cyclic_rotation([1], [2])


class TestNestingSignature:
    def test_empty(self):
        assert _nesting_signature({}, 0) == ()

    def test_single_circle(self):
        assert _nesting_signature({0: None}, 1) == ((),)

    def test_two_disjoint(self):
        assert _nesting_signature({0: None, 1: None}, 2) == ((), ())

    def test_two_nested(self):
        assert _nesting_signature({0: None, 1: 0}, 2) == (((),),)

    def test_disjoint_vs_nested_differ(self):
        sig_d = _nesting_signature({0: None, 1: None}, 2)
        sig_n = _nesting_signature({0: None, 1: 0}, 2)
        assert sig_d != sig_n

    def test_three_chain_nesting(self):
        sig = _nesting_signature({0: None, 1: 0, 2: 1}, 3)
        assert sig == ((((),),),)

    def test_three_flat(self):
        sig = _nesting_signature({0: None, 1: 0, 2: 0}, 3)
        assert sig == (((), ()),)

    def test_nesting_parent_order_irrelevant(self):
        sig_a = _nesting_signature({0: None, 1: 0}, 2)
        sig_b = _nesting_signature({0: 1, 1: None}, 2)
        assert sig_a == sig_b


class TestPlacementSignature:
    def test_empty(self):
        sig = _placement_signature({}, {}, 0)
        assert sig == ((), ())

    def test_no_region_vectors(self):
        sig = _placement_signature({}, {0: None, 1: None}, 2)
        assert sig[1] == ()

    def test_lens_vs_inside_one_differ(self):
        sig_lens = _placement_signature(
            {2: frozenset({0, 1})},
            {0: None, 1: None, 2: 0},
            3,
        )
        sig_one = _placement_signature(
            {2: frozenset({0})},
            {0: None, 1: None, 2: 0},
            3,
        )
        assert sig_lens != sig_one


# ── Arrangement.from_circles: basic topology ──────────────────────


class TestArrangementFromCircles:
    def test_zero_circles(self):
        arr = Arrangement.from_circles([])
        assert arr.n_circles == 0
        assert arr.n_vertices == 0
        assert arr.n_edges == 0
        assert arr.n_faces == 1

    def test_one_circle(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1)])
        assert arr.n_circles == 1
        assert arr.n_vertices == 0
        assert arr.n_edges == 0
        assert arr.n_faces == 2
        assert arr.nesting == {0: None}

    def test_two_disjoint(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(5, 0, 1)])
        assert arr.n_circles == 2
        assert arr.n_vertices == 0
        assert arr.n_edges == 0
        assert arr.n_faces == 3

    def test_two_overlapping(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        assert arr.n_circles == 2
        assert arr.n_vertices == 2
        assert arr.n_edges == 4
        assert arr.n_faces == 4

    def test_two_nested(self):
        arr = Arrangement.from_circles([Circle(0, 0, 3), Circle(0, 0, 1)])
        assert arr.n_circles == 2
        assert arr.n_vertices == 0
        assert arr.n_edges == 0
        assert arr.n_faces == 3
        assert arr.nesting[1] == 0
        assert arr.nesting[0] is None

    def test_three_venn(self):
        circles = [
            Circle(0, 0, 1),
            Circle(1, 0, 1),
            Circle(0.5, 0.866, 1),
        ]
        arr = Arrangement.from_circles(circles)
        assert arr.n_vertices == 6
        assert arr.n_edges == 12
        assert arr.n_faces == 8

    def test_three_all_disjoint(self):
        arr = Arrangement.from_circles(
            [Circle(0, 0, 1), Circle(5, 0, 1), Circle(10, 0, 1)]
        )
        assert arr.n_vertices == 0
        assert arr.n_faces == 4

    def test_three_nested_chain(self):
        circles = [
            Circle(0, 0, 5),
            Circle(0, 0, 3),
            Circle(0, 0, 1),
        ]
        arr = Arrangement.from_circles(circles)
        assert arr.nesting[2] == 1
        assert arr.nesting[1] == 0
        assert arr.nesting[0] is None

    def test_three_two_inside_one(self):
        circles = [
            Circle(0, 0, 5),
            Circle(-2, 0, 1),
            Circle(2, 0, 1),
        ]
        arr = Arrangement.from_circles(circles)
        assert arr.nesting[1] == 0
        assert arr.nesting[2] == 0
        assert arr.nesting[0] is None


# ── Region vectors ────────────────────────────────────────────────


class TestRegionVectors:
    def test_disjoint_empty_rv(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(5, 0, 1)])
        assert arr.region_vectors[0] == frozenset()
        assert arr.region_vectors[1] == frozenset()

    def test_nested_rv(self):
        arr = Arrangement.from_circles([Circle(0, 0, 3), Circle(0, 0, 1)])
        assert arr.region_vectors[1] == frozenset({0})
        assert arr.region_vectors[0] == frozenset()

    def test_lens_rv(self):
        arr = Arrangement.from_circles(
            [Circle(0, 0, 2), Circle(2, 0, 2), Circle(1, 0, 0.3)]
        )
        assert arr.region_vectors[2] == frozenset({0, 1})

    def test_inside_one_rv(self):
        arr = Arrangement.from_circles(
            [Circle(0, 0, 2), Circle(3, 0, 2), Circle(-1.5, 0, 0.1)]
        )
        assert arr.region_vectors[2] == frozenset({0})

    def test_intersecting_circles_not_in_rv(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        assert 0 not in arr.region_vectors
        assert 1 not in arr.region_vectors


# ── Euler's formula consistency ───────────────────────────────────


class TestEulerFormula:
    @pytest.mark.parametrize(
        "circles,expected_vef",
        [
            ([], (0, 0, 1)),
            ([Circle(0, 0, 1)], (0, 0, 2)),
            ([Circle(0, 0, 1), Circle(5, 0, 1)], (0, 0, 3)),
            ([Circle(0, 0, 1), Circle(1, 0, 1)], (2, 4, 4)),
            ([Circle(0, 0, 2), Circle(0, 0, 0.5)], (0, 0, 3)),
            (
                [
                    Circle(0, 0, 1),
                    Circle(1, 0, 1),
                    Circle(0.5, 0.866, 1),
                ],
                (6, 12, 8),
            ),
        ],
        ids=[
            "empty",
            "single",
            "2-disjoint",
            "2-overlapping",
            "2-nested",
            "3-venn",
        ],
    )
    def test_vef_values(self, circles, expected_vef):
        arr = Arrangement.from_circles(circles)
        assert (arr.n_vertices, arr.n_edges, arr.n_faces) == expected_vef

    def test_euler_holds_for_overlapping(self):
        configs = [
            [Circle(0, 0, 1), Circle(1, 0, 1)],
            [
                Circle(0, 0, 1),
                Circle(1, 0, 1),
                Circle(0.5, 0.866, 1),
            ],
            [Circle(0, 0, 1), Circle(1.5, 0, 1), Circle(3, 0, 1)],
            [Circle(0, 0, 1), Circle(0.8, 0, 1), Circle(1.6, 0, 1)],
        ]
        for circles in configs:
            arr = Arrangement.from_circles(circles)
            if arr.n_vertices > 0:
                c = nx.number_connected_components(arr.graph)
                circles_in_graph = set()
                for _, _, _, data in arr.graph.edges(data=True, keys=True):
                    circles_in_graph.add(data["circle"])
                n_iso = arr.n_circles - len(circles_in_graph)
                expected = arr.n_edges - arr.n_vertices + c + 1 + n_iso
                assert arr.n_faces == expected


# ── Intersection matrix ───────────────────────────────────────────


class TestIntersectionMatrix:
    def test_no_circles(self):
        assert Arrangement.from_circles([]).intersection_matrix() == []

    def test_two_disjoint(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(5, 0, 1)])
        assert arr.intersection_matrix() == [
            [False, False],
            [False, False],
        ]

    def test_two_overlapping(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        assert arr.intersection_matrix() == [
            [False, True],
            [True, False],
        ]

    def test_symmetric(self):
        circles = [
            Circle(0, 0, 1),
            Circle(1, 0, 1),
            Circle(0.5, 0.866, 1),
        ]
        mat = Arrangement.from_circles(circles).intersection_matrix()
        for i in range(3):
            for j in range(3):
                assert mat[i][j] == mat[j][i]


# ── Circle orders ─────────────────────────────────────────────────


class TestCircleOrders:
    def test_no_intersections_empty(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(5, 0, 1)])
        assert arr.circle_orders[0] == []
        assert arr.circle_orders[1] == []

    def test_two_overlapping_length(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        assert len(arr.circle_orders[0]) == 2
        assert len(arr.circle_orders[1]) == 2

    def test_three_venn_length(self):
        circles = [
            Circle(0, 0, 1),
            Circle(1, 0, 1),
            Circle(0.5, 0.866, 1),
        ]
        arr = Arrangement.from_circles(circles)
        for ci in range(3):
            assert len(arr.circle_orders[ci]) == 4

    def test_nodes_valid(self):
        circles = [
            Circle(0, 0, 1),
            Circle(1, 0, 1),
            Circle(0.5, 0.866, 1),
        ]
        arr = Arrangement.from_circles(circles)
        all_nodes = set(arr.graph.nodes)
        for ci in range(3):
            for node in arr.circle_orders[ci]:
                assert node in all_nodes

    def test_each_node_on_two_circles(self):
        circles = [
            Circle(0, 0, 1),
            Circle(1, 0, 1),
            Circle(0.5, 0.866, 1),
        ]
        arr = Arrangement.from_circles(circles)
        for node in arr.graph.nodes:
            count = sum(1 for ci in range(3) if node in arr.circle_orders[ci])
            assert count == 2


# ── Equivalence: is_isomorphic_to ─────────────────────────────────


class TestIsomorphism:
    # -- Self-isomorphism --

    def test_self_iso_empty(self):
        arr = Arrangement.from_circles([])
        assert arr.is_isomorphic_to(arr)

    def test_self_iso_single(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1)])
        assert arr.is_isomorphic_to(arr)

    def test_self_iso_overlapping(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        assert arr.is_isomorphic_to(arr)

    def test_self_iso_venn(self):
        arr = Arrangement.from_circles(
            [
                Circle(0, 0, 1),
                Circle(1, 0, 1),
                Circle(0.5, 0.866, 1),
            ]
        )
        assert arr.is_isomorphic_to(arr)

    # -- Same topology, different placement --

    def test_overlap_different_positions(self):
        a = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        b = Arrangement.from_circles([Circle(10, 10, 1), Circle(11, 10, 1)])
        assert a.is_isomorphic_to(b)

    def test_overlap_different_radii(self):
        a = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        b = Arrangement.from_circles([Circle(0, 0, 3), Circle(3, 0, 3)])
        assert a.is_isomorphic_to(b)

    def test_overlap_different_orientation(self):
        a = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        b = Arrangement.from_circles([Circle(0, 0, 1), Circle(0, 1, 1)])
        assert a.is_isomorphic_to(b)

    def test_venn_different_placement(self):
        a = Arrangement.from_circles(
            [
                Circle(0, 0, 1),
                Circle(1, 0, 1),
                Circle(0.5, 0.866, 1),
            ]
        )
        b = Arrangement.from_circles(
            [
                Circle(10, 10, 2),
                Circle(12, 10, 2),
                Circle(11, 11.732, 2),
            ]
        )
        assert a.is_isomorphic_to(b)

    # -- n=2: all three distinct --

    def test_n2_all_distinct(self):
        arrs = [
            Arrangement.from_circles([Circle(0, 0, 1), Circle(5, 0, 1)]),
            Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)]),
            Arrangement.from_circles([Circle(0, 0, 3), Circle(0, 0, 1)]),
        ]
        for i in range(3):
            for j in range(i + 1, 3):
                assert not arrs[i].is_isomorphic_to(arrs[j])

    # -- Different n_circles --

    def test_different_n_circles(self):
        a = Arrangement.from_circles([Circle(0, 0, 1)])
        b = Arrangement.from_circles([Circle(0, 0, 1), Circle(5, 0, 1)])
        assert not a.is_isomorphic_to(b)

    # -- Nesting-sensitive --

    def test_disjoint_vs_nested(self):
        a = Arrangement.from_circles([Circle(0, 0, 1), Circle(5, 0, 1)])
        b = Arrangement.from_circles([Circle(0, 0, 3), Circle(0, 0, 1)])
        assert not a.is_isomorphic_to(b)

    def test_three_disjoint_vs_two_in_one(self):
        a = Arrangement.from_circles(
            [
                Circle(0, 0, 1),
                Circle(5, 0, 1),
                Circle(10, 0, 1),
            ]
        )
        b = Arrangement.from_circles(
            [
                Circle(0, 0, 5),
                Circle(-2, 0, 1),
                Circle(2, 0, 1),
            ]
        )
        assert not a.is_isomorphic_to(b)

    def test_chain_nest_vs_flat_nest(self):
        a = Arrangement.from_circles(
            [
                Circle(0, 0, 5),
                Circle(0, 0, 3),
                Circle(0, 0, 1),
            ]
        )
        b = Arrangement.from_circles(
            [
                Circle(0, 0, 5),
                Circle(-2, 0, 1),
                Circle(2, 0, 1),
            ]
        )
        assert not a.is_isomorphic_to(b)

    # -- Lens vs inside-one --

    def test_lens_vs_inside_one(self):
        lens = Arrangement.from_circles(
            [
                Circle(0, 0, 2),
                Circle(2, 0, 2),
                Circle(1, 0, 0.3),
            ]
        )
        inside = Arrangement.from_circles(
            [
                Circle(0, 0, 2),
                Circle(2, 0, 2),
                Circle(0, 0, 0.3),
            ]
        )
        assert not lens.is_isomorphic_to(inside)

    # -- Permutation invariance --

    def test_circle_order_irrelevant(self):
        a = Arrangement.from_circles(
            [
                Circle(0, 0, 1),
                Circle(1, 0, 1),
                Circle(0.5, 0.866, 1),
            ]
        )
        b = Arrangement.from_circles(
            [
                Circle(0.5, 0.866, 1),
                Circle(0, 0, 1),
                Circle(1, 0, 1),
            ]
        )
        assert a.is_isomorphic_to(b)

    def test_nested_circle_order_irrelevant(self):
        a = Arrangement.from_circles(
            [
                Circle(0, 0, 5),
                Circle(-2, 0, 1),
                Circle(2, 0, 1),
            ]
        )
        b = Arrangement.from_circles(
            [
                Circle(2, 0, 1),
                Circle(0, 0, 5),
                Circle(-2, 0, 1),
            ]
        )
        assert a.is_isomorphic_to(b)


# ── Counting: verify a(n) for small n ─────────────────────────────


class TestKnownCounts:
    def test_a0_equals_1(self):
        arr = Arrangement.from_circles([])
        assert arr.n_circles == 0
        assert arr.n_faces == 1

    def test_a1_equals_1(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1)])
        assert arr.n_circles == 1

    def test_a2_equals_3(self):
        reps = [
            Arrangement.from_circles([Circle(0, 0, 1), Circle(5, 0, 1)]),
            Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)]),
            Arrangement.from_circles([Circle(0, 0, 3), Circle(0, 0, 1)]),
        ]
        for i in range(3):
            for j in range(i + 1, 3):
                assert not reps[i].is_isomorphic_to(reps[j])

    def test_a3_equals_14(self):
        """Construct all 14 arrangements of 3 circles and verify
        they are pairwise non-isomorphic.

        Organized by number of intersecting pairs:
        - 0 edges: 4 (nesting tree variants)
        - 1 edge:  4 (third circle placement variants)
        - 2 edges: 3 (cyclic order + nesting variants)
        - 3 edges: 3 (cyclic order variants)
        """
        reps = [
            # --- 0 edges (4) ---
            # 1. All disjoint
            Arrangement.from_circles(
                [
                    Circle(0, 0, 1),
                    Circle(5, 0, 1),
                    Circle(10, 0, 1),
                ]
            ),
            # 2. One nested pair + one separate
            Arrangement.from_circles(
                [
                    Circle(0, 0, 3),
                    Circle(0, 0, 1),
                    Circle(10, 0, 1),
                ]
            ),
            # 3. Two inside one (flat)
            Arrangement.from_circles(
                [
                    Circle(0, 0, 5),
                    Circle(-2, 0, 1),
                    Circle(2, 0, 1),
                ]
            ),
            # 4. Chain nesting (3 levels)
            Arrangement.from_circles(
                [
                    Circle(0, 0, 5),
                    Circle(0, 0, 3),
                    Circle(0, 0, 1),
                ]
            ),
            # --- 1 edge (4) ---
            # 5. Overlap pair + third disjoint outside
            Arrangement.from_circles(
                [
                    Circle(0, 0, 1),
                    Circle(1, 0, 1),
                    Circle(10, 0, 1),
                ]
            ),
            # 6. Overlap pair + third inside one of the pair
            Arrangement.from_circles(
                [
                    Circle(0, 0, 2),
                    Circle(3, 0, 2),
                    Circle(-1.5, 0, 0.1),
                ]
            ),
            # 7. Overlap pair + third contains both
            Arrangement.from_circles(
                [
                    Circle(0, 0, 1),
                    Circle(1, 0, 1),
                    Circle(0.5, 0, 5),
                ]
            ),
            # 8. Overlap pair + third in the lens (inside both)
            Arrangement.from_circles(
                [
                    Circle(0, 0, 2),
                    Circle(2, 0, 2),
                    Circle(1, 0, 0.3),
                ]
            ),
            # --- 2 edges (3) ---
            # 9. Two pairs, grouped cyclic order
            Arrangement.from_circles(
                [
                    Circle(0, 0, 1.5),
                    Circle(-1, 0, 1),
                    Circle(1, 0, 1),
                ]
            ),
            # 10. Two pairs with nesting
            Arrangement.from_circles(
                [
                    Circle(0, 0, 5),
                    Circle(0, 0, 3),
                    Circle(4, 0, 2),
                ]
            ),
            # 11. Two pairs, alternating cyclic order
            Arrangement.from_circles(
                [
                    Circle(-0.5, 2, 1),
                    Circle(0, 2, 0.7),
                    Circle(-0.5, 2, 1),
                ]
            ),
            # --- 3 edges (3) ---
            # 12. Venn diagram (alternating on all circles)
            Arrangement.from_circles(
                [
                    Circle(0, 0, 1),
                    Circle(1, 0, 1),
                    Circle(0.5, 0.866, 1),
                ]
            ),
            # 13. Linear (grouped on all circles)
            Arrangement.from_circles(
                [
                    Circle(0, 0, 1),
                    Circle(0.8, 0, 1),
                    Circle(1.6, 0, 1),
                ]
            ),
            # 14. Mixed (one alternating, two grouped)
            Arrangement.from_circles(
                [
                    Circle(0, 0, 1),
                    Circle(1.2, 0, 1),
                    Circle(0.6, 0, 0.8),
                ]
            ),
        ]

        n = len(reps)
        assert n == 14, f"Expected 14 representatives, got {n}"

        for i in range(n):
            for j in range(i + 1, n):
                assert not reps[i].is_isomorphic_to(reps[j]), (
                    f"a(3) reps {i} and {j} should differ:\n"
                    f"  {i}: {reps[i]}\n"
                    f"  {j}: {reps[j]}"
                )


# ── Edge cases and robustness ─────────────────────────────────────


class TestEdgeCases:
    def test_repr(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1)])
        r = repr(arr)
        assert "n_circles=1" in r
        assert "vertices=0" in r

    def test_graph_is_multigraph(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        assert isinstance(arr.graph, nx.MultiGraph)

    def test_edges_have_circle_attr(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        for _, _, _, data in arr.graph.edges(data=True, keys=True):
            assert "circle" in data
            assert data["circle"] in (0, 1)

    def test_nodes_have_pos_attr(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        for node in arr.graph.nodes:
            assert "pos" in arr.graph.nodes[node]

    def test_node_keys_are_triples(self):
        arr = Arrangement.from_circles([Circle(0, 0, 1), Circle(1, 0, 1)])
        for node in arr.graph.nodes:
            i, j, k = node
            assert i < j
            assert k in (0, 1)

    def test_four_circles_euler(self):
        circles = [
            Circle(0, 0, 1),
            Circle(1, 0, 1),
            Circle(0.5, 0.866, 1),
            Circle(0.5, -0.866, 1),
        ]
        arr = Arrangement.from_circles(circles)
        if arr.n_vertices > 0:
            c = nx.number_connected_components(arr.graph)
            cig = set()
            for _, _, _, d in arr.graph.edges(data=True, keys=True):
                cig.add(d["circle"])
            n_iso = arr.n_circles - len(cig)
            expected = arr.n_edges - arr.n_vertices + c + 1 + n_iso
            assert arr.n_faces == expected
