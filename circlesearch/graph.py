from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx


class ArrangementGraph:
    """Planar arrangement graph built from circle intersections.

    Nodes: intersection points ``(i, j, k)`` where
           ``i < j`` are circle indices and ``k in {0, 1}``.

    Edges: arcs along circles between consecutive
           intersection points in cyclic order.
           Each edge has a ``'circle'`` attribute.

    Attributes:
        graph: The underlying ``nx.MultiGraph``.
        parent: Containment map (circle index -> parent index or None).
        intersections: Set of intersecting circle pairs.
        cyclic_orders: Per-circle cyclic ordering of intersection labels.
    """

    def __init__(self, parent, intersections, cyclic_orders):
        self.parent = dict(parent) if parent else {}
        self.intersections = set(intersections)
        self.cyclic_orders = dict(cyclic_orders)
        self.graph = self._build()

    def _build(self) -> nx.MultiGraph:
        """
        Build a unified graph encoding:
          - circle nodes
          - intersection nodes
          - containment edges
          - incidence edges
          - arc edges (cyclic order)
        """

        G = nx.MultiDiGraph()

        # --- Step 1: Add circle nodes ---
        for i in self.parent:
            G.add_node(f"C{i}", type="circle")

        # --- Step 2: Add containment edges ---
        for child, par in self.parent.items():
            if par is not None:
                G.add_edge(f"C{child}", f"C{par}", type="containment")

        # --- Step 3: Add intersection nodes ---
        for i, j in self.intersections:
            if i > j:
                i, j = j, i

            for k in (0, 1):
                node = f"X{i}_{j}_{k}"
                G.add_node(node, type="intersection")

                # incidence to both circles
                G.add_edge(f"C{i}", node, type="incidence")
                G.add_edge(f"C{j}", node, type="incidence")

        # --- Step 4: Add arc edges from cyclic order ---
        for i, order in self.cyclic_orders.items():
            if not order:
                continue

            nodes_on_circle = []

            for j, k in order:
                a, b = sorted((i, j))
                node = f"X{a}_{b}_{k}"
                nodes_on_circle.append(node)

            n = len(nodes_on_circle)
            for idx in range(n):
                u = nodes_on_circle[idx]
                v = nodes_on_circle[(idx + 1) % n]

                G.add_edge(u, v, type="arc", circle=i)

        return G

    # --- Convenience properties ---

    @property
    def n_vertices(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def n_components(self) -> int:
        if self.n_vertices == 0:
            return 0
        return nx.number_connected_components(self.graph.to_undirected())

    # --- Display methods ---

    def draw(self, ax=None, **kwargs) -> None:
        """Draw the arrangement graph using networkx.

        Edges are colored by type (arc / containment / incidence).
        Parallel edges between the same pair of nodes are curved
        apart so they remain visually distinguishable.

        Pass an existing matplotlib ``Axes`` via *ax*, or a new
        figure is created.
        """
        from matplotlib.patches import FancyArrowPatch

        G = self.graph
        if G.number_of_nodes() == 0:
            print("(empty graph — nothing to draw)")
            return

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(7, 7))

        pos = nx.spring_layout(G, seed=0)
        node_size = kwargs.get("node_size", 500)
        font_size = kwargs.get("font_size", 8)
        lw = kwargs.get("width", 2.0)

        # --- Draw nodes ---
        node_colors = []
        for node in G.nodes:
            ntype = G.nodes[node].get("type")
            if ntype == "circle":
                node_colors.append("#3b82f6")
            else:
                node_colors.append("#94a3b8")

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_size,
        )
        labels = {node: str(node) for node in G.nodes}
        nx.draw_networkx_labels(
            G,
            pos,
            labels,
            ax=ax,
            font_size=font_size,
        )

        # --- Colour map for arc edges ---
        cmap = plt.cm.Set2
        arc_cids = sorted(
            {d["circle"] for _, _, d in G.edges(data=True) if d.get("type") == "arc"}
        )
        cid_to_color = {
            cid: cmap(i / max(len(arc_cids), 1)) for i, cid in enumerate(arc_cids)
        }

        # --- Count parallel edges per node pair ---
        pair_count: dict[tuple, int] = defaultdict(int)
        pair_index: dict[tuple, int] = defaultdict(int)

        edge_list = list(G.edges(data=True))
        for u, v, _ in edge_list:
            key = (min(u, v), max(u, v))
            pair_count[key] += 1

        # --- Draw edges with curvature ---
        for u, v, d in edge_list:
            key = (min(u, v), max(u, v))
            n_parallel = pair_count[key]
            idx = pair_index[key]
            pair_index[key] += 1

            # Compute curvature: spread symmetrically
            if n_parallel == 1:
                rad = 0.0
            else:
                spread = 0.15 * (n_parallel - 1)
                rad = -spread + 2 * spread * idx / (n_parallel - 1)

            etype = d.get("type")
            if etype == "arc":
                color = cid_to_color[d["circle"]]
                style = "solid"
            elif etype == "containment":
                color = "#dc2626"
                style = "dashed"
            else:
                color = "#9ca3af"
                style = "dotted"

            arrow = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=20,
                shrinkA=10,
                shrinkB=10,
                color=color,
                linewidth=lw,
                linestyle=style,
            )
            ax.add_patch(arrow)

        # --- Legend ---
        for cid, color in cid_to_color.items():
            ax.plot(
                [],
                [],
                color=color,
                linewidth=3,
                label=f"circle {cid} arcs",
            )
        ax.plot(
            [],
            [],
            color="#dc2626",
            linewidth=2,
            linestyle="--",
            label="containment",
        )
        ax.plot(
            [],
            [],
            color="#9ca3af",
            linewidth=2,
            linestyle=":",
            label="incidence",
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Arrangement Graph", fontsize=12)

    def summary(self) -> str:
        """Return a formatted string summarizing the graph."""
        G = self.graph
        lines = []
        lines.append("=" * 50)
        lines.append("  Arrangement Graph Summary")
        lines.append("=" * 50)
        lines.append(
            f"  Vertices: {self.n_vertices}    "
            f"Edges: {self.n_edges}    "
            f"Components: {self.n_components}"
        )

        # Intersecting pairs
        pairs = sorted(tuple(sorted(p)) for p in self.intersections)
        lines.append(
            f"  Intersecting pairs ({len(pairs)}): "
            + ", ".join(f"({a},{b})" for a, b in pairs)
        )

        # Containment / nesting
        if self.parent:
            lines.append("  Containment (child -> parent):")
            for ci in sorted(self.parent):
                p = self.parent[ci]
                if p is not None:
                    lines.append(f"    circle {ci} inside circle {p}")
                else:
                    lines.append(f"    circle {ci} (root)")
        else:
            lines.append("  Containment: (none)")

        # Cyclic orders
        lines.append("  Cyclic orders:")
        for ci in sorted(self.cyclic_orders):
            order = self.cyclic_orders[ci]
            if order:
                labels = [f"({j},{k})" for j, k in order]
                lines.append(f"    circle {ci}: " + " -> ".join(labels))
            else:
                lines.append(f"    circle {ci}: (no intersections)")

        # Per-edge detail grouped by type
        edges_by_type = defaultdict(list)
        for u, v, d in G.edges(data=True):
            edges_by_type[d.get("type", "unknown")].append((u, v, d))

        for etype in ["containment", "incidence", "arc"]:
            group = edges_by_type.get(etype, [])
            if not group:
                continue
            lines.append(f"  {etype.capitalize()} edges ({len(group)}):")
            for u, v, d in group:
                extra = ""
                if etype == "arc" and "circle" in d:
                    extra = f"  [circle {d['circle']}]"
                lines.append(f"    {u} -> {v}{extra}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the graph summary to stdout."""
        print(self.summary())

    def __repr__(self) -> str:
        return (
            f"ArrangementGraph("
            f"vertices={self.n_vertices}, "
            f"edges={self.n_edges})"
        )


def build_arrangement_graph(parent, intersections, cyclic_orders):
    """Build an ArrangementGraph (backward-compatible wrapper).

    Returns the underlying ``nx.MultiGraph``.  Prefer using
    ``ArrangementGraph`` directly for access to draw/summary.
    """
    ag = ArrangementGraph(parent, intersections, cyclic_orders)
    return ag.graph
