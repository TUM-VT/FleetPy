import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."))
NW_DIR = os.path.join(REPO_ROOT, "data", "networks")
DEMAND_DIR = os.path.join(REPO_ROOT, "data", "demand", "agimo_wp1", "matched")

# Set one of these two values for your scenario source.
NETWORK_NAME = "grid_l1_w1_hubs2_cell100"

# Set to None to skip the demand overlay.
DEMAND_FILE = f"{DEMAND_DIR}/{NETWORK_NAME}/50rph_dir0.5_seed0_spatial_uniform_temporal_uniform_user_all_normal.csv"

# Output settings.
OUTPUT_FILE = "network_setup.png"
SHOW_FIGURE = True


def resolve_network_dir():
    if NETWORK_NAME:
        return os.path.join(NW_DIR, NETWORK_NAME)
    raise ValueError("Set either NETWORK_NAME or NETWORK_DIR in this file.")


def read_hubs_csv(path):
    try:
        hubs_df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["node_index", "hub_id"])
    return hubs_df


def visualize_network(network_dir, output_path, demand_df=None, show=False):
    base_dir = os.path.join(network_dir, "base")
    nodes_path = os.path.join(base_dir, "nodes.csv")
    edges_path = os.path.join(base_dir, "edges.csv")
    hubs_path = os.path.join(base_dir, "hubs.csv")

    for path in [nodes_path, edges_path, hubs_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input file not found: {path}")

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    hubs_df = read_hubs_csv(hubs_path)

    # Stored network coordinates are (x=length, y=width).
    # Swap for display so length is horizontal (right) and width is vertical (down).
    pos_by_node = {
        int(row.node_index): (float(row.pos_x), float(row.pos_y))
        for row in nodes_df.itertuples(index=False)
    }

    seen_undirected = set()
    unique_edges = []
    for row in edges_df.itertuples(index=False):
        a = int(row.from_node)
        b = int(row.to_node)
        key = (a, b) if a < b else (b, a)
        if key in seen_undirected:
            continue
        seen_undirected.add(key)
        unique_edges.append((a, b))

    fig, ax = plt.subplots(figsize=(10, 7))

    for from_node, to_node in unique_edges:
        x1, y1 = pos_by_node[from_node]
        x2, y2 = pos_by_node[to_node]
        ax.plot([x1, x2], [y1, y2], color="0.8", linewidth=0.8, zorder=1)

    ax.scatter(
        nodes_df["pos_x"],
        nodes_df["pos_y"],
        s=15,
        c="#2c7fb8",
        alpha=0.9,
        label="nodes",
        zorder=2,
    )

    if not hubs_df.empty and "node_index" in hubs_df.columns:
        hub_xy = [pos_by_node[int(nid)] for nid in hubs_df["node_index"].tolist()]
        if hub_xy:
            hub_x, hub_y = zip(*hub_xy)
            ax.scatter(
                hub_x,
                hub_y,
                s=110,
                c="#d95f02",
                edgecolors="black",
                linewidths=0.7,
                marker="*",
                label="hubs",
                zorder=3,
            )

    if demand_df is not None and not demand_df.empty:
        hub_nodes = set(hubs_df["node_index"].astype(int).tolist()) if not hubs_df.empty else set()
        jitter_scale = 30
        rng = np.random.default_rng(seed=0)

        has_groups = "user_group" in demand_df.columns
        groups = sorted(demand_df["user_group"].unique()) if has_groups else [None]
        cmap = plt.get_cmap("tab10")
        group_colors = {g: cmap(i % 10) for i, g in enumerate(groups)}

        for group in groups:
            subset = demand_df if group is None else demand_df[demand_df["user_group"] == group]
            color = group_colors[group]
            label_suffix = f" ({group})" if group is not None else ""

            origin_xy, dest_xy = [], []
            for row in subset.itertuples(index=False):
                start, end = int(row.start), int(row.end)
                if start in hub_nodes and end in pos_by_node:
                    dest_xy.append(pos_by_node[end])
                elif end in hub_nodes and start in pos_by_node:
                    origin_xy.append(pos_by_node[start])

            if origin_xy:
                ox, oy = zip(*origin_xy)
                ox = np.array(ox) + rng.uniform(-jitter_scale, jitter_scale, len(ox))
                oy = np.array(oy) + rng.uniform(-jitter_scale, jitter_scale, len(oy))
                ax.scatter(ox, oy, s=30, color=color, alpha=0.4, label=f"origins{label_suffix}", zorder=4)
            if dest_xy:
                dx, dy = zip(*dest_xy)
                dx = np.array(dx) + rng.uniform(-jitter_scale, jitter_scale, len(dx))
                dy = np.array(dy) + rng.uniform(-jitter_scale, jitter_scale, len(dy))
                ax.scatter(dx, dy, s=30, color=color, alpha=0.4, marker="x", label=f"destinations{label_suffix}", zorder=4)

    ax.set_title(f"Network Setup: {os.path.basename(network_dir)}")
    ax.set_xlabel("length [m]")
    ax.set_ylabel("width [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()

    plt.close(fig)


def main():
    network_dir = resolve_network_dir()

    if os.path.isabs(OUTPUT_FILE):
        output_path = OUTPUT_FILE
    else:
        output_path = os.path.join(network_dir, OUTPUT_FILE)

    demand_df = None
    if DEMAND_FILE is not None:
        demand_df = pd.read_csv(DEMAND_FILE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visualize_network(network_dir, output_path, demand_df=demand_df, show=SHOW_FIGURE)
    print(f"Saved network visualization: {output_path}")


if __name__ == "__main__":
    main()
