#!/usr/bin/env python3
"""Match network nodes to zone polygons and write node_zone_info.csv.

Usage:
  python match_nodes_to_zones.py ZONE_SYSTEM NETWORK_NAME

Examples:
  python match_nodes_to_zones.py Chicago_2023_6min_max Chicago_2023

The script looks for:
- zones/polygon_definition.geojson under data/zones/{ZONE_SYSTEM}/polygon_definition.geojson
- network node files under data/networks/{NETWORK_NAME}/ (searches for nodes*.geojson or nodes*.csv)

Output:
- data/zones/{ZONE_SYSTEM}/{NETWORK_NAME}/node_zone_info.csv
  with columns: node_index, zone_id (-1 if none)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"


def find_nodes_file(network_dir: Path) -> Path | None:
    # prefer geojson named nodes or nodes_all_infos, else csv
    patterns = ["nodes_all_infos.geojson", "nodes*.geojson", "nodes*.json", "nodes*.csv", "*.nodes.geojson"]
    for root, dirs, files in os.walk(network_dir):
        for fname in files:
            for pat in patterns:
                # simple glob-like match
                if pat.startswith("nodes*") and fname.startswith("nodes") and (fname.endswith('.geojson') or fname.endswith('.json') or fname.endswith('.csv')):
                    return Path(root) / fname
                if pat == "nodes_all_infos.geojson" and fname == pat:
                    return Path(root) / fname
    return None


def load_zones(zone_dir: Path) -> gpd.GeoDataFrame:
    path = zone_dir / "polygon_definition.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Missing polygon_definition.geojson in {zone_dir}")
    zones = gpd.read_file(path)
    if "zone_id" not in zones.columns:
        raise ValueError(f"polygon_definition.geojson must contain 'zone_id' property")
    return zones


def load_nodes(nodes_path: Path, zones_crs) -> gpd.GeoDataFrame:
    if nodes_path.suffix.lower() in (".geojson", ".json"):
        nodes = gpd.read_file(nodes_path)
        # ensure geometry column exists
        if nodes.geometry.is_empty.all():
            raise ValueError(f"No geometries in {nodes_path}")
        if zones_crs is not None and nodes.crs is not None and nodes.crs != zones_crs:
            nodes = nodes.to_crs(zones_crs)
        elif zones_crs is not None and nodes.crs is None:
            nodes.set_crs(zones_crs, inplace=True)
        return nodes

    # assume CSV with x,y columns
    df = pd.read_csv(nodes_path)
    # common coordinate column names (include pos_x/pos_y used in example networks)
    x_cols = [c for c in df.columns if c.lower() in ("x", "lon", "longitude", "pos_x", "posx", "easting", "east")]
    y_cols = [c for c in df.columns if c.lower() in ("y", "lat", "latitude", "pos_y", "posy", "northing", "north")]
    if not x_cols or not y_cols:
        raise ValueError(f"CSV nodes file must contain x/y or lon/lat columns: {nodes_path}")
    xcol = x_cols[0]
    ycol = y_cols[0]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=[Point(xy) for xy in zip(df[xcol], df[ycol])])
    if zones_crs is not None:
        gdf.set_crs(zones_crs, inplace=True)
    return gdf


def detect_node_id_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = ["node_index", "nodeId", "node_id", "id", "ID", "osmid", "node"]
    for c in candidates:
        if c in gdf.columns:
            return c
    # fallback to the index
    return "__index__"


def main():
    parser = argparse.ArgumentParser(description="Match network nodes to zones and write node_zone_info.csv")
    parser.add_argument("zone_system", help="Folder name under data/zones containing polygon_definition.geojson")
    parser.add_argument("network_name", help="Folder name under data/networks containing node files")
    parser.add_argument("--nodes-file", help="Optional explicit path to nodes file (overrides auto-discovery)")
    args = parser.parse_args()

    zone_dir = DATA_DIR / "zones" / args.zone_system
    network_dir = DATA_DIR / "networks" / args.network_name

    if not zone_dir.exists():
        print(f"Zone folder not found: {zone_dir}")
        sys.exit(2)
    if not network_dir.exists():
        print(f"Network folder not found: {network_dir}")
        sys.exit(2)

    zones = load_zones(zone_dir)

    # find nodes file
    nodes_path = Path(args.nodes_file) if args.nodes_file else find_nodes_file(network_dir)
    if nodes_path is None or not nodes_path.exists():
        print(f"Could not auto-detect nodes file under {network_dir}. Use --nodes-file to specify.")
        sys.exit(3)

    nodes = load_nodes(nodes_path, zones.crs)

    node_id_col = detect_node_id_column(nodes)
    if node_id_col == "__index__":
        nodes = nodes.reset_index().rename(columns={"index": "node_index"})
        node_id_col = "node_index"

    # spatial join: which polygon contains each node
    # ensure same crs
    if nodes.crs != zones.crs:
        try:
            nodes = nodes.to_crs(zones.crs)
        except Exception:
            pass

    joined = gpd.sjoin(nodes, zones[["zone_id", "geometry"]], how="left", predicate="within")

    out_df = pd.DataFrame()
    out_df["node_index"] = joined[node_id_col]
    # replace NaN zone_id with -1
    out_df["zone_id"] = joined["zone_id"].where(joined["zone_id"].notna(), -1)

    # coerce numeric zone ids to int when possible
    try:
        out_df["zone_id"] = out_df["zone_id"].astype(int)
    except Exception:
        pass

    # ensure output directory
    out_dir = zone_dir / args.network_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "node_zone_info.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
