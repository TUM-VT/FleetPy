import json
import os

import pandas as pd

from utils.demand_utils import get_hubs_for_network
from utils.network_utils import get_row_cols

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PT_DIR = os.path.join(REPO_ROOT, "data", "pubtrans")


def _middle_row_stations(rows, cols, cell_size, station_spacing_m, hub_node_index):
    """Return (stations, terminus_station_id). Stations run along the middle row, spaced out
    from the hub (the corridor's terminus/anchor) rather than from the grid edge, so that
    station_spacing_m is respected between the hub and its neighboring stations instead of
    being absorbed into a leftover fractional gap at whichever end the hub happens to sit on."""
    mid_row = rows // 2
    pos_y = mid_row * cell_size
    grid_length_m = (cols - 1) * cell_size

    hub_col = hub_node_index - mid_row * cols
    hub_x = round(hub_col * cell_size, 6)

    xs = [hub_x]
    x = hub_x - station_spacing_m
    while x >= -1e-6:
        xs.append(round(x, 6))
        x -= station_spacing_m
    x = hub_x + station_spacing_m
    while x <= grid_length_m + 1e-6:
        xs.append(round(x, 6))
        x += station_spacing_m
    xs = sorted(set(xs))

    stations = []
    terminus_station_id = None
    for sid, sx in enumerate(xs):
        col = round(sx / cell_size)
        node_index = mid_row * cols + col
        stations.append({"station_id": sid, "node_index": node_index, "pos_x": sx, "pos_y": pos_y})
        if node_index == hub_node_index:
            terminus_station_id = sid

    if terminus_station_id is None:
        raise ValueError("Hub node was not placed on any generated station.")

    return stations, terminus_station_id


def _round_trip_schedule(ordered_stations, speed_kmh, line_id, vehicle_type, boarding_time_s):
    """ordered_stations must be hub-first (index 0 = terminus, last = far end of the corridor).
    Emits a single trip_id=0 round trip (terminus -> far end -> terminus) starting at departure=0;
    pt_regular_headway (a separate scenario parameter) drives runtime dispatch, this file is only
    a one-shot schedule template. Every station, including both endpoints, must appear twice in the
    round trip -- SemiOnDemandBatchAssignmentFleetcontrol.set_veh_plan_schedule looks up the second
    occurrence of the fixed/flexible boundary station's departure time to derive return-leg timing.
    Returns (rows, round_trip_time_s)."""
    speed_ms = speed_kmh * 1000 / 3600
    rows = []
    departure = 0.0
    prev = None

    def _append(st):
        rows.append({"departure": round(departure), "station_id": st["station_id"], "trip_id": 0,
                     "line_vehicle_id": 0, "LINE": line_id, "vehicle_type": vehicle_type})

    for st in ordered_stations:
        if prev is not None:
            departure += abs(prev["pos_x"] - st["pos_x"]) / speed_ms + boarding_time_s
        _append(st)
        prev = st

    # turnaround dwell at the far end (duplicate row, same station, incremented time)
    departure += boarding_time_s
    _append(ordered_stations[-1])

    for st in ordered_stations[-2::-1]:
        departure += abs(prev["pos_x"] - st["pos_x"]) / speed_ms + boarding_time_s
        _append(st)
        prev = st

    return rows, departure


def _write_alignment_geojson(ordered_stations, pt_out_dir, line_id, pt_name):
    """ordered_stations must be hub-first: PtLine measures km-run as distance from the geojson's
    first coordinate, and check_point_flexible treats everything beyond pt_fixed_length km from
    that start as the flexible zone -- so the fixed portion must start at the hub."""
    coords = [[s["pos_x"], s["pos_y"]] for s in ordered_stations]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "name": pt_name,
                    "description": f"Hub-anchored corridor line. Stations: "
                                    f"{[s['station_id'] for s in ordered_stations]}.",
                },
            }
        ],
    }
    path = os.path.join(pt_out_dir, f"{line_id}_line_alignment.geojson")
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)

    return abs(ordered_stations[0]["pos_x"] - ordered_stations[-1]["pos_x"]) / 1000


def generate_pubtrans(ranges):
    """Generate hub-anchored corridor-line PT infrastructure (stations.csv, schedules.csv,
    alignment geojson) for each network x station_spacing x headway combination. `sod` and
    `fixed_line` service types share these same files (they only differ in the pt_fixed_length
    scenario value, not in the PT infrastructure itself), so this is generated once regardless of
    which/how many service types consume it.

    Returns a list of dicts: pt_name, network_name, terminus_station_id, route_length_km,
    station_spacing_m, headway_min.
    """
    nw_ranges = ranges["network"]
    pt_ranges = ranges.get("pubtrans", {})

    headways_min = pt_ranges.get("headways_min", [10])
    station_spacings_m = pt_ranges.get("station_spacings_m", [200])
    boarding_time_s = pt_ranges.get("boarding_time_s", 30)
    vehicle_type = pt_ranges.get("vehicle_type", "veh_20")
    line_id = pt_ranges.get("line_id", 1)
    speed_kmh = nw_ranges["default_speed"]
    cell_size = nw_ranges["cell_size"]

    pt_variants = []

    for length in nw_ranges["lengths"]:
        for width in nw_ranges["widths"]:
            for n_hubs in nw_ranges["num_hubs"]:
                nw_name = f"grid_l{length}_w{width}_hubs{n_hubs}_cell{cell_size}"
                hubs = get_hubs_for_network(nw_name)
                if len(hubs) != 1:
                    raise ValueError(
                        f"PT line generation requires exactly 1 hub per network; "
                        f"{nw_name} has {len(hubs)}. Multi-hub PT lines are out of scope."
                    )
                hub_node_index = hubs[0]
                rows, cols = get_row_cols(length, width, cell_size)

                for station_spacing_m in station_spacings_m:
                    stations, terminus_station_id = _middle_row_stations(
                        rows, cols, cell_size, station_spacing_m, hub_node_index)
                    ordered = sorted(stations, key=lambda s: s["pos_x"], reverse=True)
                    assert ordered[0]["station_id"] == terminus_station_id

                    for hw_min in headways_min:
                        headway_s = hw_min * 60
                        pt_name = f"{nw_name}_mid_sp{station_spacing_m}_hw{hw_min}"
                        pt_out_dir = os.path.join(PT_DIR, pt_name)
                        os.makedirs(pt_out_dir, exist_ok=True)

                        pd.DataFrame(
                            [{"station_id": s["station_id"], "network_node_index": s["node_index"]}
                             for s in ordered]
                        ).to_csv(os.path.join(pt_out_dir, "stations.csv"), index=False)

                        schedule_rows, round_trip_time = _round_trip_schedule(
                            ordered, speed_kmh, line_id, vehicle_type, boarding_time_s)
                        pd.DataFrame(schedule_rows).to_csv(
                            os.path.join(pt_out_dir, "schedules.csv"), index=False)

                        route_length_km = _write_alignment_geojson(ordered, pt_out_dir, line_id, pt_name)

                        if round_trip_time > headway_s:
                            print(
                                f"  WARNING: {pt_name}: round trip {round_trip_time:.0f}s exceeds "
                                f"headway {headway_s}s -- more than 1 vehicle will be needed to "
                                f"sustain this headway at runtime (set pt_n_veh accordingly)."
                            )

                        pt_variants.append({
                            "pt_name": pt_name,
                            "network_name": nw_name,
                            "terminus_station_id": terminus_station_id,
                            "route_length_km": route_length_km,
                            "station_spacing_m": station_spacing_m,
                            "headway_min": hw_min,
                        })

    return pt_variants
