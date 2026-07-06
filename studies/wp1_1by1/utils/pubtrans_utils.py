import json
import os

import pandas as pd

from utils.demand_utils import get_hubs_for_network
from utils.network_utils import get_row_cols

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PT_DIR = os.path.join(REPO_ROOT, "data", "pubtrans")


def _middle_row_stations(rows, cols, cell_size, station_spacing_m, hub_node_index, x_lo=None, x_hi=None):
    """Return (stations, terminus_station_id). Stations run along the middle row, spaced out
    from the hub (the corridor's terminus/anchor) rather than from the grid edge, so that
    station_spacing_m is respected between the hub and its neighboring stations instead of
    being absorbed into a leftover fractional gap at whichever end the hub happens to sit on.

    x_lo/x_hi bound the along-corridor (x) span the stations may cover (in m). They default to
    the whole grid; for the two-hub case each half-line is bounded to [hub, corridor midpoint] so
    the two lines meet (but do not overlap) at the middle."""
    mid_row = rows // 2      # should use the same mid_row as the hub node
    pos_y = mid_row * cell_size
    grid_length_m = (cols - 1) * cell_size
    if x_lo is None:
        x_lo = 0.0
    if x_hi is None:
        x_hi = grid_length_m

    hub_col = hub_node_index - mid_row * cols
    hub_x = round(hub_col * cell_size, 6)

    xs = [hub_x]
    x = hub_x - station_spacing_m
    while x >= x_lo - 1e-6:
        xs.append(round(x, 6))
        x -= station_spacing_m
    x = hub_x + station_spacing_m
    while x <= x_hi + 1e-6:
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


def _build_line_defs(rows, cols, cell_size, station_spacing_m, hubs, speed_kmh, vehicle_type,
                     boarding_time_s):
    """Build the per-line station/schedule/alignment definitions for one network x station_spacing.

    One line for a single hub (spanning the whole corridor), or two half-lines for two hubs (each
    hub -> corridor midpoint and back). Station ids are made globally unique across the returned
    lines (each subsequent line's ids are offset), so they can be concatenated into one stations.csv
    that the fleet control reads into a single station_dict.

    Returns a list of dicts, one per line: line_id, terminus_station_id, hub_node, ordered (station
    dicts, hub-first), schedule_rows, route_length_km, round_trip_time.
    """
    grid_length_m = (cols - 1) * cell_size
    x_mid = grid_length_m / 2.0
    mid_row = rows // 2

    line_defs = []
    sid_offset = 0
    for li, hub_node_index in enumerate(hubs):
        line_id = li + 1
        hub_col = hub_node_index - mid_row * cols
        hub_x = hub_col * cell_size

        if len(hubs) == 1:
            x_lo, x_hi = 0.0, grid_length_m
        elif hub_x <= x_mid:  # left hub -> covers [start, midpoint]
            x_lo, x_hi = 0.0, x_mid
        else:                 # right hub -> covers [midpoint, end]
            x_lo, x_hi = x_mid, grid_length_m

        stations, terminus_station_id = _middle_row_stations(
            rows, cols, cell_size, station_spacing_m, hub_node_index, x_lo=x_lo, x_hi=x_hi)
        # offset station ids so they are unique across lines
        for s in stations:
            s["station_id"] += sid_offset
        terminus_station_id += sid_offset
        sid_offset += len(stations)

        # hub-first ordering (terminus at index 0, then increasing distance from the hub), so the
        # alignment/schedule fixed portion starts at the hub regardless of which end the hub is on
        ordered = sorted(stations, key=lambda s: abs(s["pos_x"] - hub_x))
        assert ordered[0]["station_id"] == terminus_station_id

        schedule_rows, round_trip_time = _round_trip_schedule(
            ordered, speed_kmh, line_id, vehicle_type, boarding_time_s)
        route_length_km = abs(ordered[0]["pos_x"] - ordered[-1]["pos_x"]) / 1000

        line_defs.append({
            "line_id": line_id,
            "terminus_station_id": terminus_station_id,
            "hub_node": hub_node_index,
            "ordered": ordered,
            "schedule_rows": schedule_rows,
            "route_length_km": route_length_km,
            "round_trip_time": round_trip_time,
        })
    return line_defs


def generate_pubtrans(ranges):
    """Generate hub-anchored corridor-line PT infrastructure (stations.csv, schedules.csv,
    alignment geojson) for each network x station_spacing x headway combination. 
    Generated once regardless of which service types consume it e.g. fixed lines and sod.

    For a single-hub network one line spans the whole corridor; for a two-hub network two
    half-lines are generated, each running hub -> corridor midpoint and back to the same hub.

    Returns a list of dicts: pt_name, network_name, station_spacing_m, headway_min, lines. `lines`
    carries per-line info (line_id, terminus_station_id, hub_node, route_length_km) -- one entry for
    a single-hub network, two half-lines for a two-hub network. Consumers must read per-line values
    from `lines`.
    """
    nw_ranges = ranges["network"]
    pt_ranges = ranges.get("pubtrans", {})

    headways_min = pt_ranges.get("headways_min", [10])
    station_spacings_m = pt_ranges.get("station_spacings_m", [200])
    boarding_time_s = pt_ranges.get("boarding_time_s", 30)
    vehicle_type = pt_ranges.get("vehicle_type", "veh_20")
    speed_kmh = nw_ranges["default_speed"]
    cell_size = nw_ranges["cell_size"]

    pt_variants = []

    for length in nw_ranges["lengths"]:
        for width in nw_ranges["widths"]:
            for n_hubs in nw_ranges["num_hubs"]:
                nw_name = f"grid_l{length}_w{width}_hubs{n_hubs}_cell{cell_size}"
                hubs = get_hubs_for_network(nw_name)
                if len(hubs) not in (1, 2):
                    raise ValueError(
                        f"PT line generation supports 1 or 2 hubs per network; "
                        f"{nw_name} has {len(hubs)}."
                    )
                rows, cols = get_row_cols(length, width, cell_size)

                for station_spacing_m in station_spacings_m:
                    line_defs = _build_line_defs(
                        rows, cols, cell_size, station_spacing_m, hubs, speed_kmh, vehicle_type,
                        boarding_time_s)

                    # combined stations across all lines (unique ids); combined schedule
                    all_stations = [s for ld in line_defs for s in ld["ordered"]]
                    all_stations = sorted(all_stations, key=lambda s: s["station_id"])
                    all_schedule_rows = [r for ld in line_defs for r in ld["schedule_rows"]]

                    for hw_min in headways_min:
                        pt_name = f"{nw_name}_mid_sp{station_spacing_m}_hw{hw_min}"
                        pt_out_dir = os.path.join(PT_DIR, pt_name)
                        os.makedirs(pt_out_dir, exist_ok=True)

                        pd.DataFrame(
                            [{"station_id": s["station_id"], "network_node_index": s["node_index"]}
                             for s in all_stations]
                        ).to_csv(os.path.join(pt_out_dir, "stations.csv"), index=False)

                        pd.DataFrame(all_schedule_rows).to_csv(
                            os.path.join(pt_out_dir, "schedules.csv"), index=False)

                        for ld in line_defs:
                            _write_alignment_geojson(ld["ordered"], pt_out_dir, ld["line_id"], pt_name)

                        pt_variants.append({
                            "pt_name": pt_name,
                            "network_name": nw_name,
                            "station_spacing_m": station_spacing_m,
                            "headway_min": hw_min,
                            "lines": [
                                {"line_id": ld["line_id"],
                                 "terminus_station_id": ld["terminus_station_id"],
                                 "hub_node": ld["hub_node"],
                                 "route_length_km": ld["route_length_km"]}
                                for ld in line_defs
                            ],
                        })

    return pt_variants
