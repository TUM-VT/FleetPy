import functools
import json
import os

import pandas as pd
import yaml

from utils.demand_utils import get_hubs_for_network
from utils.network_utils import get_row_cols
from utils.supply_utils import fleet_entries, scenario_row

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PT_DIR = os.path.join(REPO_ROOT, "data", "pubtrans")
CONST_CFG_PATH = os.path.join(REPO_ROOT, "studies", "wp1_1by1", "scenarios", "const_cfg.yaml")

# sentinel fixed-route length (km) meaning "the whole line is fixed" (fixed_line service)
PT_FIXED_LENGTH_SENTINEL_KM = 999999


@functools.lru_cache(maxsize=1)
def _load_const_cfg():
    with open(CONST_CFG_PATH) as f:
        return yaml.safe_load(f)


def _const_cfg_default(key, fallback):
    """Read key's default from const_cfg.yaml, falling back to `fallback` if the file or key is
    missing. Used for defaults that must still be resolved to an explicit per-row value (e.g.
    a sweepable axis) rather than left unset for const_cfg.yaml's own NaN-backfill to fill in."""
    try:
        return _load_const_cfg().get(key, fallback)
    except (OSError, yaml.YAMLError):
        return fallback


def _normalize_spacing(spacing):
    """A spacing spec is either a plain number (uniform spacing) or a (near, far, radius) tuple/
    list (hybrid: near_spacing within radius_m of the hub, far_spacing beyond it) -- returns
    (near, far, radius) with radius=None meaning "never switches" (uniform)."""
    if isinstance(spacing, (tuple, list)):
        near, far, radius = spacing
        return near, far, radius
    return spacing, spacing, None


def _spacing_tag(spacing):
    """String tag for pt_name/scenario_name -- "sp500" for uniform, "sp200-750r1000" for hybrid."""
    if isinstance(spacing, (tuple, list)):
        near, far, radius = spacing
        return f"{near}-{far}r{radius}"
    return f"{spacing}"


def _spaced_offsets(near_spacing, far_spacing, near_radius, max_extent):
    """Positive offsets from the hub (0) out to max_extent, stepping by near_spacing while still
    within near_radius of the hub and by far_spacing beyond it. near_radius=None means always
    near_spacing (uniform case). Always includes max_extent itself as the last offset (the
    corridor's far terminus gets a station even if the last regular step overshoots it and gets
    dropped, leaving a station-less leftover segment shorter than far_spacing -- see
    hybrid-spacing discussion in main_plots.ipynb for why a leftover gap at the corridor end is
    otherwise silently unstationed)."""
    if near_radius is None:
        near_radius = max_extent + 1
    offsets = []
    x = 0.0
    while True:
        step = near_spacing if x < near_radius - 1e-6 else far_spacing
        x += step
        if x > max_extent + 1e-6:
            break
        offsets.append(x)
    if max_extent > 1e-6 and (not offsets or offsets[-1] < max_extent - 1e-6):
        offsets.append(max_extent)
    return offsets


def _middle_row_stations(rows, cols, cell_size, station_spacing_m, hub_node_index, x_lo=None, x_hi=None):
    """Return (stations, terminus_station_id). Stations run along the middle row, spaced out
    from the hub (the corridor's terminus/anchor) rather than from the grid edge, so that
    station_spacing_m is respected between the hub and its neighboring stations instead of
    being absorbed into a leftover fractional gap at whichever end the hub happens to sit on.

    station_spacing_m is either a plain number (uniform spacing) or a (near, far, radius) tuple
    for hybrid spacing -- near_spacing within radius_m of the hub, far_spacing beyond it (finer
    granularity for the short first/last-mile trips concentrated near the hub, without paying
    that dwell-time overhead across the whole corridor -- see main_plots.ipynb discussion of why
    uniform tight spacing kills long-trip feasibility).

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
    near, far, radius = _normalize_spacing(station_spacing_m)

    xs = [hub_x]
    for off in _spaced_offsets(near, far, radius, hub_x - x_lo):
        xs.append(round(hub_x - off, 6))
    for off in _spaced_offsets(near, far, radius, x_hi - hub_x):
        xs.append(round(hub_x + off, 6))
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
                     boarding_time_s, num_parallel_lines=1):
    """Build the per-line station/schedule/alignment definitions for one network x station_spacing.

    One line for a single hub (spanning the whole corridor), or two half-lines for two hubs (each
    hub -> corridor midpoint and back). num_parallel_lines > 1 replicates each hub's line that many
    times -- same route/stations, each replica gets its own line_id and its own vehicle subset (see
    SemiOnDemandBatchAssignmentFleetcontrol._pick_least_loaded_line for how a request picks among
    parallel lines sharing a hub), so sod's flexible-zone throughput isn't capped by a single
    line's headway/insertion-feasibility bottleneck. Station ids are made globally unique across
    the returned lines (each subsequent line's ids are offset, even replicas of the same physical
    route), so they can be concatenated into one stations.csv that the fleet control reads into a
    single station_dict.

    Returns a list of dicts, one per line: line_id, terminus_station_id, hub_node, ordered (station
    dicts, hub-first), schedule_rows, route_length_km, round_trip_time.
    """
    grid_length_m = (cols - 1) * cell_size
    x_mid = grid_length_m / 2.0
    mid_row = rows // 2

    line_defs = []
    sid_offset = 0
    line_id = 0
    for hub_node_index in hubs:
        hub_col = hub_node_index - mid_row * cols
        hub_x = hub_col * cell_size

        if len(hubs) == 1:
            x_lo, x_hi = 0.0, grid_length_m
        elif hub_x <= x_mid:  # left hub -> covers [start, midpoint]
            x_lo, x_hi = 0.0, x_mid
        else:                 # right hub -> covers [midpoint, end]
            x_lo, x_hi = x_mid, grid_length_m

        for _ in range(num_parallel_lines):
            line_id += 1
            stations, terminus_station_id = _middle_row_stations(
                rows, cols, cell_size, station_spacing_m, hub_node_index, x_lo=x_lo, x_hi=x_hi)
            # offset station ids so they are unique across lines (including parallel replicas of
            # the same physical route)
            for s in stations:
                s["station_id"] += sid_offset
            terminus_station_id += sid_offset
            sid_offset += len(stations)

            # hub-first ordering (terminus at index 0, then increasing distance from the hub), so
            # the alignment/schedule fixed portion starts at the hub regardless of which end the
            # hub is on
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
    alignment geojson) for each network x station_spacing x headway x num_parallel_lines x
    vehicle_type combination.

    For a single-hub network one line spans the whole corridor; for a two-hub network two
    half-lines are generated, each running hub -> corridor midpoint and back to the same hub.

    vehicle_type is baked into schedules.csv (read back into vehicles_to_initialize by
    SemiOnDemandBatchAssignmentFleetcontrol), so a separate schedule set must be generated per
    distinct veh_type actually used by a PT-line service type -- one ranges file's service_types
    block commonly mixes several (e.g. sod's 20-seat vs sod_8's 8-seat vs fixed_line_44's 44-seat
    variant, all under one shared pubtrans block). The vehicle types generated here are derived
    from ranges["service_types"]'s own veh_type fields (for PT-line services only), NOT a single
    ranges["pubtrans"]["vehicle_type"] -- using one type for all of them previously left every
    non-matching service silently running its scheduled vehicles as whatever type was generated
    (e.g. sod_8/fixed_line_44 actually dispatching 20-seat scheduled vehicles). Falls back to
    pubtrans.vehicle_type (or "veh_20") only when service_types defines no PT-line service (e.g.
    other callers/tests).

    Returns a list of dicts: pt_name, network_name, station_spacing_m, headway_min,
    num_parallel_lines, vehicle_type, lines. `lines` carries per-line info (line_id,
    terminus_station_id, hub_node, route_length_km) -- one entry for a single-hub network, two
    half-lines for a two-hub network (each further split num_parallel_lines-many ways).
    Consumers must read per-line values from `lines`.
    """
    nw_ranges = ranges["network"]
    pt_ranges = ranges.get("pubtrans", {})

    headways_min = pt_ranges.get("headways_min", [10])
    # a hybrid spacing spec loads from YAML as a list ([near, far, radius]) -- normalize to a
    # tuple so it's hashable (used as part of the pt_variant_lookup dict key below)
    station_spacings_m = [tuple(s) if isinstance(s, list) else s
                           for s in pt_ranges.get("station_spacings_m", [200])]
    boarding_time_s = pt_ranges.get("boarding_time_s", 30)
    vehicle_types = sorted({
        st_cfg["veh_type"] for st_cfg in ranges.get("service_types", {}).values()
        if is_pt_line_service(st_cfg)
    }) or [pt_ranges.get("vehicle_type", "veh_20")]
    speed_kmh = nw_ranges["default_speed"]
    cell_size = nw_ranges["cell_size"]
    # parallel lines per hub, sharing the same physical route -- lets a service's fleet be split
    # across independent dispatch queues instead of one line's headway/insertion bottleneck.
    # Scalar or list (like every other axis here, see _as_list) -- swept, each value gets its
    # own PT variant (see pl_tag below), consumed by every station_spacing x headway combo. Lives
    # under pubtrans: (not network:) -- it's purely a PT-infrastructure generation axis, never
    # consumed by generate_networks.
    num_parallel_lines_list = _as_list(pt_ranges.get("num_parallel_lines", 1))

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
                    for num_parallel_lines in num_parallel_lines_list:
                        for vehicle_type in vehicle_types:
                            line_defs = _build_line_defs(
                                rows, cols, cell_size, station_spacing_m, hubs, speed_kmh, vehicle_type,
                                boarding_time_s, num_parallel_lines=num_parallel_lines)

                            # combined stations across all lines (unique ids); combined schedule
                            all_stations = [s for ld in line_defs for s in ld["ordered"]]
                            all_stations = sorted(all_stations, key=lambda s: s["station_id"])
                            all_schedule_rows = [r for ld in line_defs for r in ld["schedule_rows"]]

                            for hw_min in headways_min:
                                pl_tag = f"_pl{num_parallel_lines}" if num_parallel_lines != 1 else ""
                                # vehicle_type must be part of the on-disk directory key -- see
                                # this function's docstring for why (schedules.csv is
                                # vehicle-type-specific, and distinct services can share every
                                # other axis here).
                                pt_name = f"{nw_name}_mid_sp{_spacing_tag(station_spacing_m)}_hw{hw_min}_{vehicle_type}{pl_tag}"
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
                                    "num_parallel_lines": num_parallel_lines,
                                    "vehicle_type": vehicle_type,
                                    "lines": [
                                        {"line_id": ld["line_id"],
                                         "terminus_station_id": ld["terminus_station_id"],
                                         "hub_node": ld["hub_node"],
                                         "route_length_km": ld["route_length_km"]}
                                        for ld in line_defs
                                    ],
                                })

    return pt_variants


# --- PT-line scenario config (sod / fixed_line service types) ---

def is_pt_line_service(st_cfg):
    """True for the PT-line service types (sod, fixed_line) that ride on a generated PT variant."""
    return st_cfg.get("op_module") == "SemiOnDemandBatchAssignmentFleetcontrol"


def pt_variant_lookup(pt_variants):
    """(network_name, station_spacing_m, headway_min, vehicle_type) -> list of PT variants, one
    per num_parallel_lines value generated for that combo (see generate_pubtrans). A service
    type doesn't pick num_parallel_lines itself -- pt_scenario_rows fans out over every variant
    found here, so it's purely a network-level generation axis. vehicle_type IS picked by the
    service (its own veh_type), which is why it's part of this key."""
    lookup = {}
    for v in pt_variants:
        key = (v["network_name"], v["station_spacing_m"], v["headway_min"], v["vehicle_type"])
        lookup.setdefault(key, []).append(v)
    return lookup


def _split_fleet_across_lines(n, pt_variant, hub_counts):
    """Split a fleet of n vehicles evenly across a PT variant's lines.

    Single-line variants return the scalar n unchanged. Multi-line (two-hub) variants split n
    equally across the lines (largest-remainder apportionment, at least one vehicle per line) and
    return a "line:n,line:n" mapping string consumed by
    SemiOnDemandBatchAssignmentFleetcontrol._parse_n_veh_per_line. This mirrors the even 50/50 hub
    split used for the dtd/stops services. hub_counts (per-hub demand) is accepted but unused for
    now; restore demand-proportional weights here to split by demand instead.
    """
    lines = pt_variant.get("lines")
    if not lines or len(lines) <= 1:
        return n

    # Even split across lines (weights all equal); odd remainders go to the first line(s).
    k = len(lines)
    weights = [1] * k
    total_w = k

    if n <= k:
        counts = [1] * k  # over-provision a bit rather than leave a line with no vehicle
    else:
        rem = n - k
        shares = [w / total_w * rem for w in weights]
        base = [int(s) for s in shares]
        counts = [1 + b for b in base]
        leftover = rem - sum(base)
        order = sorted(range(k), key=lambda i: shares[i] - base[i], reverse=True)
        for i in range(leftover):
            counts[order[i]] += 1

    # ";"-delimited so FleetPy's config loader (decode_config_str) parses it into a {line_id: n} dict
    return ";".join(f"{ld['line_id']}:{c}" for ld, c in zip(lines, counts))


def _as_list(v):
    """Coerce a scalar-or-list scenario field to a list, so a design axis can be either pinned
    (scalar) or swept (list) in the ranges yaml without changing the config schema."""
    return v if isinstance(v, list) else [v]


def _pt_extra_cols(st_cfg, pt_variant, headway_min, fixed_length_km, n_veh,
                   flex_detour, zone_min_detour_time, zone_max_detour_time, dispatch_delay,
                   line_id, wait_time_2):
    cols = {
        "gtfs_name": pt_variant["pt_name"],
        "station_file": "stations.csv",
        "schedule_file": "schedules.csv",
        "alignment_file": "{line_id}_line_alignment.geojson",
        "line_id": line_id,
        "pt_route_id": line_id,
        "pt_regular_headway": headway_min * 60,
        "pt_fixed_length": fixed_length_km,
        "pt_flex_detour": flex_detour,
        "pt_zone_min_detour_time": zone_min_detour_time,
        "pt_zone_max_detour_time": zone_max_detour_time,
        "pt_dispatch_delay": dispatch_delay,
        "pt_n_veh": n_veh,

        "walking_speed": 5,  # matches the "normal" demand-side user group's own walking_speed
                             # (see scenario_ranges *.yaml user_group_params.normal.walking_speed)

        # op_max_wait_time_2 is a sweepable axis (see pt_scenario_rows), so it needs an explicit
        # per-row value -- default sourced from const_cfg.yaml, see that file's comment on the key.
        "op_max_wait_time_2": wait_time_2,
    }
    # Left OUT of cols unless a ranges file overrides them -- same as dtd/stops (supply_utils.py)
    # -- so they flow from const_cfg.yaml's single shared default (via write_scenario_cfg's
    # NaN-backfill) instead of a second Python-side copy of the same value.
    for key in ("op_max_wait_time", "op_max_detour_time_factor", "user_max_decision_time"):
        if key in st_cfg:
            cols[key] = st_cfg[key]
    return cols


def pt_scenario_rows(base_name, dv, st_name, st_cfg, sim_end_time, variant_lookup, default_line_id=1):
    """Build scenario rows for a PT-line service type (sod/fixed_line), sweeping station spacing,
    headway, fleet size, and (for sod only) the fixed-route/flexible split of the corridor.

    line_id (the FleetPy-facing route id, not to be confused with the per-replica line_id used
    internally for parallel-line splitting in generate_pubtrans/_split_fleet_across_lines) is the
    same static value for every PT-line service in a ranges file, so it's sourced from the shared
    pubtrans.line_id default (passed in as default_line_id) rather than repeated in every
    service_types block -- a service can still set its own st_cfg["line_id"] to override."""
    rows = []
    is_fixed_line = st_cfg.get("fixed_line", False)
    line_id = st_cfg.get("line_id", default_line_id)

    # Detour/flex-zone design axes may be pinned (scalar) or swept (list). A tag is only appended
    # to the scenario name when an axis is actually swept, so existing single-value runs keep their
    # current names.
    flex_detours = _as_list(st_cfg["pt_flex_detour"])
    zone_min_times = _as_list(st_cfg["pt_zone_min_detour_time"])
    zone_max_times = _as_list(st_cfg["pt_zone_max_detour_time"])
    dispatch_delays = _as_list(st_cfg["pt_dispatch_delay"])
    # op_max_wait_time_2 is a sweepable axis (see const_cfg.yaml's comment on this key); user_max_
    # decision_time/op_max_wait_time flow from const_cfg.yaml unconditionally (see _pt_extra_cols).
    wait_time_2s = _as_list(st_cfg.get("op_max_wait_time_2", _const_cfg_default("op_max_wait_time_2", 900)))

    # normalize hybrid specs ([near, far, radius] lists) to tuples, matching generate_pubtrans's
    # pt_variant_lookup key normalization
    station_spacings_m = [tuple(s) if isinstance(s, list) else s for s in st_cfg["station_spacings_m"]]
    for station_spacing_m in station_spacings_m:
        for headway_min in st_cfg["headways_min"]:
            key = (dv["network_name"], station_spacing_m, headway_min, st_cfg["veh_type"])
            variants = variant_lookup.get(key)
            if not variants:
                raise KeyError("No PT variant generated.")

            # num_parallel_lines is a network-level generation axis (see generate_pubtrans) --
            # by default every variant found at this key is used, one scenario set per value. A
            # service can instead pick its own num_parallel_lines subset (scalar or list) to
            # differ from other services sharing the same generated variants.
            if "num_parallel_lines" in st_cfg:
                wanted = set(_as_list(st_cfg["num_parallel_lines"]))
                variants = [v for v in variants if v.get("num_parallel_lines", 1) in wanted]
                if not variants:
                    raise KeyError(f"No PT variant with num_parallel_lines in {wanted} for {key}.")

            # pl_tag is only appended when more than one variant is actually in play here.
            for pt_variant in variants:
                num_parallel_lines = pt_variant.get("num_parallel_lines", 1)
                pl_tag = f"_pl{num_parallel_lines}" if len(variants) > 1 else ""

                if is_fixed_line:
                    fixed_length_variants = [(PT_FIXED_LENGTH_SENTINEL_KM, "full")]
                else:
                    # TODO (later if needed) same fixed length applies to both lines in a two-hub scenario
                    fixed_length_variants = [
                        (frac * pt_variant["lines"][0]["route_length_km"], f"fl{frac}")
                        for frac in st_cfg.get("fixed_length_fractions", [0])
                    ]
                    # TODO (later if needed) find_closest_station_to_x resolves the fixed/flex boundary to the hub itself
                    # if the fixed_length is shorter than the first non-hub station
                    sp_near = station_spacing_m[0] if isinstance(station_spacing_m, tuple) else station_spacing_m
                    for fixed_length_km, fl_tag in fixed_length_variants:
                        if fixed_length_km * 1000 < sp_near:
                            raise ValueError(
                                f"{st_name}/{fl_tag} at sp{_spacing_tag(station_spacing_m)}: fixed_length="
                                f"{fixed_length_km * 1000:.0f}m is shorter than the first non-hub "
                                f"station ({sp_near}m from hub) -- the fixed-route segment "
                                f"would degenerate to just the hub. Raise this fixed_length_fraction "
                                f"or reduce station_spacing_m."
                            )

                for n, size_tag in fleet_entries(st_cfg, dv["total_lambda"]):
                    # pt_n_veh is the total n for a single line, or a per-line "line:n,..." split for
                    # two hubs; op_fleet_composition (via scenario_row's n) always gets the total.
                    pt_n_veh = _split_fleet_across_lines(n, pt_variant, dv.get("hub_counts", {}))
                    for fixed_length_km, fl_tag in fixed_length_variants:
                        for flex_detour in flex_detours:
                            for zone_min_t in zone_min_times:
                                for zone_max_t in zone_max_times:
                                    for dispatch_delay in dispatch_delays:
                                        for wait_time_2 in wait_time_2s:
                                            detour_tag = ""
                                            if len(flex_detours) > 1:
                                                detour_tag += f"_fd{flex_detour}"
                                            if len(zone_min_times) > 1:
                                                detour_tag += f"_zmin{zone_min_t}"
                                            if len(zone_max_times) > 1:
                                                detour_tag += f"_zmax{zone_max_t}"
                                            if len(dispatch_delays) > 1:
                                                detour_tag += f"_dd{dispatch_delay}"
                                            if len(wait_time_2s) > 1:
                                                # "_wt2_" (not "_wt2") -- "wt2" ends in a digit, so
                                                # "_wt20"/"_wt2900" would glue the tag name to the
                                                # value unlike every other letter-ending tag.
                                                detour_tag += f"_wt2_{wait_time_2}"
                                            scenario_name = (
                                                f"{base_name}_{st_name}_sp{_spacing_tag(station_spacing_m)}_hw{headway_min}"
                                                f"{pl_tag}_{fl_tag}{detour_tag}_{size_tag}"
                                            )
                                            extra_cols = _pt_extra_cols(
                                                st_cfg, pt_variant, headway_min, fixed_length_km, pt_n_veh,
                                                flex_detour, zone_min_t, zone_max_t, dispatch_delay, line_id,
                                                wait_time_2)
                                            rows.append(scenario_row(
                                                scenario_name, dv, st_cfg, sim_end_time, n, size_tag,
                                                extra_cols=extra_cols))
    return rows
