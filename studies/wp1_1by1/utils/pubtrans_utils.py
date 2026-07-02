
# def _middle_row_stations(rows, cols, cell_size, station_spacing_m):
#     """Return list of (station_id, node_index, pos_x, pos_y) along the middle row."""
#     mid_row = rows // 2
#     pos_y = mid_row * cell_size
#     grid_length_m = (cols - 1) * cell_size

#     stations = []
#     x = 0.0
#     sid = 0
#     while x <= grid_length_m + 1e-6:
#         col = round(x / cell_size)
#         node_index = mid_row * cols + col
#         stations.append((sid, node_index, x, pos_y))
#         sid += 1
#         x += station_spacing_m

#     return stations


# def _midline_schedule(stations, station_spacing_m, speed_kmh, headway_s,
#                       sim_end_time, line_id, vehicle_type, boarding_time_s=0):
#     """Generate rows for schedules.csv for a back-and-forth middle-line service.

#     Each departure time already includes dwell (boarding_time_s) at the previous
#     stop, matching how FleetPy's PTScheduleGen bakes dwell into departure offsets.
#     """
#     speed_ms = speed_kmh * 1000 / 3600
#     tt_between = station_spacing_m / speed_ms                    # pure travel time between stops (s)
#     stop_interval = tt_between + boarding_time_s                 # travel + dwell per inter-stop segment
#     one_way_time = (len(stations) - 1) * stop_interval          # end-to-end scheduled time
#     round_trip_time = 2 * one_way_time

#     if round_trip_time > headway_s:
#         raise ValueError(
#             f"Round-trip time {round_trip_time:.0f}s exceeds headway {headway_s}s — "
#             "increase headway or reduce station spacing."
#         )

#     rows = []
#     trip_id = 0
#     dep_from_left = 0.0

#     while dep_from_left < sim_end_time:
#         # Outbound: left → right
#         for i, (sid, _, _, _) in enumerate(stations):
#             rows.append({
#                 "departure": dep_from_left + i * stop_interval,
#                 "station_id": sid,
#                 "trip_id": trip_id,
#                 "line_vehicle_id": 0,
#                 "LINE": line_id,
#                 "vehicle_type": vehicle_type,
#             })
#         trip_id += 1

#         # Return: right → left
#         dep_from_right = dep_from_left + one_way_time
#         for i, (sid, _, _, _) in enumerate(reversed(stations)):
#             rows.append({
#                 "departure": dep_from_right + i * stop_interval,
#                 "station_id": sid,
#                 "trip_id": trip_id,
#                 "line_vehicle_id": 0,
#                 "LINE": line_id,
#                 "vehicle_type": vehicle_type,
#             })
#         trip_id += 1

#         dep_from_left += headway_s

#     return rows


# def _write_alignment_geojson(stations, pt_out_dir, line_id, pt_name):
#     """Write a GeoJSON LineString alignment for the middle-line route."""
#     coords = [[x, y] for _, _, x, y in stations]
#     geojson = {
#         "type": "FeatureCollection",
#         "features": [
#             {
#                 "type": "Feature",
#                 "geometry": {"type": "LineString", "coordinates": coords},
#                 "properties": {
#                     "name": pt_name,
#                     "description": (
#                         f"Middle-line shuttle. "
#                         f"Stations: {[sid for sid, *_ in stations]}. "
#                         f"Node indices: {[nidx for _, nidx, *_ in stations]}."
#                     ),
#                 },
#             }
#         ],
#     }
#     path = os.path.join(pt_out_dir, f"{line_id}_line_alignment.geojson")
#     with open(path, "w") as f:
#         json.dump(geojson, f, indent=2)


# def generate_pubtrans(ranges):
#     """Generate middle-line public transit data for each network × headway combination.

#     Creates one data/pubtrans/ directory per combination, each containing
#     stations.csv, schedules.csv, and a GeoJSON route alignment.

#     Returns a list of pt_name strings.
#     """
#     nw_ranges = ranges["network"]
#     pt_ranges = ranges.get("pubtrans", {})
#     sim_end_time = ranges.get("simulation", {}).get("end_time", 3600)

#     headways_min = pt_ranges.get("headways_min", [5, 10, 20])
#     station_spacings_m = pt_ranges.get("station_spacings_m", [500])
#     boarding_time_s = pt_ranges.get("boarding_time_s", 0)
#     vehicle_type = pt_ranges.get("vehicle_type", "veh_20")
#     line_id = pt_ranges.get("line_id", 1)
#     speed_kmh = nw_ranges["default_speed"]
#     cell_size = nw_ranges["cell_size"]

#     pt_names = []

#     for length in nw_ranges["lengths"]:
#         for width in nw_ranges["widths"]:
#             rows, cols = get_row_cols(length, width, cell_size)

#             for station_spacing_m in station_spacings_m:
#                 stations = _middle_row_stations(rows, cols, cell_size, station_spacing_m)

#                 for hw_min in headways_min:
#                     headway_s = hw_min * 60
#                     pt_name = f"grid_l{length}_w{width}_cell{cell_size}_mid_sp{station_spacing_m}_hw{hw_min}"
#                     pt_out_dir = os.path.join(PT_DIR, pt_name)
#                     os.makedirs(pt_out_dir, exist_ok=True)

#                     # stations.csv
#                     station_rows = [{"station_id": sid, "network_node_index": nidx}
#                                     for sid, nidx, _, _ in stations]
#                     pd.DataFrame(station_rows).to_csv(
#                         os.path.join(pt_out_dir, "stations.csv"), index=False)

#                     # schedules.csv
#                     schedule_rows = _midline_schedule(
#                         stations, station_spacing_m, speed_kmh,
#                         headway_s, sim_end_time, line_id, vehicle_type,
#                         boarding_time_s=boarding_time_s)
#                     pd.DataFrame(schedule_rows).to_csv(
#                         os.path.join(pt_out_dir, "schedules.csv"), index=False)

#                     # GeoJSON alignment
#                     _write_alignment_geojson(stations, pt_out_dir, line_id, pt_name)

#                     speed_ms = speed_kmh * 1000 / 3600
#                     one_way_s = (len(stations) - 1) * station_spacing_m / speed_ms
#                     n_trips = len([r for r in schedule_rows if r["station_id"] == stations[0][0]])
#                     print(
#                         f"  {pt_name}: {len(stations)} stations @ {station_spacing_m}m spacing, "
#                         f"one-way {one_way_s:.0f}s, headway {hw_min}min, "
#                         f"{n_trips} outbound departures"
#                     )
#                     pt_names.append(pt_name)

#     return pt_names

