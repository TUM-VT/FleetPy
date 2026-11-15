"""Count the fleet vehicles a co-simulation run left frozen at a node.

A coupled vehicle with a driving leg but no route to send to SUMO logs "is at
node and stays there" once per second and never moves again, and every passenger
of its plan is lost. Since the fix in ``ExternallyMovingSimulationVehicle.
assign_vehicle_plan`` a vehicle standing at its next stop no longer gets such a
leg; what is left (e.g. a vehicle placed on a node with no way out) is counted
here and written beside the travel-time stats, so a finished run can be checked.
"""

#: A vehicle stuck for at least this long counts as frozen; shorter spells are
#: counted separately (one-second blips happen while a route is being set).
FROZEN_MIN_S = 300


def frozen_vehicle_stats(vehicles, min_s=FROZEN_MIN_S) -> dict:
    """Stats for ``tt_source_stats.json`` from the vehicles' ``stays_at_node_s`` counters."""
    hits = [(int(v.vid), v.stays_at_node_pos, int(getattr(v, "stays_at_node_s", 0)))
            for v in vehicles if getattr(v, "stays_at_node_s", 0) > 0]
    frozen = [(vid, None if node is None else int(node), s) for vid, node, s in hits if s >= min_s]
    return {"stays_at_node_vehicles": len(hits),
            "frozen_vehicles": len(frozen),
            "frozen_vehicle_seconds": sum(s for _, _, s in frozen),
            "frozen_vehicles_list": [list(x) for x in frozen]}
