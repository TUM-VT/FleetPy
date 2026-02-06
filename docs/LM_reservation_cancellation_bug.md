# LM Reservation Cancellation Bug Analysis

## Issue Summary

**Error**: `KeyError: '37_7'` in `RollingHorizon.reveal_requests_for_online_optimization()` when accessing `active_reservation_requests[rid]`

**Request ID Format**: `37_7` = parent_rid=37, sub_trip_id=7 (FLM_AMOD_1 - Last Mile AMoD leg of a First-Last-Mile trip)

## Stack Trace

```
File "RollingHorizon.py", line 53, in reveal_requests_for_online_optimization
    del self.active_reservation_requests[rid]
KeyError: '37_7'
```

## Root Cause Analysis

### Data Structures in RollingHorizon

The `RollingHorizonReservation` class maintains three data structures that must stay synchronized:

| Data Structure | Type | Purpose |
|----------------|------|---------|
| `active_reservation_requests` | Dict[rid, PlanRequest] | Source of truth for active reservations |
| `sorted_rids_with_epa` | List[(rid, earliest_pickup_time)] | Sorted list for reveal timing |
| `rid_to_assigned_vid` | Dict[rid, vid] | Vehicle assignments |

**Critical Invariant**: All three structures must remain synchronized. Any deletion must update all three.

### Bug Location: `RollingHorizon.user_cancels_request()` (lines 98-112)

```python
def user_cancels_request(self, rid, simulation_time):
    if self.rid_to_assigned_vid.get(rid) is not None:  # <-- BUG: conditional check
        vid = self.rid_to_assigned_vid[rid]
        assigned_plan = self.fleetctrl.veh_plans[vid]
        veh_obj = self.fleetctrl.sim_vehicles[vid]
        new_plan = simple_remove(...)
        self.fleetctrl.assign_vehicle_plan(veh_obj, new_plan, simulation_time)
        del self.rid_to_assigned_vid[rid]
        del self.active_reservation_requests[rid]
        # MISSING: No removal from sorted_rids_with_epa!
```

### Two Critical Issues

1. **Conditional Deletion**: The method only deletes from `active_reservation_requests` if `rid_to_assigned_vid.get(rid) is not None`. If the request was never assigned to a vehicle (common for future reservation requests), nothing gets deleted.

2. **Missing `sorted_rids_with_epa` Cleanup**: Even when deletion occurs, `sorted_rids_with_epa` is never updated, leaving stale entries.

### Contrast with `RollingHorizonNoGuarantee.user_cancels_request()` (lines 104-117)

```python
def user_cancels_request(self, rid, simulation_time):
    del self.active_reservation_requests[rid]  # Always delete
    to_del = None
    for i, entry in enumerate(self.sorted_rids_with_epa):
        if entry[0] == rid:
            to_del = i
            break
    if to_del is not None:
        self.sorted_rids_with_epa.pop(to_del)  # Also remove from sorted list
```

This implementation is correct - it always deletes from both structures unconditionally.

## Trigger Scenario

### Step-by-Step Flow

1. **Request Creation** (PTBrokerEI):
   - User creates FLM (First-Last-Mile) intermodal request (rid=37)
   - PTBrokerEI creates sub-requests: `37_5` (FLM_AMOD_0), `37_6` (FLM_PT), `37_7` (FLM_AMOD_1)
   - LM AMoD sub-request `37_7` added to reservation module (future pickup time)
   - Both `active_reservation_requests['37_7']` and `sorted_rids_with_epa` contain the request

2. **FM Service Completes**:
   - Vehicle completes FM AMoD leg (`37_5`)
   - `PTBrokerEI.acknowledge_user_alighting()` called
   - Checks PT catchability via `_check_fm_pt_catchability()`

3. **PT Uncatchable** (alighting time > PT latest arrival):
   - `_handle_uncatchable_pt()` called
   - Cancels subsequent sub-requests including `37_7`:
   ```python
   op.user_cancels_request(lm_amod_rid_struct, sim_time)  # '37_7'
   ```

4. **Cancellation Bug**:
   - `RollingHorizon.user_cancels_request('37_7', ...)` called
   - If `37_7` was assigned: deletes from `active_reservation_requests` but NOT from `sorted_rids_with_epa`
   - If `37_7` was not assigned: does NOTHING (conditional check fails)

5. **Reveal Failure**:
   - Later, `time_trigger()` → `reveal_requests_for_online_optimization()`
   - Iterates `sorted_rids_with_epa`, finds `37_7`
   - Tries `del self.active_reservation_requests['37_7']`
   - **KeyError**: Already deleted (or never properly tracked)

## Affected Scenarios

| Scenario | Risk Level | Reason |
|----------|------------|--------|
| PTBrokerEI with FLM requests | **High** | PT cancellation triggers LM AMoD cancellation |
| PTBrokerEI with LM requests | Medium | If PT becomes unavailable after booking |
| Any reservation cancellation | Medium | Data structure inconsistency |

## Recommended Fix

### Fix for `RollingHorizon.user_cancels_request()`

```python
def user_cancels_request(self, rid, simulation_time):
    """in case a reservation request which could be assigned earlier cancels the request
    this function removes the request from the assigned vehicle plan and deletes all entries in the database
    :param rid: request id
    :param simulation_time: current simulation time
    """
    # Remove from vehicle plan if assigned
    if self.rid_to_assigned_vid.get(rid) is not None:
        vid = self.rid_to_assigned_vid[rid]
        assigned_plan = self.fleetctrl.veh_plans[vid]
        veh_obj = self.fleetctrl.sim_vehicles[vid]
        new_plan = simple_remove(veh_obj, assigned_plan, rid, simulation_time,
            self.routing_engine, self.fleetctrl.vr_ctrl_f, self.fleetctrl.rq_dict,
            self.fleetctrl.const_bt, self.fleetctrl.add_bt)
        self.fleetctrl.assign_vehicle_plan(veh_obj, new_plan, simulation_time)
        del self.rid_to_assigned_vid[rid]

    # Always delete from active_reservation_requests (unconditional)
    if rid in self.active_reservation_requests:
        del self.active_reservation_requests[rid]

    # Always remove from sorted_rids_with_epa (FIX: was missing!)
    self.sorted_rids_with_epa = [(r, epa) for r, epa in self.sorted_rids_with_epa if r != rid]
```

### Defensive Check in `reveal_requests_for_online_optimization()`

```python
def reveal_requests_for_online_optimization(self, sim_time):
    reveal_index = 0
    while reveal_index < len(self.sorted_rids_with_epa) and \
          self.sorted_rids_with_epa[reveal_index][1] <= sim_time + self.rolling_horizon:
        reveal_index += 1
    to_return = [self.sorted_rids_with_epa[x][0] for x in range(reveal_index)]
    self.sorted_rids_with_epa = self.sorted_rids_with_epa[reveal_index:]

    valid_to_return = []
    for rid in to_return:
        # Defensive check: skip if already deleted (e.g., cancelled)
        if rid not in self.active_reservation_requests:
            LOG.warning(f"Request {rid} in sorted_rids_with_epa but not in active_reservation_requests - skipping (likely cancelled)")
            continue
        valid_to_return.append(rid)
        del self.active_reservation_requests[rid]
        try:
            del self.rid_to_assigned_vid[rid]
        except KeyError:
            pass

    LOG.debug(f"reveal following reservation requests at time {sim_time} : {valid_to_return}")
    return valid_to_return
```

## PTBroker Strategy Comparison

| Strategy | FLM Request Creation | Risk of Reservation Bug |
|----------|---------------------|------------------------|
| PTBrokerBasic | Empty (pass) | Low - no FLM sub-requests created |
| PTBrokerEI | Upfront (all at request time) | **High** - PT cancellation triggers bug |
| PTBrokerPAYG | Real-time (step by step) | Low - sim_time ≈ earliest_pickup_time |

## Testing Recommendations

1. Create a test scenario with FLM requests where PT becomes uncatchable
2. Verify that `sorted_rids_with_epa` and `active_reservation_requests` stay synchronized after cancellation
3. Test reservation cancellation for requests that are:
   - Already assigned to a vehicle
   - Not yet assigned to any vehicle (future reservations)

## Related Files

- `src/fleetctrl/reservation/RollingHorizon.py` - Primary bug location
- `src/fleetctrl/reservation/RollingHorizonNoGuarantee.py` - Correct implementation reference
- `src/broker/PTBrokerEI.py` - Trigger location (`_handle_uncatchable_pt`)
- `src/fleetctrl/FleetControlBase.py` - Calls `reveal_requests_for_online_optimization`
