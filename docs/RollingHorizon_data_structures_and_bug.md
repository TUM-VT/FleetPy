# RollingHorizon Data Structures and Bug Analysis

## Data Structure Overview

| Data Structure | Purpose | Analogy |
|----------------|---------|---------|
| `sorted_rids_with_epa` | List of requests sorted by pickup time, waiting to be revealed | **Schedule**: When to process which request |
| `rid_to_assigned_vid` | Maps request ID to assigned vehicle ID | **Assignment Table**: Which vehicle serves which passenger |
| `active_reservation_requests` | Detailed information of active reservation requests | **Archive**: Complete request data |

**Critical Invariant**: These three structures must remain synchronized. Any deletion must update all three.

## Example: FLM Request rid=37

### Scenario Setup

At `sim_time=1000`, a user creates a First-Last-Mile (FLM) intermodal request:

```
User Journey: Home → [AMoD] → Metro Station A → [Metro] → Metro Station B → [AMoD] → Office
```

PTBrokerEI creates 3 sub-requests:

| Sub-request | rid_struct | Type | earliest_pickup_time |
|-------------|------------|------|---------------------|
| FM AMoD | `37_5` | FLM_AMOD_0 | 1000 (now) |
| PT | `37_6` | FLM_PT | 1300 (estimated) |
| LM AMoD | `37_7` | FLM_AMOD_1 | 1800 (estimated) |

### Step 1: Sub-requests Created and Confirmed

```
sim_time = 1000

# FM AMoD (37_5) - Immediate request, not in reservation module

# LM AMoD (37_7) - Future request, enters reservation module
sorted_rids_with_epa = [('37_7', 1800)]  # Schedule: reveal at 1800
rid_to_assigned_vid = {'37_7': 5}        # Assignment: Vehicle 5 responsible
active_reservation_requests = {'37_7': <PlanRequest>}  # Archive: Full data
```

### Step 2: FM AMoD Service Completes, But Delayed

```
sim_time = 1500  (200 seconds later than estimated 1300)

# Passenger alights, check if they can catch the metro
alighting_time = 1500
PT_latest_arrival = 1400  # Latest time to arrive at metro station

1500 > 1400 → PT is uncatchable!
```

### Step 3: PTBrokerEI Cancels Subsequent Sub-requests

```python
# PTBrokerEI._handle_uncatchable_pt() calls:
op.user_cancels_request('37_7', 1500)
```

### Step 4: The Bug in RollingHorizon.user_cancels_request

```python
def user_cancels_request(self, rid, simulation_time):
    if self.rid_to_assigned_vid.get(rid) is not None:  # '37_7' exists, condition is True
        vid = self.rid_to_assigned_vid[rid]  # vid = 5
        # ... remove 37_7 from vehicle 5's plan ...
        del self.rid_to_assigned_vid[rid]           # ✓ Deleted
        del self.active_reservation_requests[rid]   # ✓ Deleted
        # sorted_rids_with_epa ???                  # ✗ FORGOT TO DELETE!
```

State after execution:
```
sorted_rids_with_epa = [('37_7', 1800)]  # Still there! Schedule not updated
rid_to_assigned_vid = {}                  # Deleted
active_reservation_requests = {}          # Deleted
```

### Step 5: Time Progresses, Reveal Triggered

```
sim_time = 1750  (approaching 1800)
rolling_horizon = 100

# FleetControlBase.time_trigger() calls:
# RollingHorizon.reveal_requests_for_online_optimization(1750)

# Check: 1800 <= 1750 + 100 = 1850 → Yes, time to reveal
to_return = ['37_7']  # Retrieved from sorted_rids_with_epa

for rid in to_return:
    del self.active_reservation_requests[rid]  # KeyError: '37_7'
    # Because '37_7' was already deleted in Step 4!
```

## Timeline Diagram

```
Timeline:
1000        1300        1500        1750        1800
  |           |           |           |           |
  v           v           v           v           v
Request    Estimated   Actual      Reveal      Original
Created    PT time     Alighting   Triggered   Pickup
  |                       |           |
  |                       |           +---> KeyError!
  |                       |                 sorted_rids_with_epa has '37_7'
  |                       |                 active_reservation_requests doesn't
  |                       |
  |                       +---> Cancel '37_7'
  |                             Deleted from active_reservation_requests
  |                             FORGOT to delete from sorted_rids_with_epa
  |
  +---> '37_7' enters reservation module
        All three data structures have it
```

## Why Normal Scenarios Don't Trigger This Bug

In normal reservation cancellation flow:
1. User cancels manually → Usually happens **before confirm**
2. At that point, `sorted_rids_with_epa` doesn't have this rid yet (only added after confirm)
3. So not cleaning it is harmless

What's special about PTBrokerEI:
1. LM request is **auto-confirmed** at creation (system-created sub-request)
2. Cancellation happens **after confirm**
3. `sorted_rids_with_epa` already has this rid, but it's not cleaned up

## The Fix

The `user_cancels_request` method should always clean up `sorted_rids_with_epa`, regardless of whether the request was assigned to a vehicle:

```python
def user_cancels_request(self, rid, simulation_time):
    # If assigned to a vehicle, remove from vehicle plan
    if self.rid_to_assigned_vid.get(rid) is not None:
        vid = self.rid_to_assigned_vid[rid]
        assigned_plan = self.fleetctrl.veh_plans[vid]
        veh_obj = self.fleetctrl.sim_vehicles[vid]
        new_plan = simple_remove(veh_obj, assigned_plan, rid, simulation_time,
            self.routing_engine, self.fleetctrl.vr_ctrl_f, self.fleetctrl.rq_dict,
            self.fleetctrl.const_bt, self.fleetctrl.add_bt)
        self.fleetctrl.assign_vehicle_plan(veh_obj, new_plan, simulation_time)
        del self.rid_to_assigned_vid[rid]

    # Unconditionally clean up (FIX)
    if rid in self.active_reservation_requests:
        del self.active_reservation_requests[rid]

    # Clean up sorted_rids_with_epa (FIX - reference NoGuarantee implementation)
    self.sorted_rids_with_epa = [(r, epa) for r, epa in self.sorted_rids_with_epa if r != rid]
```

## Comparison: RollingHorizon vs RollingHorizonNoGuarantee

| Aspect | RollingHorizon (Buggy) | RollingHorizonNoGuarantee (Correct) |
|--------|------------------------|-------------------------------------|
| `user_cancels_request` condition | Only if `rid_to_assigned_vid` exists | Unconditional |
| Cleans `active_reservation_requests` | Conditional | Always |
| Cleans `sorted_rids_with_epa` | **Never** | Always |

The `RollingHorizonNoGuarantee` implementation is correct and should be used as reference for fixing `RollingHorizon`.
