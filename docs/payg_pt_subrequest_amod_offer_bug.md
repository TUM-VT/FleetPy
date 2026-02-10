# PTBrokerPAYG Bug: PT Sub-request Records AMOD Offer

## Problem Description

In `studies/example_study/results/example_im_ptbrokerPAYG/1_user-stats.csv`, PT leg sub-requests (sub_trip_id=2 FM_PT, sub_trip_id=6 FLM_PT) incorrectly record AMOD offers in addition to PT offers.

### Observed Data

**FM_PT sub-request (request_id=2, sub_trip_id=2.0)** — Line 114:
```
offers = 0:t_wait:54.01;t_drive:248.71;fare:177|-2:t_wait:239;t_drive:270;fare:0;...
```
- `0:...` is the FM_AMOD offer (should NOT be here)
- `-2:...` is the PT offer (correct)

**FLM_PT sub-request (request_id=4, sub_trip_id=6.0)** — Line 117:
```
offers = 0:t_wait:79.67;t_drive:301.54;fare:152|-2:t_wait:100;t_drive:270;fare:0;...
```
- `0:...` is the FLM_AMOD_0 offer (should NOT be here)
- `-2:...` is the PT offer (correct)

**LM_PT sub-request (request_id=0, sub_trip_id=3.0)** — Line 111:
```
offers = -2:t_wait:59;t_drive:270;fare:0;...
```
- Only PT offer (correct, no AMOD contamination)

### Pattern

| Sub-trip Type | sub_trip_id | Has AMOD Offer? | Expected? |
|---------------|-------------|-----------------|-----------|
| FM_PT         | 2           | Yes             | No        |
| FLM_PT        | 6           | Yes             | No        |
| LM_PT         | 3           | No              | Correct   |

LM_PT is not affected because in PAYG mode, the LM flow queries PT at request time (before any AMOD offers exist on the parent).

---

## Root Cause Analysis

### The `deepcopy` Inheritance Chain

The bug originates from `create_SubTripRequest` in `src/demand/TravelerModels.py:248`:

```python
def create_SubTripRequest(self, subtrip_id, leg_o_node=None, ...):
    sub_rq_obj = deepcopy(self)  # <-- Copies ALL fields, including self.offer dict
    sub_rq_obj.sub_rid_struct = f"{old_rid}_{subtrip_id}"
    # ... updates origin/destination/timing but NOT the offer dictionary
    return sub_rq_obj
```

The `deepcopy(self)` copies the parent request's entire `self.offer` dictionary. If the parent has already accumulated offers from previous steps, those offers are inherited by the new sub-request.

### Full Flow (FM example)

**Step 1: Request arrives** — `ImmediateDecisionsSimulation.py:89-94`
```python
self.broker.inform_request(rid, rq_obj, sim_time)     # Creates FM_AMOD sub-request
amod_offers = self.broker.collect_offers(rid, sim_time) # Returns {0: fm_amod_offer}
for op_id, amod_offer in amod_offers.items():
    rq_obj.receive_offer(op_id, amod_offer, sim_time)  # Records AMOD offer on PARENT request
```
After this step: `parent_request.offer = {0: fm_amod_offer}`

**Step 2: User books FM_AMOD** — `PTBrokerPAYG.py:246-256`
FM_AMOD sub-request is confirmed. Parent still has `offer = {0: fm_amod_offer}`.

**Step 3: FM_AMOD vehicle drops off user** — `PTBrokerPAYG.py:322-325`
```python
# acknowledge_user_alighting → _handle_fm_amod_alighting
```

**Step 4: PT sub-request created** — `PTBrokerPAYG.py:349-352` → `PTBrokerBasic.py:387`
```python
pt_arrival = self._inform_pt_sub_request(
    rq_obj,           # Parent request (has offer = {0: fm_amod_offer})
    RQ_SUB_TRIP_ID.FM_PT.value, ...
)

# Inside _inform_pt_sub_request (PTBrokerBasic.py:387):
pt_sub_rq_obj = self.demand.create_sub_requests(rq_obj, sub_trip_id, ...)
# → calls create_SubTripRequest → deepcopy(parent) → inherits parent.offer!
```
After deepcopy: `pt_sub_rq_obj.offer = {0: fm_amod_offer}` (inherited!)

**Step 5: PT offer recorded** — `PTBrokerPAYG.py:364-366`
```python
pt_sub_rq_obj.receive_offer(self.pt_operator_id, pt_offer, None)
```
Final state: `pt_sub_rq_obj.offer = {0: fm_amod_offer, -2: pt_offer}` (both!)

**Step 6: CSV output** — `TravelerModels.py:856-859`
```python
for op_id, operator_offer in self.offer.items():
    all_offer_info.append(f"{op_id}:" + operator_offer.to_output_str())
record_dict[G_RQ_OFFERS] = "|".join(all_offer_info)
```
Result: `"0:t_wait:54;t_drive:248;fare:177|-2:t_wait:239;t_drive:270;fare:0;..."`

### Why LM_PT is Not Affected

For LM requests, PT is queried at request time in `_process_inform_lastmile_request` (line 102), which happens during `inform_request` — **before** `collect_offers` is called by the simulation. At that point, the parent request's `offer` dict is still empty `{}`, so the deepcopy produces a clean PT sub-request.

### Why This Bug is PAYG-specific

In PTBrokerBasic/PTBrokerEI, all sub-requests (including PT legs) are created during `inform_request` before `collect_offers`. The parent's `offer` dict is always empty at sub-request creation time. In PAYG, FM_PT and FLM_PT are created **after** AMOD alighting, by which point the parent has already accumulated AMOD offers from the initial `collect_offers` cycle.

---

## Affected Code Locations

| File | Line | Description |
|------|------|-------------|
| `src/demand/TravelerModels.py` | 248 | `deepcopy(self)` copies parent's `offer` dict |
| `src/demand/TravelerModels.py` | 195 | `receive_offer` stores `self.offer[operator_id]` |
| `src/demand/TravelerModels.py` | 856-859 | CSV output iterates all offers |
| `src/ImmediateDecisionsSimulation.py` | 93 | `rq_obj.receive_offer(...)` records offer on parent |
| `src/broker/PTBrokerPAYG.py` | 349-352 | `_handle_fm_amod_alighting` creates PT sub via parent |
| `src/broker/PTBrokerPAYG.py` | 388-391 | `_handle_flm_amod_0_alighting` creates PT sub via parent |
| `src/broker/PTBrokerBasic.py` | 387 | `_inform_pt_sub_request` calls `create_sub_requests` |

---

## Potential Fixes

### Option A: Clear offers after deepcopy in PTBrokerPAYG (recommended, minimal change)

In `_handle_fm_amod_alighting` and `_handle_flm_amod_0_alighting`, clear the inherited offers from the PT sub-request after creation:

```python
# After _inform_pt_sub_request creates the PT sub-request:
pt_rid_struct = f"{rid}_{RQ_SUB_TRIP_ID.FM_PT.value}"
pt_sub_rq_obj = self.demand[pt_rid_struct]
pt_sub_rq_obj.offer = {}  # Clear inherited AMOD offers

# Then record only the PT offer:
pt_offer = self.pt_operator.get_current_offer(pt_rid_struct, amod_op_id)
if pt_offer is not None:
    pt_sub_rq_obj.receive_offer(self.pt_operator_id, pt_offer, None)
```

### Option B: Reset offers in `create_SubTripRequest` (broader fix)

Add `sub_rq_obj.offer = {}` in `create_SubTripRequest` to always start with a clean offer dict:

```python
def create_SubTripRequest(self, subtrip_id, ...):
    sub_rq_obj = deepcopy(self)
    sub_rq_obj.offer = {}  # Don't inherit parent's offers
    ...
```

This is a more general fix but could affect other broker strategies if they rely on inherited offers (unlikely but needs verification).

### Option C: Pass a clean copy to `_inform_pt_sub_request`

Create a temporary copy of the parent request with cleared offers before passing to `_inform_pt_sub_request`. This is more defensive but adds complexity.

---

## Fix Applied (Option A)

In `src/broker/PTBrokerPAYG.py`, added `pt_sub_rq_obj.offer = {}` after obtaining the PT sub-request
in both `_handle_fm_amod_alighting` and `_handle_flm_amod_0_alighting`, clearing inherited AMOD offers
before recording the PT offer.

**Verification**: Re-ran `example_im_ptbrokerPAYG` scenario. All 70 PT sub-requests now contain only
PT offers (operator -2). Zero AMOD offer contamination.

---

## Impact Assessment

- **Data correctness**: The AMOD offer in PT sub-requests is misleading but does not affect the actual simulation behavior (PT is still booked correctly).
- **Evaluation metrics**: If evaluation code reads the `offers` column of PT sub-requests, it could misinterpret the AMOD offer as a real offer for the PT leg, potentially distorting metrics.
- **Parent request stats**: The parent request's `offers` column (`1_user-stats_parent.csv`) is also affected — it shows only the initial AMOD offer, not the complete intermodal offer chain. This is by design for PAYG (parent only sees the first leg at booking time).
