# Intermodal evaluation

`intermodal_evaluation` writes `standard_eval.csv` with one `MoD_<operator id>`
column per AMoD operator and an `Intermodal` column for parent journeys. This
replaces the older single-column format; downstream scripts using its former
`amod_*`, `op*`, or journey `travel time` rows need to use the new columns and
names. `1_user-stats_parent.csv` retains detailed journey records, while
`1_user-stats_amod_legs.csv` contains the selected, completed AMoD legs used in
the operator user statistics.

The request cohort is selected by the parent request time in the half-open
evaluation interval. All selected children are retained. Completion is observed
through simulation `end_time` (or the explicit `completion_end_time` passed to
`create_parent_user_stats`), independent of the cohort interval. PT booking
prefills times: completion additionally checks the horizon, terminal failure
states, required legs and transfers. It is an inference from the recorded PT
schedule and state, not an independently observed PT vehicle event. Vehicle
records are filtered by vehicle-leg start time, as in `standard_evaluation`;
vehicle and customer denominators therefore need not describe the same events
when an evaluation interval is used.

In each `MoD` column, user counts, traveler counts, waiting/travel time,
detour, revenue and direct distance use completed selected AMoD legs. Two FLM
legs count twice, and a completed leg remains included if the later journey
fails. `waiting time` is pickup minus the leg's `rq_time` and `waiting time
from ept` is pickup minus its earliest pickup time, matching the standard
formulas. An intermodal child can inherit the original parent `rq_time`; in
particular, an LM leg's waiting time can include the previous PT journey. The
`LM_Wait_Time_*` journey fields instead measure the interval between PT
arrival plus target walking and LM pickup. Detour uses each leg's direct road
travel time and the serving operator's boarding time. Missing values are not
replaced with a guessed travel speed.

Modal split counts unique parent requests or passengers with a completed leg
served by the operator, divided by the parent cohort. A parent may be counted
for more than one operator; operator shares do not have to sum to one.
Reservation/online counts are leg counts, whereas their served percentages use
unique parents classified by parent request time and earliest pickup time.
Undefined ratios are NaN. `% created offers` is NaN because the broker does not
retain rejected parent offers; `candidate offers [%]` counts parents with a
recorded candidate involving the operator, including partial PAYG candidates.
It is not equivalent to the standard offer rate.

The `Intermodal` column reports journey completion separately from DRT leg
completion. Journey waiting time is the sum of the recorded component waits;
`journey first pickup to last dropoff [s]` is the span between those events,
including transfers but excluding walking outside those events. It is not a
door-to-door time and must not be added to the waiting-time sum. PT and LM
transfer-wait metrics cover completed journeys with the corresponding legs.
`Service_Rate [%]` is completed parent journeys divided by all parent requests;
`DRT_leg_service_rate [%]` is completed selected AMoD legs divided by required
selected AMoD legs. Neither includes an assumed walking fallback.

The standard evaluator groups requests by recorded operator rather than checking
completed pickup and dropoff; direct comparisons of its user metrics require
checking the underlying pure-AMoD records for incomplete service. Both
calculations use the corrected km/h conversion for average trip velocity.
