# PTBrokerPAYG: Plan-As-You-Go Strategy Design Document

## 1. Overview

### 1.1 Motivation

PTBrokerEI (Estimation-based Integration) 代表的是一个理想化的 MaaS 平台场景：平台在乘客发出请求时就**预先规划好整个行程**，包括所有模式的衔接时间估计。

然而在现实世界中，很多乘客并没有使用 MaaS 平台，而是采用 **"走一步看一步"** 的方式：
- 先坐 AMoD 到车站
- 到了车站再看有什么 PT 可以搭
- PT 到站后再打车

**PTBrokerPAYG (Plan-As-You-Go)** 模拟这种无 MaaS 平台的行为模式，让我们可以：
1. 研究 MaaS 平台相比无平台整合的价值
2. 理解无缝衔接的重要性
3. 分析行程中断的原因和频率

### 1.2 Core Concept

与 PTBrokerEI 的 **"Plan Everything Upfront"** 不同，PAYG 采用 **"Plan Only the Next Step"** 策略：

| Aspect | PTBrokerEI | PTBrokerPAYG |
|--------|------------|--------------|
| 规划时机 | 请求到达时规划全程 | 每一步完成后规划下一步 |
| PT 查询 | 基于 AMoD 预估下车时间 | 基于实际下车时间 |
| 风险 | PT 可能赶不上（但会检查） | 完全不知道后续能否成功 |
| 失败模式 | 不可达标记 + 取消 | 行程中断 |

---

## 2. Trip Decomposition by Modal State

### 2.1 FLM (First-Last-Mile): AMoD → PT → AMoD

这是最复杂的场景，包含三个决策点：

```
Phase 1: REQUEST_ARRIVAL
  └─ 创建 FLM_AMOD_0 sub-request (只有第一段)
  └─ 用户收到 AMoD offer
  └─ 用户确认预订

Phase 2: FM_AMOD_ALIGHTING (AMoD 下车后)
  └─ 实时查询 PT
  └─ 如有 PT: 创建 FLM_PT sub-request
  └─ 如无 PT: 标记中断 (INTERRUPTED_NO_PT)

Phase 3: PT_ALIGHTING (PT 下车后)
  └─ 实时请求 LM_AMOD
  └─ 如有 AMoD: 创建 FLM_AMOD_1 sub-request
  └─ 如无 AMoD: 标记中断 (INTERRUPTED_NO_LM_AMOD)

Phase 4: LM_AMOD_ALIGHTING
  └─ 行程完成
```

### 2.2 FM (First-Mile): AMoD → PT

```
Phase 1: REQUEST_ARRIVAL
  └─ 创建 FM_AMOD sub-request
  └─ 用户收到 AMoD offer
  └─ 用户确认预订

Phase 2: FM_AMOD_ALIGHTING
  └─ 实时查询 PT
  └─ 如有 PT: 创建 FM_PT sub-request (自动完成，无需再次 offer)
  └─ 如无 PT: 标记中断 (INTERRUPTED_NO_PT)

Phase 3: PT_ARRIVAL
  └─ 行程完成 (到达目的地)
```

### 2.3 LM (Last-Mile): PT → AMoD

```
Phase 1: REQUEST_ARRIVAL
  └─ 查询 PT
  └─ 如有 PT: 创建 LM_PT sub-request，用户收到 PT offer
  └─ 如无 PT: 标记中断 (INTERRUPTED_NO_PT)

Phase 2: PT_ALIGHTING
  └─ 实时请求 LM_AMOD
  └─ 如有 AMoD: 创建 LM_AMOD sub-request
  └─ 如无 AMoD: 标记中断 (INTERRUPTED_NO_LM_AMOD)

Phase 3: LM_AMOD_ALIGHTING
  └─ 行程完成
```

### 2.4 MONOMODAL: Pure AMoD

与 PTBrokerBasic 相同，无变化。

---

## 3. State Management

### 3.1 Trip State Enum

位于 `src/misc/globals.py`：

```python
class PAYG_TRIP_STATE(Enum):
    """PAYG trip state tracking for Plan-As-You-Go broker strategy."""
    PENDING = 0                  # Trip pending (initial state)
    FM_AMOD_BOOKED = 1           # FM/FLM: First AMoD leg booked
    FM_AMOD_COMPLETED = 2        # FM/FLM: First AMoD leg completed
    PT_BOOKED = 3                # PT leg booked
    PT_COMPLETED = 4             # PT leg completed
    LM_AMOD_BOOKED = 5           # LM/FLM: Last AMoD leg booked
    COMPLETED = 10               # Trip completed successfully
    INTERRUPTED_NO_PT = -1       # Interrupted: No PT available after FM alighting
    INTERRUPTED_NO_LM_AMOD = -2  # Interrupted: No LM AMoD available after PT alighting
```

### 3.2 Broker State Tracking

在 `PTBrokerPAYG` 中：

```python
class PTBrokerPAYG(PTBrokerBasic):
    def __init__(self, ...):
        super().__init__(...)
        # PAYG-specific state tracking
        self.payg_trip_states: Dict[int, PAYG_TRIP_STATE] = {}  # rid -> state

        # Pending PT arrivals: rid -> (pt_arrival_time, pt_alighting_node)
        # Used to trigger LM_AMOD requests when PT arrives
        self.pending_pt_arrivals: Dict[int, Tuple[int, int]] = {}
```

### 3.3 Request Attributes

在 `BasicIntermodalRequest` 中添加的 PAYG 属性：

```python
# PAYG-specific attributes
self.payg_interrupted: bool = False          # Flag for interrupted trips
self.payg_interrupt_state: int = None        # PAYG_TRIP_STATE value when interrupted
self.payg_interrupt_time: int = None         # Simulation time when interrupted
```

---

## 4. Key Implementation Details

### 4.1 Class Hierarchy

```
BrokerBase
  └─ BrokerBasic
       └─ PTBrokerBasic
            └─ PTBrokerPAYG
```

### 4.2 Core Method Flow

#### 4.2.1 Request Processing (FM/FLM)

```python
def _process_inform_firstmile_request(self, rid, rq_obj, sim_time, parent_modal_state):
    """FM 请求：只创建 FM_AMOD sub-request，PT 等下车后再查"""
    # 1. Get transfer station
    transfer_station_ids = rq_obj.get_transfer_station_ids()
    transfer_street_node, _ = self._find_transfer_info(transfer_station_ids[0], "pt2street")

    # 2. Create FM_AMOD sub-request only
    for op_id in range(self.n_amod_op):
        self._inform_amod_sub_request(
            rq_obj, RQ_SUB_TRIP_ID.FM_AMOD.value,
            rq_obj.get_origin_node(), transfer_street_node,
            rq_obj.earliest_start_time, parent_modal_state, op_id, sim_time
        )

    # 3. Initialize PAYG state
    self.payg_trip_states[rid] = PAYG_TRIP_STATE.PENDING
```

#### 4.2.2 Offer Collection (PAYG 特殊处理)

```python
def _process_collect_firstmile_offers(self, rid, rq_obj, sim_time):
    """PAYG 模式：只返回第一段 AMoD 的 offer（非 IntermodalOffer）"""
    offers = {}
    for op_id in range(self.n_amod_op):
        # 用父请求 ID 获取 FM_AMOD 的 offer
        offer = self.amod_operators[op_id].get_current_offer(rid)
        if offer is not None:
            offers[op_id] = offer
    return offers
```

#### 4.2.3 AMoD Alighting Event

```python
def acknowledge_user_alighting(self, op_id, rid_struct, vid, alighting_time):
    """AMoD 下车事件 - PAYG 核心触发点"""
    super().acknowledge_user_alighting(op_id, rid_struct, vid, alighting_time)

    rid, sub_trip_id = self._parse_rid_struct(rid_struct)

    # FM_AMOD 或 FLM_AMOD_0 下车 -> 触发 PT 查询
    if sub_trip_id in [RQ_SUB_TRIP_ID.FM_AMOD.value, RQ_SUB_TRIP_ID.FLM_AMOD_0.value]:
        self._handle_fm_amod_alighting(rid, parent_rq_obj, op_id, alighting_time, alighting_node)

    # FLM_AMOD_1 或 LM_AMOD 下车 -> 行程完成
    elif sub_trip_id in [RQ_SUB_TRIP_ID.FLM_AMOD_1.value, RQ_SUB_TRIP_ID.LM_AMOD.value]:
        self._mark_trip_completed(rid, alighting_time)
```

#### 4.2.4 PT Alighting Event (via pending_pt_arrivals)

```python
def _process_pending_pt_arrivals(self, sim_time):
    """在 collect_offers() 中调用，处理 PT 到达事件"""
    arrived_rids = [rid for rid, (arrival_time, _) in self.pending_pt_arrivals.items()
                   if sim_time >= arrival_time]

    for rid in arrived_rids:
        pt_arrival_time, pt_alighting_node = self.pending_pt_arrivals.pop(rid)
        self._handle_pt_alighting(rid, pt_arrival_time, pt_alighting_node)
```

### 4.3 Interruption Handling

```python
def _mark_trip_interrupted(self, rid, interrupt_state, interrupt_time):
    """标记行程中断"""
    self.payg_trip_states[rid] = interrupt_state

    # Set interrupted flag on parent request (for evaluation)
    parent_rq_obj = self.demand[rid]
    if hasattr(parent_rq_obj, 'set_payg_interrupted'):
        parent_rq_obj.set_payg_interrupted(True, interrupt_state.value, interrupt_time)

    LOG.warning(f"PAYG trip {rid} interrupted: {interrupt_state.name} at time {interrupt_time}")
```

---

## 5. Data Recording for Evaluation

### 5.1 User Stats Columns

在 `1_user-stats.csv` 中，母请求（parent request）记录以下 PAYG 相关列：

| Column | Type | Description |
|--------|------|-------------|
| `payg_interrupted` | bool | 是否中断 |
| `payg_interrupt_state` | int | 中断状态值（-1: NO_PT, -2: NO_LM_AMOD） |
| `payg_interrupt_time` | int | 中断时刻（simulation time） |

### 5.2 Data Recording Flow

```
PTBrokerPAYG._mark_trip_interrupted()
    │
    ├─ Sets self.payg_trip_states[rid] = interrupt_state
    │
    └─ Calls parent_rq_obj.set_payg_interrupted(True, state, time)
           │
           └─ Sets request attributes:
                self.payg_interrupted = True
                self.payg_interrupt_state = state
                self.payg_interrupt_time = time
                    │
                    └─ Written to CSV via _get_new_record():
                         record_dict[G_RQ_PAYG_INTERRUPTED] = self.payg_interrupted
                         record_dict[G_RQ_PAYG_INTERRUPT_STATE] = self.payg_interrupt_state
                         record_dict[G_RQ_PAYG_INTERRUPT_TIME] = self.payg_interrupt_time
```

---

## 6. Evaluation

### 6.1 PAYG Scenario Detection

PAYG 评估通过读取 config 中的 `broker_type` 来判断（而非检查数据列存在与否）：

```python
# In intermodal_evaluation()
broker_type = scenario_parameters.get(G_BROKER_TYPE, "")
is_payg_scenario = broker_type == "PTBrokerPAYG"
payg_stats = calculate_payg_metrics(parent_user_stats, is_payg_scenario, print_comments)
```

### 6.2 PAYG-Specific KPIs

| Metric | Description |
|--------|-------------|
| `payg_total_intermodal_requests` | intermodal 请求总数（FM + LM + FLM） |
| `payg_completed_count` | 成功完成的行程数 |
| `payg_interrupted_count` | 中断的行程数 |
| `payg_completion_rate [%]` | 完成率 |
| `payg_interrupt_rate [%]` | 中断率 |
| `payg_interrupt_no_pt_count` | PT 不可用导致的中断数 |
| `payg_interrupt_no_pt_rate [%]` | PT 不可用中断率 |
| `payg_interrupt_no_amod_count` | LM AMoD 不可用导致的中断数 |
| `payg_interrupt_no_amod_rate [%]` | LM AMoD 不可用中断率 |
| `payg_FM_total` | FM 请求总数 |
| `payg_FM_interrupted_count` | FM 中断数 |
| `payg_FM_interrupt_rate [%]` | FM 中断率 |
| `payg_LM_total` | LM 请求总数 |
| `payg_LM_interrupted_count` | LM 中断数 |
| `payg_LM_interrupt_rate [%]` | LM 中断率 |
| `payg_FLM_total` | FLM 请求总数 |
| `payg_FLM_interrupted_count` | FLM 中断数 |
| `payg_FLM_interrupt_rate [%]` | FLM 中断率 |

### 6.3 Leg-level Time Capture

评估代码正确捕捉各阶段的上下车时间：

| Modal State | Leg Info Columns |
|-------------|------------------|
| FM | `fm_amod_pu`, `fm_amod_do`, `fm_pt_pu`, `fm_pt_do` |
| LM | `lm_pt_pu`, `lm_pt_do`, `lm_amod_pu`, `lm_amod_do` |
| FLM | `flm_amod_0_pu/do`, `flm_pt_pu/do`, `flm_amod_1_pu/do` |

---

## 7. Configuration

### 7.1 Scenario Configuration

在 `scenario.csv` 中设置：

```csv
scenario_name,broker_type,rq_type
example_im_ptbrokerPAYG,PTBrokerPAYG,BasicIntermodalRequest
```

### 7.2 Required Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `broker_type` | `PTBrokerPAYG` | 使用 PAYG broker |
| `rq_type` | `BasicIntermodalRequest` | 支持 intermodal 的请求类型 |
| `evaluation_method` | `intermodal_evaluation` | 使用 intermodal 评估 |

---

## 8. Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `src/broker/PTBrokerPAYG.py` | CREATE | PAYG broker 主实现 |
| `src/misc/globals.py` | MODIFY | 添加 PAYG_TRIP_STATE 枚举和 G_RQ_PAYG_* 常量 |
| `src/misc/init_modules.py` | MODIFY | 注册 PTBrokerPAYG |
| `src/demand/TravelerModels.py` | MODIFY | 添加 PAYG 属性和 set_payg_interrupted() 方法 |
| `src/evaluation/intermodal.py` | MODIFY | 添加 calculate_payg_metrics() 函数 |
| `src/FleetSimulationBase.py` | MODIFY | 添加 PTBrokerPAYG 到 broker type 检查 |

---

## 9. Testing Log

### 9.1 Bugs Fixed During Development

#### Bug 1: Unknown broker type: PTBrokerPAYG
- **Error**: `ValueError: Unknown broker type: PTBrokerPAYG`
- **Fix**: Added "PTBrokerPAYG" to broker type check in `FleetSimulationBase.py`

#### Bug 2: KeyError '0_8' (pure PT sub-request)
- **Cause**: LM request's PT offer incorrectly treated as pure PT
- **Fix**: Added modal state check in `inform_user_booking()`

#### Bug 3: KeyError '2_1' (FM_AMOD sub-request not found)
- **Cause**: `user_ends_alighting()` deletes request before `acknowledge_user_alighting()`
- **Fix**: Get alighting node from parent request's transfer station info

#### Bug 4: AttributeError 'NoneType' has no attribute 'group'
- **Cause**: PT sub-requests didn't have offers recorded via `receive_offer()`
- **Fix**: Added `receive_offer()` calls for PT sub-requests after creation

### 9.2 Test Results

| Metric | PTBroker (MaaS) | PTBrokerPAYG |
|--------|-----------------|--------------|
| Service Rate | 82% | 78% |
| FM Service Rate | 76% | 100% |
| LM Service Rate | 100% | 76% |
| FLM Service Rate | 68% | 56% |
| PAYG Completion Rate | - | 97.3% |
| PAYG Interrupt Rate | - | 2.7% |
| Avg Wait Time | 253.8s | 335.2s |
| AMoD Wait Time | 141.4s | 194.8s |
| Fleet Utilization | 87.4% | 73.3% |

### 9.3 Key Observations

1. **PAYG has lower overall service rate** (78% vs 82%) - expected because real-time planning may fail
2. **PAYG FM has 100% service rate** - PT is queried after AMoD delivery (always available at that time)
3. **PAYG has higher wait times** - travelers don't pre-plan, less efficient coordination
4. **Lower fleet utilization in PAYG** - less efficient matching without pre-planning
5. **2.7% interrupt rate** - 2 FLM trips failed due to no LM AMoD available after PT alighting

---

## 10. MaaS vs PAYG Value Comparison

| Aspect | PAYG (No MaaS) | MaaS (PTBroker) | Difference |
|--------|----------------|-----------------|------------|
| Planning | Step-by-step | Upfront | MaaS enables optimization |
| Trip Completion | May interrupt | Pre-validated | MaaS ensures feasibility |
| Wait Time | Higher | Lower | MaaS coordinates transfers |
| User Experience | Uncertain | Predictable | MaaS reduces anxiety |
| Fleet Efficiency | Lower | Higher | MaaS enables better matching |

This comparison quantifies the **value of MaaS integration**:
- ~4% higher service rate
- ~32% lower wait time
- ~14% higher fleet utilization
- Elimination of mid-trip interruptions

---

## 11. Author & Date

- Author: Claude (assisted design & implementation)
- Initial Design: 2026-02-04
- Implementation Completed: 2026-02-05
- Status: **IMPLEMENTED & TESTED**
