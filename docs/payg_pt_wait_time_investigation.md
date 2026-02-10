# PTBrokerPAYG PT 等待时间调查

## 1. 问题描述

在 PTBrokerPAYG 策略下，train headway 从 10 min → 20 min → 30 min 三种情景中，用户的 PT 等待时间基本不变，不符合预期（理论上平均等待时间 ≈ headway / 2）。

## 2. 调查计划

### 2.1 代码层面

- [x] 追踪 PAYG 的 PT 查询完整数据流：`alighting_time` → `_inform_pt_sub_request` → RAPTOR → PTOffer → user_stats
- [x] 验证 RAPTOR 的 `source_station_departure_secs` 和 `source_waiting_time` 的语义
- [x] 分析 `calculate_pt_wait_time()` 评估公式的正确性
- [x] 检查时间单位一致性（simulation time vs GTFS time）

### 2.2 数据/配置层面（需用户验证）

- [ ] 确认三个场景是否使用了不同的 GTFS 数据（不同 headway）
- [ ] 检查 FM/FLM 请求的样本量
- [ ] 检查原始 `1_user-stats.csv` 中 PT 子请求的 `t_wait` 实际值

---

## 3. 代码分析结果

### 3.1 PAYG PT 查询时间链

对于 FM/FLM 请求，PAYG 在 AMoD 下车后实时查询 PT：

```
AMoD 乘客下车 (alighting_time)
    ↓
_handle_fm_amod_alighting() 调用 _inform_pt_sub_request()
    ↓ leg_start_time = alighting_time
create_sub_requests() → sub_rq.earliest_start_time = alighting_time
    ↓
_query_street_node_pt_travel_costs_1to1()
    ↓ est = alighting_time
    ↓ source_station_departure_seconds = est + walking_time
    ↓ source_station_departure_datetime = sim_start_datetime + timedelta(seconds=est + walking_time)
    ↓
RAPTOR C++ 查询 (departure_datetime = 到达车站时刻)
    ↓
返回 pt_journey_plan_dict:
    - source_station_departure_secs = query_time (= est + walking_time, 午夜起算秒数)
    - source_waiting_time = first_departure_from_stop - query_time - source_transfer_time
    - target_station_arrival_secs = 最后一步到达时间 + target_transfer_time
    ↓
PTOffer(
    source_station_departure_time = source_station_departure_secs,  ← 查询时刻，非列车发车时刻!
    waiting_time = source_waiting_time,  ← 在站台实际等待时间
    offered_waiting_time = waiting_time  ← 存入 t_wait
)
    ↓
user_confirms_booking():
    pu_time = source_station_departure_time = est + walking_time (查询时刻)
    do_time = target_station_arrival_time
```

### 3.2 关键发现：`source_station_departure_secs` 的语义

**`source_station_departure_secs` 是查询时间（旅客到达车站的时刻），不是 GTFS 列车发车时间！**

C++ 代码证据 (`Raptor.cpp:502`):
```cpp
journey.source_station_departure_secs = Utils::timeToSeconds(query_.departure_time);
```

这意味着 `source_station_departure_secs` = `est + walking_time`（午夜起算秒数），即旅客到达 PT 车站的时刻。

而 `source_waiting_time` 的计算 (`Raptor.cpp:528`):
```cpp
journey.source_waiting_time = journey.steps.front().departure_secs
                            - journey.source_station_departure_secs
                            - journey.source_transfer_time;
```

即：`source_waiting_time = 列车实际发车时间 - 旅客到达车站时间 - 站内换乘时间`

**这个值应该随 headway 变化而变化。**

### 3.3 评估公式分析

`calculate_pt_wait_time()` in `src/evaluation/intermodal.py:18-31`:

```python
pt_wait_time = (pt_pickup_time - amod_dropoff_time) - source_walking_time + pt_station_wait_time
```

各变量含义：
| 变量 | 来源 | PAYG 实际值 |
|------|------|-------------|
| `pt_pickup_time` (G_RQ_PU) | PT子请求的 pu_time = `source_station_departure_time` | `alighting_time + walking_time` |
| `amod_dropoff_time` (G_RQ_DO) | FM_AMOD子请求的实际下车时间 | `alighting_time` |
| `source_walking_time` | PTOffer 的 source_walking_time | `walking_time` |
| `pt_station_wait_time` (t_wait) | PTOffer 的 offered_waiting_time = RAPTOR source_waiting_time | 站台等待时间 |

**PAYG 模式下公式简化：**
```
pt_wait_time = (alighting_time + walking_time - alighting_time) - walking_time + source_waiting_time
             = walking_time - walking_time + source_waiting_time
             = source_waiting_time
```

**结论：对于 PAYG，公式正确地给出了 RAPTOR 的 `source_waiting_time`，即在站台的实际等待时间。这个值应该随 headway 变化。**

### 3.4 时间单位一致性验证

| 时间来源 | 单位 | 基准点 |
|----------|------|--------|
| 仿真时间 (sim_time) | 秒 | 午夜起算（如 start_time=0） |
| RAPTOR source_station_departure_secs | 秒 | 午夜起算 |
| RAPTOR source_waiting_time | 秒 | 相对时间差 |
| FM_AMOD G_RQ_DO | 秒 | 午夜起算 |
| PT G_RQ_PU | 秒 | 午夜起算 |

时间单位一致，无 mismatch 问题。

---

## 4. 可能的原因分析

### 代码逻辑无误，问题很可能在配置或数据层面

| 可能原因 | 可能性 | 验证方法 |
|----------|--------|----------|
| **三个场景使用了相同的 GTFS 数据** | ★★★★★ | 检查 scenario.csv 中 `gtfs_name` 参数是否指向不同 GTFS 目录 |
| **FM/FLM 样本量太少** | ★★★★ | 检查 `standard_eval.csv` 中 `FM_count` 和 `FLM_count` |
| **Selection bias: 长 headway 场景中高等待时间请求被过滤** | ★★★ | 比较各场景的 PAYG 中断率 (`payg_interrupt_rate`) |
| **offer 字段中 t_wait 始终为 0** | ★★ | 直接检查 `1_user-stats.csv` 中 PT 子请求的 offers 列 |
| **station_stop_transfer_time 过大** | ★ | 检查 `stations_fp.txt` 中的 `station_stop_transfer_times` |

---

## 5. 建议验证步骤

### Step 1: 确认 GTFS 数据差异
```bash
# 检查三个场景的 GTFS 配置
grep -i "gtfs" studies/<your_study>/scenarios/scenario.csv
grep -i "gtfs" studies/<your_study>/scenarios/constant_config.csv
```

### Step 2: 检查原始 PT 子请求数据
```python
import pandas as pd
import re

# 对每个场景读取 user-stats
df = pd.read_csv("studies/<scenario>/results/<name>/1_user-stats.csv")

# 筛选 PT 子请求 (sub_trip_id in [2, 3, 6])
pt_subs = df[df['sub_trip_id'].isin([2, 3, 6])]

# 提取 t_wait 值
def extract_t_wait(offer_str):
    if pd.isna(offer_str):
        return None
    match = re.search(r't_wait:(\d+)', str(offer_str))
    return int(match.group(1)) if match else None

pt_subs['t_wait_extracted'] = pt_subs['offers'].apply(extract_t_wait)

# 查看 t_wait 分布
print(pt_subs['t_wait_extracted'].describe())
print(f"样本量: {len(pt_subs)}")
```

### Step 3: 检查样本量和 PAYG 中断率
```python
# 检查 standard_eval.csv
eval_df = pd.read_csv("studies/<scenario>/results/<name>/standard_eval.csv", index_col=0)
print("FM count:", eval_df.loc['FM_count'])
print("FLM count:", eval_df.loc['FLM_count'])
print("PAYG interrupt rate:", eval_df.loc['payg_interrupt_rate [%]'])
```

### Step 4: 直接验证 RAPTOR 返回值
```python
# 在 Python 中直接测试 RAPTOR 对不同 GTFS 数据的返回值
from datetime import datetime
from src.routing.pt.RaptorRouterCpp import RaptorRouterCpp

# 分别加载三个 GTFS 数据
for gtfs_dir in ["data/pt/.../10min", "data/pt/.../20min", "data/pt/.../30min"]:
    router = RaptorRouterCpp(gtfs_dir)
    result = router.find_fastest_pt_journey_1to1("s1", "s5", datetime(2024, 1, 1, 7, 0, 0))
    if result:
        print(f"GTFS: {gtfs_dir}")
        print(f"  source_waiting_time: {result['source_waiting_time']}s")
        print(f"  source_station_departure_time: {result['source_station_departure_time']}s")
        print(f"  steps[0].departure_secs: {result.get('steps', [{}])[0].get('departure_time', 'N/A')}")
```

---

## 6. 附注：非 PAYG 场景下的公式问题

对于 **PTBrokerEI**（非 PAYG），`calculate_pt_wait_time` 公式可能存在问题：

- `pt_pickup_time` (G_RQ_PU) = `source_station_departure_time` = **查询时刻**（旅客到达车站时间），非列车实际发车时间
- `amod_dropoff_time` = 实际 AMoD 下车时间，可能早于或晚于查询时刻（因为 PTBrokerEI 使用估计时间）

如果 AMoD 提前送达（`amod_dropoff < est`），`pt_pickup_time - amod_dropoff_time` 会偏大；如果迟到（`amod_dropoff > est`），则偏小。但在 PTBrokerEI 的设计中，PT 查询时间是基于估计的 AMoD 到达时间，所以偏差取决于估计精度。

**PAYG 模式下不存在此问题**，因为 PT 查询时间 = AMoD 实际下车时间。
