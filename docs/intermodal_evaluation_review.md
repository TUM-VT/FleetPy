# Intermodal Evaluation 代码审阅报告

## 1. 概述

本报告审阅 `src/evaluation/intermodal.py` 的实现，对比 `src/evaluation/standard.py`，分析计算方法的正确性、结果的可比性，以及缺失的 KPIs。

---

## 2. 代码结构对比

### 2.1 Standard Evaluation (`standard.py`)
- **输入**: `1_user-stats.csv`, `2-{op_id}_op-stats.csv`
- **分组**: 按 `operator_id` 分组评估
- **输出**: 每个运营商一列的 KPIs

### 2.2 Intermodal Evaluation (`intermodal.py`)
- **预处理**: 将 sub-trips 聚合为 parent requests (`create_parent_user_stats`)
- **输入**: `1_user-stats.csv` → `1_user-stats_parent.csv`
- **分组**: 整体评估（不区分运营商）
- **输出**: 单一 "Intermodal" 列的 KPIs

---

## 3. 计算方法分析

### 3.1 Wait Time 计算 ✅ 正确

**MONOMODAL (DRT_only)**:
```python
total_wait_time = pickup_time - request_time
```

**FIRSTMILE (FM)**: AMoD → PT
```python
fm_amod_wait = fm_amod_pickup - request_time
fm_pt_wait = (pt_pickup - amod_dropoff) - source_walking_time + pt_station_wait_time
total_wait_time = fm_amod_wait + fm_pt_wait
```

**LASTMILE (LM)**: PT → AMoD
```python
lm_pt_wait = pt_station_wait_time (from offer string)
lm_amod_wait = lm_amod_pickup - pt_dropoff - target_walking_time
total_wait_time = lm_pt_wait + lm_amod_wait
```

**FIRSTLASTMILE (FLM)**: AMoD → PT → AMoD
```python
flm_amod_0_wait = flm_amod_0_pickup - request_time
flm_pt_wait = calculate_pt_wait_time(...)
flm_amod_1_wait = flm_amod_1_pickup - pt_dropoff - target_walking_time
total_wait_time = flm_amod_0_wait + flm_pt_wait + flm_amod_1_wait
```

**评价**: 逻辑正确，考虑了步行时间和换乘等待。

### 3.2 Travel Time 计算 ⚠️ 需要确认

当前实现:
```python
# FM: total_travel_time = pt_dropoff - amod_pickup
# LM: total_travel_time = amod_dropoff - pt_pickup
# FLM: total_travel_time = last_amod_dropoff - first_amod_pickup
```

**问题**: Travel time 是否应包含中间的换乘等待时间？当前实现**不包含**换乘等待，但这可能导致与 standard evaluation 不一致。

**建议**: 明确定义:
- `travel_time`: 纯车内时间（当前实现）
- `total_trip_time`: 包含所有等待的完整行程时间

### 3.3 Service Rate 计算 ✅ 正确

```python
is_served = all mandatory sub-trips are served
service_rate = served_count / total_count * 100
```

逻辑正确，只有所有必需的 sub-trips 都完成才算 served。

### 3.4 Vehicle Metrics 计算 ✅ 基本正确

从 `2-{op_id}_op-stats.csv` 计算:
- Fleet utilization
- Total VKM
- Occupancy
- Empty VKM
- CO2 emissions

与 standard evaluation 使用相同的方法。

---

## 4. Standard vs Intermodal 输出对比

### 4.1 Standard Evaluation 输出的 KPIs (49 项)

| KPI | 描述 | Intermodal 是否有 |
|-----|------|------------------|
| `operator_id` | 运营商 ID | ✅ (-3 for Intermodal) |
| `number users` | 服务用户数 | ✅ |
| `number travelers` | 服务乘客数 | ✅ |
| `modal split` | 模式分担率 (pax) | ✅ |
| `modal split rq` | 模式分担率 (requests) | ✅ |
| `reservation users` | 预约用户数 | ❌ 缺失 |
| `reservation pax` | 预约乘客数 | ❌ 缺失 |
| `served reservation users [%]` | 预约服务率 | ❌ 缺失 |
| `served reservation pax [%]` | 预约乘客服务率 | ❌ 缺失 |
| `online users` | 即时用户数 | ❌ 缺失 |
| `online pax` | 即时乘客数 | ❌ 缺失 |
| `served online users [%]` | 即时服务率 | ❌ 缺失 |
| `served online pax [%]` | 即时乘客服务率 | ❌ 缺失 |
| `% created offers` | Offer 创建率 | ❌ 缺失 |
| `utility` | 平均效用 | ❌ 缺失 |
| `travel time` | 平均行程时间 | ✅ |
| `travel distance` | 平均行程距离 | ❌ 缺失 |
| `waiting time` | 平均等待时间 | ✅ |
| `waiting time from ept` | 从 EPT 开始的等待时间 | ❌ 缺失 |
| `waiting time (median)` | 等待时间中位数 | ✅ |
| `waiting time (90% quantile)` | 等待时间 90 分位数 | ❌ 缺失 |
| `detour time` | 绕行时间 | ❌ 缺失 |
| `rel detour` | 相对绕行率 | ❌ 缺失 |
| `% fleet utilization` | 车队利用率 | ✅ (op0_fleet_utilization) |
| `rides per veh rev hours` | 每车时服务乘客 | ❌ 缺失 |
| `rides per veh rev hours rq` | 每车时服务请求 | ❌ 缺失 |
| `total vkm` | 总车公里 | ✅ (op0_total_vkm) |
| `occupancy` | 平均载客率 | ✅ (op0_occupancy) |
| `occupancy rq` | 请求载客率 | ❌ 缺失 |
| `% empty vkm` | 空驶率 | ✅ (op0_empty_vkm) |
| `% repositioning vkm` | 调度空驶率 | ❌ 缺失 |
| `customer direct distance [km]` | 乘客直接距离 | ❌ 缺失 |
| `saved distance [%]` | 节省距离百分比 | ❌ 缺失 |
| `trip distance per fleet distance` | 行程/车队距离比 | ❌ 缺失 |
| `trip distance per fleet distance (no reloc)` | 行程/车队距离比(无调度) | ❌ 缺失 |
| `avg driving velocity [km/h]` | 平均行驶速度 | ❌ 缺失 |
| `avg trip velocity [km/h]` | 平均行程速度 | ❌ 缺失 |
| `vehicle revenue hours [Fzg h]` | 车辆收入小时 | ✅ (op0_vehicle_revenue_hours) |
| `total toll` | 总通行费 | ❌ 缺失 |
| `mod revenue` | MoD 收入 | ✅ |
| `mod fix costs` | 固定成本 | ✅ (op0_fix_costs) |
| `mod var costs` | 可变成本 | ✅ (op0_var_costs) |
| `total CO2 emissions [t]` | CO2 排放 | ✅ (op0_total_CO2_emissions) |
| `total external emission costs` | 外部排放成本 | ❌ 缺失 |
| `parking cost` | 停车成本 | ❌ 缺失 |
| `toll` | 通行费 | ❌ 缺失 |
| `customer in vehicle distance` | 乘客车内距离 | ❌ 缺失 |
| `shared rides [%]` | 拼车率 | ❌ 缺失 |

### 4.2 Intermodal 特有的 KPIs (17 项)

| KPI | 描述 |
|-----|------|
| `Service_Rate [%]` | 整体服务率 |
| `Service_Rate_Pax [%]` | 乘客服务率 |
| `DRT_only_count` | DRT-only 行程数 |
| `FM_count` | First-mile 行程数 |
| `LM_count` | Last-mile 行程数 |
| `FLM_count` | First-last-mile 行程数 |
| `PT_only_count` | PT-only 行程数 |
| `DRT_only_service_rate [%]` | DRT-only 服务率 |
| `FM_service_rate [%]` | FM 服务率 |
| `LM_service_rate [%]` | LM 服务率 |
| `FLM_service_rate [%]` | FLM 服务率 |
| `PT_only_service_rate [%]` | PT-only 服务率 |
| `PT_Wait_Time_FM [s]` | FM 的 PT 等待时间 |
| `PT_Wait_Time_FLM [s]` | FLM 的 PT 等待时间 |
| `PT_Wait_Time_Combined [s]` | 综合 PT 等待时间 |
| `LM_Wait_Time_LM [s]` | LM 的 AMoD 等待时间 |
| `LM_Wait_Time_FLM [s]` | FLM 的 LM AMoD 等待时间 |
| `LM_Wait_Time_Combined [s]` | 综合 LM 等待时间 |

---

## 5. 缺失的 KPIs 分析

### 5.1 高优先级 - 应该添加

| 缺失 KPI | 重要性 | 添加难度 | 说明 |
|----------|--------|----------|------|
| `detour time` | 高 | 中 | 需要计算实际行程与直接行程的差异 |
| `rel detour` | 高 | 中 | 相对绕行率，服务质量关键指标 |
| `waiting time (90% quantile)` | 高 | 低 | 服务质量尾部指标 |
| `shared rides [%]` | 高 | 中 | 拼车效率指标 |
| `customer in vehicle distance` | 高 | 中 | 乘客实际行驶距离 |
| `% repositioning vkm` | 高 | 低 | 调度效率指标 |
| `rides per veh rev hours` | 中 | 低 | 运营效率指标 |

### 5.2 中优先级 - 建议添加

| 缺失 KPI | 重要性 | 添加难度 | 说明 |
|----------|--------|----------|------|
| `waiting time from ept` | 中 | 低 | 从 EPT 计算的等待时间 |
| `saved distance [%]` | 中 | 中 | 节省距离，需要直接距离数据 |
| `avg driving velocity [km/h]` | 中 | 低 | 平均速度 |
| `total external emission costs` | 中 | 低 | 外部成本 |
| `% created offers` | 中 | 中 | Offer 创建率 |

### 5.3 低优先级 - 可选添加

| 缺失 KPI | 重要性 | 说明 |
|----------|--------|------|
| `reservation users/pax` | 低 | 预约相关统计 |
| `online users/pax` | 低 | 即时相关统计 |
| `utility` | 低 | 效用计算复杂 |
| `parking cost`, `toll` | 低 | 特定场景才需要 |

---

## 6. 计算方法问题

### 6.1 问题 1: Detour Time 缺失

**Standard**:
```python
op_avg_detour_time = (travel_time_sum - direct_route_time_sum) / n_users - boarding_time
```

**Intermodal**: 未实现

**建议**: 需要为每种 modal state 分别计算 detour:
- MONOMODAL: 与 standard 相同
- FM/LM/FLM: 需要考虑 PT 段是否计入 detour

### 6.2 问题 2: Travel Time 定义不一致

**Standard**: `dropoff_time - pickup_time`（纯车内时间）

**Intermodal**:
- MONOMODAL: `dropoff_time - pickup_time` ✅
- FM: `pt_dropoff - amod_pickup`（包含换乘步行）
- LM: `amod_dropoff - pt_pickup`（包含换乘步行）
- FLM: `last_amod_dropoff - first_amod_pickup`（包含所有中间时间）

**问题**: Intermodal 的 travel time 实际上是 "door-to-door time minus initial wait"，而不是纯车内时间。

**建议**:
1. 添加 `in_vehicle_time` 指标（纯车内时间）
2. 重命名当前 `travel_time` 为 `trip_time`（从第一段 pickup 到最后 dropoff）

### 6.3 问题 3: 收入计算

**Standard**: 只计算 AMoD fare

**Intermodal**: `total_fare = sum(all_leg_fares)` 包含 PT fare

**问题**: `mod revenue` 在 intermodal 中包含了 PT revenue，与 standard 不直接可比。

**建议**: 分开报告:
- `amod_revenue`: 仅 AMoD 收入
- `pt_revenue`: 仅 PT 收入
- `total_revenue`: 总收入

### 6.4 问题 4: 缺少 DRT-only 的详细指标

当前 intermodal evaluation 将所有 modal states 混合计算，缺少对 DRT-only 行程的详细分析。DRT-only 行程应该与 standard evaluation 的结果完全一致。

**建议**: 为 DRT-only 单独输出一套与 standard 兼容的指标。

---

## 7. 改进建议

### 7.1 短期改进 (建议立即实施)

1. **添加 `waiting time (90% quantile)`**
```python
result_dict['waiting time (90% quantile)'] = served_requests['total_wait_time'].quantile(q=0.9)
```

2. **添加 `% repositioning vkm`**
```python
repo_df = op_vehicle_df[op_vehicle_df[G_VR_STATUS] == "reposition"]
repo_vkm = repo_df[G_VR_LEG_DISTANCE].sum() / 1000.0 / total_km * 100.0
```

3. **添加 `rides per veh rev hours`**
```python
result_dict[f'op{op_id}_rides_per_veh_rev_hours'] = total_served / vehicle_revenue_hours
```

4. **添加 `total external emission costs`**
```python
EMISSION_CPG = 145 * 100 / 1000**2
result_dict[f'op{op_id}_external_emission_costs'] = np.rint(EMISSION_CPG * total_co2)
```

### 7.2 中期改进 (建议下一版本)

1. **实现 detour time 计算**
   - 需要为每种 modal state 定义 "direct route"
   - MONOMODAL: origin → destination
   - FM/LM/FLM: 需要定义什么是 "direct" (纯 PT? 纯 AMoD?)

2. **实现 shared rides 计算**
   - 从 vehicle stats 中分析共乘情况
   - 复用 `standard.py` 中的 `shared_rides()` 函数

3. **分离 revenue 报告**
   - `amod_revenue`, `pt_revenue`, `total_revenue`

### 7.3 长期改进 (建议未来版本)

1. **DRT-only 兼容性输出**
   - 为 DRT-only 行程输出一套与 standard 完全兼容的指标
   - 便于与纯 AMoD 场景直接对比

2. **统一 travel time 定义**
   - 区分 `in_vehicle_time` 和 `trip_time`

3. **模块化重构**
   - 将通用的 vehicle metrics 计算提取为公共函数
   - 避免代码重复

---

## 8. 结果可比性分析

### 8.1 可直接对比的指标

| 指标 | Standard 名称 | Intermodal 名称 | 说明 |
|------|--------------|-----------------|------|
| 服务用户数 | `number users` | `number users` | ✅ 直接可比 |
| 等待时间 | `waiting time` | `waiting time` | ⚠️ 定义略有不同 |
| 车队利用率 | `% fleet utilization` | `op0_fleet_utilization [%]` | ✅ 直接可比 |
| 总 VKM | `total vkm` | `op0_total_vkm` | ✅ 直接可比 |
| 载客率 | `occupancy` | `op0_occupancy` | ✅ 直接可比 |
| 空驶率 | `% empty vkm` | `op0_empty_vkm [%]` | ✅ 直接可比 |
| CO2 排放 | `total CO2 emissions [t]` | `op0_total_CO2_emissions [t]` | ✅ 直接可比 |

### 8.2 不可直接对比的指标

| 指标 | 原因 |
|------|------|
| `travel time` | Intermodal 包含换乘时间 |
| `mod revenue` | Intermodal 包含 PT revenue |
| `modal split` | 分母定义不同 |

### 8.3 数值验证 (example_pool_sc_1 vs example_im_ptbroker)

| 指标 | Standard (pool) | Intermodal (ptbroker) | 说明 |
|------|-----------------|----------------------|------|
| Fleet utilization | 72.7% | 87.4% | Intermodal 需求更多 |
| Total VKM | 219.8 km | 204.2 km | 合理（部分行程用 PT）|
| Occupancy | 0.74 | 0.79 | 合理 |
| Empty VKM | 33.7% | 34.4% | 相近 |
| Waiting time | 137.2s | 253.8s | Intermodal 更高（包含 PT 等待）|

---

## 9. 总结

### 9.1 计算方法正确性: ✅ 基本正确

- Wait time 计算逻辑正确
- Service rate 计算正确
- Vehicle metrics 计算与 standard 一致

### 9.2 主要问题

1. **缺失关键 KPIs**: detour time, shared rides, 90% quantile wait time
2. **定义不一致**: travel time, revenue 在两种评估中定义不同
3. **可比性有限**: 部分指标无法直接与 standard 对比

### 9.3 建议优先级

1. **立即**: 添加 waiting time 90% quantile, repositioning vkm, external emission costs
2. **短期**: 添加 detour time, shared rides 计算
3. **中期**: 统一指标定义, 提高可比性
4. **长期**: 为 DRT-only 输出 standard-compatible 指标

---

## 10. 已实施的改进 (2026-02-04)

### 10.1 新增的 AMoD-only 指标

为了与 standard evaluation 保持可比性，新增了以下仅计算 AMoD 段的指标（忽略 PT 段）：

| 新增 KPI | 说明 |
|----------|------|
| `amod_waiting_time` | AMoD 段平均等待时间 |
| `amod_waiting_time (median)` | AMoD 段等待时间中位数 |
| `amod_waiting_time (90% quantile)` | AMoD 段等待时间 90 分位数 |
| `amod_travel_time` | AMoD 段平均行程时间 |
| `amod_detour_time` | AMoD 段平均绕行时间 |
| `amod_rel_detour [%]` | AMoD 段相对绕行率 |
| `amod_revenue` | AMoD 段总收入 |
| `amod_customer_direct_distance [km]` | AMoD 段乘客直接距离 |

### 10.2 新增的车辆层面指标

| 新增 KPI | 说明 |
|----------|------|
| `waiting time (90% quantile)` | 整体等待时间 90 分位数 |
| `op{id}_repositioning_vkm [%]` | 调度空驶率 |
| `op{id}_rides_per_veh_rev_hours` | 每车时服务乘客数 |
| `op{id}_rides_per_veh_rev_hours_rq` | 每车时服务请求数 |
| `op{id}_external_emission_costs` | 外部排放成本 |
| `op{id}_shared_rides [%]` | 拼车率 |
| `op{id}_customer_in_vehicle_distance` | 乘客车内距离 |

### 10.3 数据流改进

在 `create_parent_user_stats` 函数中新增了以下字段的计算：
- `amod_fare`: AMoD 段票价
- `amod_wait_time`: AMoD 段等待时间
- `amod_travel_time`: AMoD 段行程时间
- `amod_direct_distance`: AMoD 段直接距离

### 10.4 验证结果

运行 `example_im_ptbroker` 场景后的新增 KPIs 示例：
```
amod_waiting_time: 141.45 s
amod_waiting_time (90% quantile): 275.82 s
amod_travel_time: 270.93 s
amod_detour_time: 28.21 s
amod_rel_detour [%]: 9.81 %
op0_shared_rides [%]: 40.59 %
op0_customer_in_vehicle_distance: 1606.28 m
```

---

*报告生成时间: 2026-02-04*
*审阅文件: `src/evaluation/intermodal.py`, `src/evaluation/standard.py`*
*更新时间: 2026-02-04 - 添加高优先级 KPIs*
