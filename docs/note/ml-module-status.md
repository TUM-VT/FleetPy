# FleetPy ML 集成与 Fleet State：当前开发状态

> 更新：2026-07-21
> 历史代码基线：`d7f1f3c`（2026-06-22）；Fleet State 改动仍位于该基线之上的本地工作树
> 项目目标与后续路线见 [Service Prediction 项目目标与路线](service-prediction-ml-design.md)

## 1. 当前结论

FleetPy 的通用 ML 集成保持 `HookManager → Observer → Actor` 分层。当前已经完成一条不依赖 Gym 的
Service Prediction 输入采集链：

```text
Simulation event
  → HookManager
  → FleetStateObserver
  → JSONWriter
  → <scenario result directory>/fleet_states.jsonl
```

在固定连续车队、纯客运、标准 Immediate/Batch 控制和指定 optimizer 的研究范围内，FieldGetters 的数据
来源、采集时序和三组请求 ID 已通过单元测试与场景结果核对。模型训练和在线预测不属于当前已完成内容。

## 2. 现有组件

| 组件 | 位置 | 当前职责 |
| --- | --- | --- |
| Hook/事件 | `src/ml_gym/hooks_manager.py` | 在确定的仿真事件聚合 Observer 输出并调度 Actor |
| Fleet State | `src/ml_gym/Observers/fleet_state_observers.py` | 编码车辆、执行 route、规划 stop 和请求上下文 |
| JSON Writer | `src/ml_gym/Actors/writers.py` | 每个 observation 追加一行严格 JSON |
| 收集入口 | `src/ml_gym/example_fleet_state_collection.py` | 按 scenario 注册事件、Observer 和独立结果文件 |
| 多场景调度 | `src/ml_gym/FleetPyMLInterface.py` | 顺序或限并发运行不同 scenario；要求名称唯一 |
| Gym 适配 | `src/ml_gym/FleetPyGymInterface.py` | 将 Hook 往返映射为 Gymnasium `reset()/step()` |
| 重定位扩展 | `MLZoneBasedRepositioning` 及对应 Observer/Actor | 在 FleetPy 重定位前接收外部 OD 动作 |

Fleet State 收集不使用 Gym。Gym/RLlib 与重定位路径仍是通用 ML 基础设施，本次 11 个 Service
Prediction 场景没有验证其训练效果。

Gym/RLlib 当前不能视为开箱即用路径：`environment.yml` 尚未声明 Gymnasium/Ray，相关示例仍指向不存
在的 `studies/MLtest`，registry 中的 `RLRepoFleetControl` 和 `SLPoolingIRSOnly` 也指向已删除模块。

## 3. 当前事件边界

| 事件 | 触发位置 | 语义 |
| --- | --- | --- |
| `OBSERVE_FLEET_STATE_BEFORE_IMMEDIATE_REQUEST_SUBMISSION` | `ImmediateDecisionsSimulation.step()` | 每个请求提交给 broker 前 |
| `OBSERVE_FLEET_STATE_BEFORE_BATCH_TIME_TRIGGER` | `BatchOfferSimulation.step()` | 本步请求/取消处理后、operator `time_trigger()` 前 |
| `OBSERVE_BEFORE_REPOSITIONING` | `RepositioningBase.determine_and_create_repositioning_plans()` | FleetPy 自身计算重定位前 |

两个 Fleet State 事件都发生在 `update_sim_state_fleets()` 完成之后。此时车辆执行状态、Demand 的上下客
记录、FleetControl acknowledgement 和 vehicle plan 状态已经同步，不是半更新快照。

## 4. Fleet State 输出契约

```python
{
    "op_id": int,
    "sim_time": int,
    "new_request_ids": list[str],
    "optimization_request_ids": list[str],
    "prediction_request_ids": list[str],
    "fleet_state": {
        "time": int,
        "op_id": int,
        "n_vehicles": int,
        "columns": list[str],
        "leg_columns": list[str],
        "stop_columns": list[str],
        "vehicles": list[list[object]],
    },
}
```

约束：

- `fleet_state.time == sim_time`，内外 `op_id` 相同；
- `n_vehicles == len(vehicles)`；
- vehicle、assigned route 和 plan stop 行分别按三组 columns 解码，不依赖固定下标；
- 所有顶层、vehicle、leg 和 stop RID 均为字符串；
- 空集合为 `[]`，缺失值为 JSON `null`，合法 0 和有意义的负值保留；
- JSON 行不包含 simulation mode、schema version 或 run ID；mode 由同目录 `00_config.json` 提供。

## 5. FieldGetters 数据来源

| 字段组 | 权威来源 |
| --- | --- |
| 车辆身份、状态、位置、容量 | 当前 `SimulationVehicle` |
| `n_pax/pax_rids` | active BOARDING 完成态 helper；其他状态使用 `SimulationVehicle.pax` |
| `cl_*` | `SimulationVehicle` current-leg 执行字段 |
| `cumulative_distance` | 已结算距离加当前 leg 已行驶距离 |
| `assigned_route` | `SimulationVehicle.assigned_route`，执行层 |
| `plan_stops` | `FleetControl.veh_plans[vid].list_plan_stops`，规划层 |
| leg 上下客 RID | `VehicleRouteLeg.rq_dict[1/-1]` |
| stop 上下客 RID/计划时间 | `PlanStop` 的公开 getter |

关键语义：

- FleetPy 在 BOARDING 开始时先加入 boarder、结束时才移除 alighter；公开 `n_pax/pax_rids` 表示该不可
  中断停站完成后的确定状态，人数按每个请求的 `nr_pax` 求和；
- `cl_locked` 从已开始的 `assigned_route[0].locked` 读取；
- leg `started` 定义为第一条 leg 且 `cl_start_time is not None`，避免 route rebuild 后的旧标志失真；
- 已知 absolute-time sentinel 按字段转换为 `null`，不会清洗所有负数；
- `cl_remaining_time` 是 FleetPy 原始字段，不是 Observer 计算的通用 ETA。

## 6. Immediate、Batch 与请求 ID

Immediate：

```text
完整车辆/FleetControl 状态更新
→ 对每个请求：Hook → broker.inform_request → insertion/offer → user decision
```

- 每个请求一行；三组 RID 都是当前请求；
- 当前请求不会泄漏进自己的 route/plan/passenger 状态；
- 同时刻后一个请求可以看到前一个请求已经确认造成的计划变化。

Batch：

```text
完整车辆/FleetControl 状态更新
→ 提交本步全部新请求
→ waiting cancellation
→ Hook
→ operator.time_trigger → optimization/offer → user decision
```

- 每个 `(op_id, sim_time)` 一行，包括非优化步；
- `new_request_ids` 是本步到达请求；
- `prediction_request_ids` 是本轮等待 offer/rejection、使用该快照的目标请求；
- `optimization_request_ids` 是本轮分配仍可改变的完整上下文；
- 非 due step 的 optimization/prediction 必为空；due step 也可能为空；
- Insertion 返回未确认候选；Alonso–Mora 还包含已接受但未锁车的请求，并排除已上车/锁车请求。

## 7. Schema、配置与运行

`FleetStateObserver` 支持 `mini`、`medium`、`max` 和 `custom`。custom 请求任意 leg/stop 字段时会自动加入
对应嵌套容器和数量列。有效配置已记录在 `Input_Parameters.md`：

- `ml_fleet_state_detail_level`；
- `ml_fleet_state_custom_veh`；
- `ml_fleet_state_custom_leg`；
- `ml_fleet_state_custom_stop`。

这些键只由显式收集入口消费；普通 FleetPy run 不会仅凭配置自动注册 Observer/Writer。

```bash
conda run -n fleetpy-ml \
  python src/ml_gym/example_fleet_state_collection.py \
  studies/ml_test/scenarios/constant_config_ir.csv \
  studies/ml_test/scenarios/example_ml_fleet_state.csv \
  1
```

`JSONWriter` 使用 append；重跑同名 scenario 前必须使用干净结果目录或新的唯一名称。

## 8. 验证状态

验证环境为 Conda `fleetpy-ml`、Python 3.10。

| 检查 | 结果 |
| --- | ---: |
| 本地单元测试 | 32/32 通过 |
| 统一研究场景 | 11/11 成功 |
| observations | 123 |
| vehicle rows | 231 |
| executable legs | 416 |
| plan stops | 224 |
| 本次结果审查：严格 JSON/schema/RID/cadence 与 user/operator stats | 未发现错误 |

覆盖重点包括 detail levels、同站上下客完成态、Batch reopt=120、Immediate 同时刻顺序和 Alonso–Mora
mutable/locked 请求边界。

生成结果是本次本地审查证据，不作为受版本控制的回归 fixture；正式发布前应在干净结果目录复跑。

测试命令：

```bash
conda run -n fleetpy-ml \
  python -m unittest \
  tests.ml_gym.test_fleet_state_observer_hardening \
  tests.ml_gym.test_service_prediction -v
```

## 9. 当前边界与已知问题

当前完成声明的适用范围见[项目目标文档](service-prediction-ml-design.md#3-当前研究范围)。

扩展前需要处理：

- `BOARDING_WITH_CHARGING` 尚未进入完成态乘员 helper；
- `BatchAssignmentAlgorithmBase.set_request_assigned()` 的 sibling 分支错删 `rid` 而不是 `other_rid`；
- reservation reveal、Immediate retry、真实 booking cancellation、多 operator 和真实多进程尚未端到端验证；
- Observer 假设连续 vid，RID 字符串化不做碰撞检测；
- Writer 无去重、同路径锁或事务发布；
- 当前入口没有自动装配或范围检查。

## 10. 开发进度摘要

| 阶段 | 结果 |
| --- | --- |
| 通用 Hook/Observer/Actor | 已形成稳定分层，并被重定位和 Fleet State 共用 |
| 多场景与 Gym 适配 | 接口已存在；不属于本次 Service Prediction 结果验证 |
| 决策对齐 Fleet State | 当前研究范围内完成并通过验证 |
| Service Prediction 数据集 | 未实现 |
| 模型训练与在线写回 | 未实现 |

当前下一步是把 `prediction_request_ids` 展开并与 user stats 构建可重复训练样本，而不是继续扩展采集
schema。后续阶段和完成标准见项目目标文档。
