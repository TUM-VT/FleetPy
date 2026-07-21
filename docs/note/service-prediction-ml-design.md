# FleetPy Service Prediction：项目目标与路线

> 更新：2026-07-21
> 当前阶段：Fleet State 数据采集已完成；离线样本、模型训练和在线写回尚未实现
> 当前实现与验证状态见 [ML 集成与 Fleet State 当前开发状态](ml-module-status.md)

## 1. 项目目标

本项目希望在 FleetPy operator 生成 offer 或 rejection 的真实决策边界，根据当时可见的车队、路线和
请求上下文，预测用户最终实现的服务水平：

```text
realized waiting time = pickup_time - rq_time
realized arrival time = dropoff_time - rq_time
```

标签来自同一 scenario 结果目录的 `1_user-stats.csv`。Fleet State 只保存决策时可见的输入，不提前写入
最终 pickup/dropoff 结果，避免未来信息泄漏。

当前回归目标只包含最终完成服务且具有 pickup/dropoff 时间的请求。拒绝、取消和未完成请求若要保留，
应作为独立的 outcome/删失建模任务，而不是给回归标签填充伪时间。

## 2. 数据采集原则

Immediate 和 Batch 的用户决策机制不同，不能使用同一种采样粒度：

| 控制模式 | 决策对齐的 Fleet State | 训练目标关联 |
| --- | --- | --- |
| Immediate | 每个请求提交给 broker 前记录一行 | 当前请求同时出现在 `prediction_request_ids` 中 |
| Batch | 每个 `(op_id, sim_time)` 在 `time_trigger()` 前记录一行 | 只展开非空 `prediction_request_ids` 对应的请求 |

Immediate 同一时刻的请求按 FleetPy 实际顺序处理：后一个请求可以看到前一个请求已经确认造成的计划
变化，但任何请求都看不到自身提交后的状态。

Batch 的到达时刻和 offer 时刻可能不同。`new_request_ids` 记录本步到达，真正的训练样本使用实际优化/
offer 时刻的 Fleet State 和 `prediction_request_ids`。已经上车或锁车的请求由车辆和路线状态表达，
不作为新的预测目标重复展开。

## 3. 当前研究范围

当前验证只覆盖：

- 固定数量、连续 vid 的标准内置 `SimulationVehicle`；
- 纯客运 MoD；
- `ImmediateDecisionsSimulation + PoolingIRSOnly`；
- `BatchOfferSimulation + RidePoolingBatchAssignmentFleetcontrol`；
- Batch optimizer 为 `InsertionHeuristic` 或 `AlonsoMora`；
- 无 Immediate retry、advance reservation、second waiting-time window 和 mutually-exclusive sub-RID。

parcel、充电/电池、动态车辆、SoD、外部移动仿真、多 operator 和其他 FleetControl/optimizer 组合不在
当前完成声明内。

## 4. 当前开发进度

决策边界设计、Fleet State 收集和场景验证已经完成；离线样本、模型训练和在线写回尚未实现。详细验证
结果和唯一的进度表保存在[当前开发状态](ml-module-status.md#10-开发进度摘要)中。“采集完成”不表示
Service Prediction 项目已经完成。

## 5. 离线数据计划

训练数据应按以下顺序构建：

1. 读取每个 scenario 的 `fleet_states.jsonl` 和 `00_config.json`；
2. Immediate 每行展开一个 `prediction_request_id`，Batch 每行展开该列表中的全部目标请求；
3. 使用 scenario、`op_id` 和公共 RID 与 `1_user-stats.csv` 关联；
4. 计算 waiting/arrival 标签，并明确处理拒绝、取消和未完成请求；
5. 从 `columns/leg_columns/stop_columns` 解码车辆、执行 route 和 plan stop；
6. 按 scenario、随机种子或时间块划分训练/验证/测试集，避免同一轨迹跨集合泄漏。

首个模型应包含简单且可解释的基线，例如 FleetPy 初始计划、线性/树模型或对 planned time 的 residual
correction。集合/序列模型只有在基线和数据质量验证稳定后再引入。

## 6. 下一步

1. 将当前文档、测试和采集实现纳入版本控制，并记录正式 commit；
2. 在干净结果目录复跑单元测试和统一场景；
3. 实现可重复的离线样本构建与数据质量报告；
4. 建立 waiting/arrival 基线模型和按场景外推的评估；
5. 再决定是否增加未来需求特征、集合编码器和不确定性输出；
6. 在线阶段通过独立预测缓存关联后续 offer 创建，不让 Immediate Hook 直接修改尚未注册的请求对象。

项目阶段性完成标准是：采集输入可复现、训练标签可追溯、模型相对 FleetPy 计划基线有稳定增益，并且
在线推理仍严格使用决策时刻可见的信息。
