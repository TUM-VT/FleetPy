# TRE PTBroker revision scenarios

本 study 准备首次投稿论文中全部 **PTBroker／DRT 参数组合**，并新增 EI 60% 预测、rebooking 关闭和开启的配对场景。用户已明确本 study 不包括 fixed-route bus benchmark，也不加入论文以外的请求模式。这里仅准备输入配置；**尚未执行任何模拟**。

## 参数网格与数量

| 参数 | 取值 |
| --- | --- |
| 主分析期 demand size | 1270、3720、7260 |
| intermodal ratio | 20%、50%、80% |
| DRT fleet size | 50–450，间隔50 |
| vehicle capacity | 4、8、20 |
| `op_max_detour_time_factor` | 40%、80% |
| `op_max_wait_time` | 300、600 s |
| rail headway | 10、20、30 min |
| random seed | 333、666、999 |

论文明确使用三种 intermodal ratio，不包括 `ir=100`。三个 seed 来自原始实验归档，论文正文没有规定重复次数。每种策略配置和车型包含 `3 × 3 × 9 × 2 × 2 × 3 × 3 = 2916` 个场景：

| 文件组名 | 策略 / EI 配置 | 每种车型场景数 | 三种车型合计 |
| --- | --- | ---: | ---: |
| `cc_c4`、`cc_c8`、`cc_c20` | `PTBroker`（CC） | 2916 | 8748 |
| `ei100_off_c4`、`ei100_off_c8`、`ei100_off_c20` | `PTBrokerEI`，上限预测、rebooking关闭 | 2916 | 8748 |
| `payg_c4`、`payg_c8`、`payg_c20` | `PTBrokerPAYG` | 2916 | 8748 |
| `ei60_off_c4`、`ei60_off_c8`、`ei60_off_c20` | `PTBrokerEI`，60%预测、rebooking关闭 | 2916 | 8748 |
| `ei60_on_c4`、`ei60_on_c8`、`ei60_on_c20` | `PTBrokerEI`，60%预测、rebooking开启 | 2916 | 8748 |

原三策略 baseline 共 **26,244** 个；新增 EI 60% 开关场景 **17,496** 个；总计 **43,740** 个，分成 15 个 CSV，每个 CSV 2,916 行。`scenario_name` 在所有文件中唯一。每个 EI60 的 off/on 场景共享相同 demand、seed、车型、初始状态及 operator 上限。

## EI 估计参数

基线 EI 显式设置 `broker_ei_assumed_wait_time` 和 `broker_ei_assumed_detour_ratio` 为 operator 上限，并设置 `broker_ei_rebooking_enabled=False`。新增 60% 场景将两个**假设参数**分别设为上限的60%，不改变 operator 限制：

| operator 上限 | EI 60% 假设 |
| --- | --- |
| wait 300 / 600 s | 180 / 360 s |
| detour ratio 40% / 80% | 24% / 48% |

预测的完整行程时间不会直接变为原来的60%。rebooking 开启／关闭只是额外敏感性对照，开启本身不保证一定发生误车并触发重订。

## 输入和时间

- 常量配置：`scenarios/constant_config.csv`；15 个 scenario CSV 在同一目录。
- 道路网络：`data/networks/TRE-PTBroker_network`，使用 `NetworkTTMatrix`。
- 需求：`data/demand/TRE-PTBroker_demand_revision/matched/TRE-PTBroker_network`，仅含 `MONOMODAL` 和 `FIRSTLASTMILE`。
- PT：`data/pt/TRE-PTBroker_network/train-headway-{10,20,30}`；车型位于 `data/vehicles/`。
- `results/vehicle_positions_<fleet>-<seed>/final_state.csv` 是从首次投稿归档复制的 **27 份车辆初始位置输入**，每份两区各半；这些不是新仿真输出。
- `scenario_manifest.json` 记录 scenario 数量和源文件／输出文件及初始状态的 SHA-256。`scripts/generate_scenarios.py` 记录转换方法，若目标文件已存在会拒绝覆盖。

正常仿真为 `[3600,21600)`（01:00–06:00）：一小时 warm-up、三小时主期、一小时带需求的 cool-down。`evaluation_int_start=7200` 和 `evaluation_int_end=18000` 用来选择主期 parent request cohort。这是 revision 的新设计；初投稿的原始仿真只有三小时，不能将两者当作相同运行条件。

## 执行入口

`run_revision.py` 默认检查并运行全部 15 组、43,740 个 scenarios。它使用根目录 `run_scenarios.py` 的单场景执行函数，默认 24 个并行 worker（针对 i9-14900K 的 24 个物理核心），每个 worker 完成一个 scenario 后动态领取下一个。场景顺序以固定种子打散，不修改 scenario 的随机种子，以混合不同规模的任务。默认数是起点，并不保证吞吐量最优；应在服务器上根据实际吞吐、内存和温度调整 `--workers`。程序在导入数值库之前将常见 BLAS/OpenMP 线程数设为 1，`n_cpu_per_sim=1`。

先列出文件组或检查全部场景，不启动模拟：

```bash
python -B studies/TRE-PTBroker_revision/run_revision.py --list-groups
python -B studies/TRE-PTBroker_revision/run_revision.py --dry-run
```

在仓库根目录、已激活的 `fleetpy` 环境中运行全部场景：

```bash
python -B studies/TRE-PTBroker_revision/run_revision.py
```

可通过 `--workers 32` 调整并发，通过 `--progress-interval 60` 调整进度报告间隔（秒）；也可用 `--group ei60_on_c8` 单独运行 2,916 个场景。终端显示完成及失败数量、正在运行的 scenario、已用时间和基于当前吞吐量粗估的剩余时间及总时长。各场景耗时差异较大，尤其开始时估计可能不准。FleetPy 的 stdout/stderr 保存在 `run_logs/<run-id>/<scenario_name>.log`，其自身的 `00_simulation.log` 仍在 scenario 结果目录中。

每次进度报告还显示 Linux 整机 CPU 使用率、内存使用量／总量和 CPU 温度。CPU 使用率按两次报告之间的 `/proc/stat` 变化计算；内存使用量为 `/proc/meminfo` 中的 `MemTotal - MemAvailable`，它们反映整机而非仅本次仿真。温度读取 Intel `coretemp` 的 package 传感器；没有 package 读数时使用最高 core 温度并标明来源。某项系统接口或传感器不可用时，该项显示 `N/A`，不影响模拟。初次 CPU 采样没有可比较的历史值时也显示 `N/A`。无需额外安装 Python 监控库；温度是否可见取决于服务器的 `coretemp` 传感器接口。

在服务器启动模拟前，可以用 `python -B studies/TRE-PTBroker_revision/run_revision.py --hardware-check` 采样一次硬件信息；它等待约一秒以计算 CPU 使用率，不会启动模拟。

启动前会检查所有选定场景的请求模式、重复名称和已有输出。任何选定 scenario 的输出路径已存在（包括不完整结果），都会拒绝启动；同一 study 同时只允许一个本入口运行。任何 scenario 报错时，保留日志、停止派发新场景，等待已派发的场景结束后以退出码 1 退出。Ctrl-C 同样会停止派发并等待，退出码为 130；如果某个任务卡住，等待可能很长。其他终止方式可能留下不完整输出或仍在运行的子进程，重新启动前应检查。当前入口不支持自动续跑或直接选择自定义恢复文件，重新运行前需要检查并保留既有输出，再另行准备未完成场景的执行入口。大量结果和日志需要足够的磁盘空间。

## 已知边界

这套输入保留原数据的 `im_lastmile_wait_time=30`；当前 CC 可出现30秒 LM 等待，与论文 exact pickup 的字面口径不同。现有铁路时刻表的始发站 departure 到终点 arrival 为25分钟，论文写30分钟；EI 当前预测也包含 FleetPy 上下客时间。上述差异和原始论文复现之间应单独说明。

当前评价按 06:00 作为完成观察上限，之后的车辆清尾不保证所有 PAYG/PT 后续旅段继续激活；3小时车辆指标的窗口截取、分模式服务率分母和 walking fallback 的 rebooking 处理仍需另行核验。场景配置已准备和静态解析通过，不代表所有场景已经运行、策略已完全验收或论文结果已复现。

该 study 位于仓库现有 `.gitignore` 的 `studies/*` 范围。场景 CSV、车辆初始状态和实验输入为本地文件，需另行复制到服务器；不要假定 Git 克隆会包含它们。
