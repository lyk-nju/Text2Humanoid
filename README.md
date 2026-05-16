# Text2Humanoid

`Text2Humanoid` 是一个总控工程。它本身不训练上游模型，也不重写下游 tracker，而是把三个已有项目按清晰接口接起来：

- `FloodNet`：文本 + 轨迹到 human motion
- `MakeTrackingEasy`：human motion 到 G1 reference motion
- `motion_tracking`：G1 sim2sim tracking runtime

这个工程的目标不是把三个仓库源码揉成一个仓库，而是提供一层稳定的编排、桥接、状态管理和调试基础设施，让后续可以逐步把系统推进到“在线流式文本驱动人形仿真”。

## 1. 项目定位

当前 `Text2Humanoid` 解决的是“系统怎么接起来”这个问题，而不是“某个单点模型怎么再提分”。

它负责：

- 定义跨模块的数据契约
- 管理会话和状态机
- 调用 `FloodNet` 生成 human motion chunk
- 把 `FloodNet` 的 `263D` 动作桥接到 `MakeTrackingEasy` 的 `140D` 输入
- 把 `MakeTrackingEasy` 输出适配成 `motion_tracking` 风格的 G1 reference motion
- 维护 runtime 侧 future-horizon reference buffer
- 对外暴露 HTTP + WebSocket 接口
- 记录 artifact，便于回放和调试

它明确不负责：

- 改写 `FloodNet` 训练主线
- 改写 `MakeTrackingEasy` 模型结构
- 改写 `motion_tracking` 的策略训练逻辑
- 直接做 sim2real

## 2. 当前阶段目标

第一阶段的边界已经固定：

- 输入：`文本 + 轨迹`
- 输出：`G1 reference motion chunks`
- 部署：`单机单 GPU`
- 服务形态：`长驻 HTTP + WebSocket`
- 目标场景：`sim2sim`

也就是说，这个仓库首先要成为一个“在线生成与桥接服务”。当前最小 demo 主线已经接通了 file-based runtime source；同时在线 runtime bridge 也已经最小闭环：主服务配置入口已能选择 `socket` backend，producer-side TCP smoke、consumer-side 协议模拟、`motion_tracking` 侧在线 consumer，以及真实 runtime-consumer smoke 都已经存在并能在当前标准验证环境里直接跑通。当前 socket bridge 已经提升为默认 demo 主线（`demo_socket.yaml`），`demo_fixed.yaml` 保留为 file-based fallback / debug 路径。

## 3. 与三个外部仓库的关系

这个工程默认以下三个 sibling repo 已存在：

- `FloodNet`
- `MakeTrackingEasy`
- `motion_tracking`

设计原则是：

- `FloodNet` 只作为 planner backend，通过 Python wrapper 调用
- `MakeTrackingEasy` 只作为 retarget backend，通过 `infer_from_tensor()` 调用
- `motion_tracking` 只作为 execution backend，通过最小 patch 接入

这样做的好处很直接：

- 上游研究代码可以继续演化，不污染总控层
- retarget 后端可以替换，不影响 API 和 session 逻辑
- runtime 侧如果以后换 tracker，也只需要改 adapter 和 source protocol

## 4. 当前实现状态

当前仓库已经不是空骨架，而是一个可编译、可测试、可启动，并且能跑 fixed text + fixed trajectory demo、支持多次在线 command 序列最小语义的系统壳。

已经实现的部分：

- 完整的目录结构
- `contracts` 数据契约
- FastAPI HTTP/WebSocket 服务
- session manager 和 pipeline coordinator
- `FloodNet` wrapper
- `263D -> 140D` 桥接逻辑
- `MakeTrackingEasy` 推理 wrapper
- `G1ReferenceChunk` 适配逻辑
- runtime reference buffer
- planner-native session streaming
- 运行中 session 的最小受控 command transition
- file-based runtime source（`floodnet_file` backend → `motion_tracking` `FloodNetMotionSource`）
- 在线 runtime bridge（`SocketBackend` ↔ `SocketFloodNetSource` 最小主线已接通，producer-side TCP smoke、consumer-side 协议模拟，以及真实 runtime-consumer smoke 已可直接运行；`floodnet_file` 仍保留为 fallback / debug 路径）
- artifact 导出
- demo 配置、demo smoke 和最小 runbook
- 基础测试

已经修掉的关键工程问题：

- 避免直接通过 `MakeTrackingEasy/src` 顶层包导入而把完整训练栈和 `mmengine` 拉进来
- runtime buffer 按 `session` 隔离，不再全局共享
- cross-fade 对 `root_pos / root_rot / dof_pos` 做 blend，再从 blend 后状态通过 FK 重建 `local_body`，保证 FK 一致性
- `SessionManager` 现在按 chunk end time 推进下一段起始时间，而不是错误地拿 runtime 当前 `sim_time` 直接当 chunk 排程时间
- refill loop 已优先依赖 planner session，而不是简单重放最后一条 command
- `motion_tracking` 已能通过 session-scoped chunk 目录消费 `Text2Humanoid` 输出的 fixed clip / multi-chunk reference

当前还没有真正完成的部分：

- 还没有做更复杂的多次 prompt / trajectory 在线切换策略
- `FloodNetPlannerService` 当前仍然是 `model.generate()` 的 session 包装，不是最终形态的 planner-native 长驻生成服务
- waypoint 到 FloodNet 真实轨迹条件的高质量注入还比较薄，当前更偏“接口预留”
- 还没有把 planner / retarget / runtime 拆成独立后台进程和稳定 IPC

所以更准确地说，现在的 `Text2Humanoid` 已经是“可复现 demo 的 orchestrator scaffold”，但还不是“支持复杂在线交互的最终系统”。

## 5. 项目结构

```text
Text2Humanoid/
├── pyproject.toml
├── README.md
├── configs/
│   ├── system/
│   ├── floodnet/
│   ├── nmr/
│   └── runtime/
├── apps/
├── src/text2humanoid/
│   ├── contracts/
│   ├── api/
│   ├── orchestrator/
│   ├── planner/
│   ├── retarget/
│   ├── runtime/
│   ├── infra/
│   └── evaluation/
├── patches/
│   └── motion_tracking/
└── tests/
```

下面按模块说明职责。

## 6. 模块职责

### 6.1 `contracts/`

路径：

- [commands.py](./src/text2humanoid/contracts/commands.py)
- [chunks.py](./src/text2humanoid/contracts/chunks.py)
- [clips.py](./src/text2humanoid/contracts/clips.py)
- [status.py](./src/text2humanoid/contracts/status.py)

这是整个项目最重要的一层。它定义了系统内部允许跨模块传递的核心对象。

主要契约：

- `PromptCommand`
  - 文本命令
  - 轨迹条件
  - 切换模式
  - 提交时间
  - 命令元数据
- `TrajectoryCondition`
  - `waypoints`
  - `token_aligned_traj`
  - `token_mask`
- `HumanMotionChunk`
  - `FloodNet` 生成后的 `(T, 263)` 动作片段
  - 默认 `20 FPS`
- `NMRInputChunk`
  - 喂给 `MakeTrackingEasy` 的 `(T, 140)` 动作片段
  - 默认 `30 FPS`
- `G1ReferenceChunk`
  - 面向 `motion_tracking` 风格 runtime 的 reference chunk
  - 包含 `root_pos / root_rot / dof_pos / local_body_pos / local_body_rot / body_names / joint_names`
- `RuntimeStatus`
  - 当前 session 的阶段、buffer 水位、sim_time、延迟统计、错误信息等

设计原则：

- 一切跨模块数据都先落到契约对象，而不是 dict 到处乱传
- quaternion 顺序明确写在元数据里
- chunk 都带 `start_time / end_time / fps / num_frames`

### 6.2 `api/`

路径：

- [http_routes.py](./src/text2humanoid/api/http_routes.py)
- [schemas.py](./src/text2humanoid/api/schemas.py)
- [ws_events.py](./src/text2humanoid/api/ws_events.py)

这是对外控制面和观测面的定义。

当前 HTTP 接口：

- `POST /sessions`
  - 创建一个新 session
- `GET /sessions/{session_id}/status`
  - 查询当前状态
- `POST /sessions/{session_id}/commands`
  - 推入一条文本/轨迹命令
- `POST /sessions/{session_id}/refill/start`
  - 启动后台 refill loop
- `POST /sessions/{session_id}/refill/stop`
  - 停止后台 refill loop
- `POST /sessions/{session_id}/reset`
  - 重置 session
- `POST /sessions/{session_id}/stop`
  - 停止 session
- `POST /sessions/{session_id}/export`
  - 导出当前状态 bundle

当前 WebSocket 接口：

- `GET ws://host:port/ws/{session_id}`
  - 建立状态订阅连接

当前 WebSocket 行为还比较轻量：

- 建立连接时先发一条 `status`
- 当 HTTP command / refill / reset / stop 被调用时，广播事件

这意味着它已经具备“控制面 + 基本观测面”的壳，但还不是最终实时事件流系统。

### 6.3 `orchestrator/`

路径：

- [session_manager.py](./src/text2humanoid/orchestrator/session_manager.py)
- [pipeline_coordinator.py](./src/text2humanoid/orchestrator/pipeline_coordinator.py)
- [timeline.py](./src/text2humanoid/orchestrator/timeline.py)
- [state_machine.py](./src/text2humanoid/orchestrator/state_machine.py)

这是系统核心调度层。

职责拆分：

- `SessionManager`
  - 创建和管理 session
  - 记录 timeline
  - 接收 command
  - 调用 coordinator 执行一次 pipeline
- `PipelineCoordinator`
  - 串联 planner -> bridge -> retarget -> adapter -> runtime
  - 记录 planner / retarget / runtime 三段耗时
- `timeline`
  - 保存 session 内部的命令序列
- `state_machine`
  - 定义 `idle / warming / running / degraded / resetting / stopped / error`

当前行为模型：

- `push_command()` 是同步的
- 每次命令会驱动一次 chunk 生成和一次 retarget
- `next_start_time` 由上一 chunk 的结束时间推进
- running session 上的第二条 command 已有最小 transition 语义：
  - timeline 会记录 transition 边界
  - planner session 会切到新的 active command
  - refill loop 后续继续沿用新的 active command

这套逻辑适合当前 scaffold 阶段，因为它先把数据边界和调度边界固定住了。后续如果要做真正后台 streaming loop，主要改这里和 runtime/source protocol。

### 6.4 `planner/`

路径：

- [floodnet_service.py](./src/text2humanoid/planner/floodnet_service.py)
- [stream_driver.py](./src/text2humanoid/planner/stream_driver.py)
- [prompt_transition.py](./src/text2humanoid/planner/prompt_transition.py)
- [traj_conditioning.py](./src/text2humanoid/planner/traj_conditioning.py)

职责是把 `FloodNet` 变成可被总控层调用的 planner backend。

当前已经实现：

- 自动加载 `FloodNet` 配置、VAE、主模型和 EMA checkpoint
- `warmup(text)`
- `reset()`
- `generate_chunk(command, start_time)`

当前 planner 的实现方式：

- 使用 `FloodNet` 的 `model.generate()` 生成整段 latent
- 通过 VAE decode 得到 `(T, 263)` motion
- 包装成 `HumanMotionChunk`

需要明确的一点：

- 这不是最终在线 streaming planner，只是第一版可工作的 planner wrapper
- `stream_driver.py` 目前更像占位层，后续要承接真正的流式补帧逻辑

### 6.5 `retarget/`

路径：

- [bridge_263_to_140.py](./src/text2humanoid/retarget/bridge_263_to_140.py)
- [nmr_service.py](./src/text2humanoid/retarget/nmr_service.py)
- [g1_reference_adapter.py](./src/text2humanoid/retarget/g1_reference_adapter.py)
- [fk_features.py](./src/text2humanoid/retarget/fk_features.py)
- [mte_imports.py](./src/text2humanoid/retarget/mte_imports.py)

这是整个系统里算法风险最高的一层。

它做四件事：

1. `263D -> 22 joint + root`
2. `22 joint + root -> 140D NMR input`
3. `140D -> G1 dof/root`
4. `G1 dof/root -> motion_tracking 风格 reference chunk`

具体职责：

- `bridge_263_to_140.py`
  - 吸收原来根目录 `bridge.py` 的核心逻辑
  - 将 `FloodNet` 的动作特征恢复到 joint space
  - 组装 `MakeTrackingEasy` 需要的 `140D` 输入
- `nmr_service.py`
  - 最小包装 `MakeTrackingEasy/inference.py`
  - 调用 `load_all()` 和 `infer_from_tensor()`
- `fk_features.py`
  - 读取 `motion_tracking` 的 `tracking.yaml`
  - 使用 `MakeTrackingEasy` 的 `KinematicsModel`
  - 计算 `body_pos_w / body_rot_w / local_body_pos / local_body_rot`
- `g1_reference_adapter.py`
  - 统一 root quaternion 顺序
  - remap DOF 顺序
  - 产出 `G1ReferenceChunk`
- `mte_imports.py`
  - 避免不必要的顶层导入副作用
  - 保证只加载 retarget 必需的最小工具模块

这一层最大的难点不是代码，而是语义对齐：

- `FloodNet` 输出的人体表示和 `MakeTrackingEasy` 原始输入分布不一致
- `motion_tracking` 期待的 reference 字段语义必须和它训练用数据一致

所以任何“效果不稳”的问题，第一怀疑对象都应该是这一层，而不是先怪 tracker。

### 6.6 `runtime/`

路径：

- [motion_tracking_client.py](./src/text2humanoid/runtime/motion_tracking_client.py)
- [reference_buffer.py](./src/text2humanoid/runtime/reference_buffer.py)
- [source_protocol.py](./src/text2humanoid/runtime/source_protocol.py)
- [sync_manager.py](./src/text2humanoid/runtime/sync_manager.py)
- [fallback_policy.py](./src/text2humanoid/runtime/fallback_policy.py)

**重要：** 当前 runtime 层已经不再只有内存 shim。`MotionTrackingClient` 现在是一个 facade，后面可以挂：

- `ShimBackend`
  - 继续服务于单元测试和纯内存契约验证
- `FloodNetFileBackend`
  - 把 `G1ReferenceChunk` 按 session 写成 `chunk_*.npz`
  - 由 `motion_tracking` 侧的 `FloodNetMotionSource` 消费

也就是说，fixed demo 主线已经接上了最小真实 runtime source；但这仍然是 file-based bridge，不是最终的网络 source / 真正在线 streaming runtime API。

当前 `MotionTrackingClient` / backend 层的职责：

- 按 `session` 维护独立 reference buffer
- 接受 `G1ReferenceChunk`
- 维护 `buffer_frames`
- 维护 `sim_time`
- 支持 `consume_step()` 和 `reset_session()`
- 对 file backend 额外维护：
  - `chunk_index.json`
  - `stream_status.json`
  - session-scoped output dir

当前 `ReferenceBuffer` 的职责：

- append 新 chunk
- 支持 overlap / cross-fade
- 支持 horizon 读取
- 支持 cursor 推进

这里有两个重点：

- 当前 facade 同时支持 shim 和 file backend
- 真实 `motion_tracking` 已能通过 `FloodNetMotionSource` 消费 file backend 产出的 chunk 序列

也就是说，下一步不再是“从零接 runtime”，而是要在已接通的 file-based 主线之上，继续收敛在线切换和更稳定的连续生成语义。

### 6.7 `infra/`

路径：

- [paths.py](./src/text2humanoid/infra/paths.py) — 统一路径管理，`get_root()` / `set_root()`
- [config_loader.py](./src/text2humanoid/infra/config_loader.py) — 路径解析 + 组件构建
- [artifact_store.py](./src/text2humanoid/infra/artifact_store.py)
- [clocks.py](./src/text2humanoid/infra/clocks.py)
- [logging.py](./src/text2humanoid/infra/logging.py)
- [process_manager.py](./src/text2humanoid/infra/process_manager.py)

这是工程基础设施层。

当前已经实现：

- `ArtifactStore`
  - 保存 JSON
  - 保存 NPZ
  - 导出状态 bundle
- logging wrapper
- 时钟工具占位
- 进程管理占位

当前 artifact 默认很轻，只存状态和数组。后续如果要做完整回放，需要继续把：

- 原始 command
- human motion chunk
- NMR input
- G1 reference chunk
- runtime status timeline
- 可视化视频

都统一落盘。

### 6.8 `evaluation/`

路径：

- [online_metrics.py](./src/text2humanoid/evaluation/online_metrics.py)
- [buffer_metrics.py](./src/text2humanoid/evaluation/buffer_metrics.py)
- [replay_checks.py](./src/text2humanoid/evaluation/replay_checks.py)

这一层现在主要是结构占位，用来明确“在线系统的验收口径应该在这里，而不是复用 FloodNet 训练期 eval 脚本”。

原则很重要：

- 上游生成质量和下游执行稳定性要分开评估
- 在线延迟、buffer 水位、切换平滑度要单独记录

## 7. 数据流

当前系统的标准数据流是：

1. 客户端通过 HTTP 提交 `PromptCommand`
2. `SessionManager` 将命令写入 timeline
3. `PipelineCoordinator` 调用 `FloodNetPlannerService`
4. planner 生成 `HumanMotionChunk`
5. `bridge_263_to_140.py` 把其变成 `NMRInputChunk`
6. `NMRRetargetService` 调用 `MakeTrackingEasy`
7. `G1ReferenceAdapter` 产出 `G1ReferenceChunk`
8. runtime buffer 接收 chunk，并做 overlap / cross-fade
9. WebSocket 和状态接口返回 buffer / sim_time / latest_chunk 等信息

更紧凑的链路表示如下：

```text
HTTP PromptCommand
-> SessionManager
-> FloodNetPlannerService
-> HumanMotionChunk (T, 263)
-> bridge_263_to_140
-> NMRInputChunk (T, 140)
-> MakeTrackingEasy infer_from_tensor
-> G1 dof/root
-> G1ReferenceChunk
-> ReferenceBuffer
-> motion_tracking runtime source plugin (future)
```

## 8. 关键接口和数据语义

### 8.1 `PromptCommand`

核心字段：

- `text`
- `trajectory`
- `submit_time`
- `transition_mode`
- `command_id`
- `metadata`

其中 `trajectory` 当前支持两类表达，通过 `TrajectoryCondition` 的 `to_source()` 进入统一的 `TrajectorySource` 接口：

- `waypoints` — 高层来源，手工/UI 轨迹输入
- `token_aligned_traj + token_mask` — 低层兼容来源，pre-computed FloodNet token 特征

轨迹编译链：`TrajectoryCondition → TrajectorySource → CanonicalTrajectory → FloodNet adapter`，所有来源最终收敛到同一个中间表示 `CanonicalTrajectory`。

### 8.2 `HumanMotionChunk`

这是 `FloodNet` 和总控层之间的 canonical human-motion 片段。

当前要求：

- shape: `(T, 263)`
- `fps = 20`
- 带 `start_time`
- 带 `text`

### 8.3 `NMRInputChunk`

这是喂给 `MakeTrackingEasy` 的桥接结果。

当前要求：

- shape: `(T, 140)`
- `fps = 30`
- 保持时间轴连续

### 8.4 `G1ReferenceChunk`

这是最重要的执行层契约。

当前包含：

- `root_pos`
- `root_rot`
- `dof_pos`
- `local_body_pos`
- `local_body_rot`
- `body_names`
- `joint_names`

约定：

- `root_rot` 使用 `xyzw`
- `local_body_rot` 使用 `xyzw`
- `dof_pos` 已经 remap 到 `motion_tracking` 期待的 joint order

## 9. 配置文件说明

### 9.1 路径解析规则

所有路径配置（包括 `artifacts_root`、`planner.config_path`、`retarget.xml_path`、`runtime.tracking_config`）按统一规则解析：

1. 绝对路径（以 `/` 开头）原样使用
2. 相对路径统一相对于 `root_path` 解析

`root_path` 的优先级：

1. YAML 中的 `root_path` 字段（如果显式设置为非 `auto` 的路径）
2. 环境变量 `TEXT2MOTION_ROOT`
3. 自动检测（从 `paths.py` 位置向上找到包含 `FloodNet`、`MakeTrackingEasy`、`motion_tracking` 的工作区根目录）

`--config` 参数（CLI）独立解析：

- 绝对路径原样使用
- 相对路径相对于 `Text2Humanoid/` 项目根目录（而非 cwd）
- 目标：从任意 cwd 启动 API server，行为一致

### 9.2 `configs/system/`

系统级配置（这是唯一运行时真正加载的配置入口）：

- [demo_socket.yaml](./configs/system/demo_socket.yaml)
  - **默认推荐主线** — socket bridge 在线 runtime
  - `runtime.backend: socket`，与 `motion_tracking` 侧 `socket_floodnet` 配套
- [demo_fixed.yaml](./configs/system/demo_fixed.yaml)
  - **fallback / debug 路径** — file-based (`floodnet_file` backend)
  - 适合离线调试和文件系统消费
- [local_dev.yaml](./configs/system/local_dev.yaml)
  - 本机开发用
  - `127.0.0.1:8080`
  - planner chunk 比较保守
- [sim2sim_single_gpu.yaml](./configs/system/sim2sim_single_gpu.yaml)
  - 单机单 GPU 在线运行配置
  - 更大的 chunk 和更高的 buffer watermark
- [debug_replay.yaml](./configs/system/debug_replay.yaml)
  - 调试模式
  - 更小 horizon
  - `retarget.apply_filter = false`
  - 单独的调试 artifacts 目录

关键字段：

| 字段 | 说明 |
|------|------|
| `root_path` | 工作区根路径，`auto` 表示自动检测 |
| `artifacts_root` | artifact 输出目录（相对 root_path） |
| `host` / `port` | API 服务监听地址 |
| `planner.config_path` | FloodNet 模型配置路径（相对 root_path） |
| `planner.chunk_frames` | 每次生成的帧数 |
| `retarget.apply_filter` | 是否启用 Butterworth 低通滤波 |
| `retarget.tgt_fps` | 输出目标 FPS（传给 MakeTrackingEasy） |
| `retarget.xml_path` | G1 运动学模型 XML 路径（相对 root_path） |
| `runtime.tracking_config` | motion_tracking tracking.yaml 路径（相对 root_path） |
| `runtime.control_hz` | 控制频率 |
| `runtime.low_watermark_frames` | buffer 低水位（低于此值触发 DEGRADED） |
| `runtime.high_watermark_frames` | buffer 高水位（高于此值恢复 RUNNING） |
| `runtime.future_horizon_frames` | 前瞻帧数 |

### 9.3 `configs/floodnet/planner.yaml`

FloodNet planner 模板配置（字段已整合到 system config 中）。

### 9.4 `configs/nmr/retarget.yaml`

NMR retarget 模板配置（字段已整合到 system config 中）。

### 9.5 `configs/runtime/motion_tracking.yaml`

Runtime 模板配置（字段已整合到 system config 中）。

## 10. 命令行入口

### 10.1 启动 API 服务

入口：

- [apps/api_server.py](./apps/api_server.py)

示例：

```bash
# 在任何目录下启动（推荐）
PYTHONPATH=/path/to/Text2Humanoid/src \
  python /path/to/Text2Humanoid/apps/api_server.py \
  --config /path/to/Text2Humanoid/configs/system/local_dev.yaml

# 或设置环境变量指定工作区
TEXT2MOTION_ROOT=/path/to/workspace \
  PYTHONPATH=/path/to/Text2Humanoid/src \
  python /path/to/Text2Humanoid/apps/api_server.py \
  --config /path/to/Text2Humanoid/configs/system/local_dev.yaml
```

这个脚本会：

- 解析 `--config` 路径（独立于 cwd）
- 读取系统配置，解析 `root_path`
- 创建 `ArtifactStore`
- 创建 planner / retarget / adapter / runtime / fallback / coordinator / session manager
- 启动 FastAPI + Uvicorn

### 10.2 创建 session 并发送一条命令

入口：

- [apps/launch_session.py](./apps/launch_session.py)

示例：

```bash
cd /path/to/Text2Humanoid
PYTHONPATH=src python apps/launch_session.py --text "walk forward slowly"
```

这个脚本会：

- 调 `POST /sessions`
- 再调 `POST /sessions/{id}/commands`
- 打印 session id

### 10.3 查看已保存的 reference NPZ

入口：

- [apps/replay_reference.py](./apps/replay_reference.py)

示例：

```bash
cd /path/to/Text2Humanoid
python apps/replay_reference.py path/to/reference_chunk.npz
```

### 10.4 创建 artifact 目录

入口：

- [apps/export_debug_bundle.py](./apps/export_debug_bundle.py)

这个脚本目前比较简单，主要是 artifact 层的辅助入口。

### 10.5 离线 reference replay

入口：

- [apps/replay_trajectory.py](./apps/replay_trajectory.py)

**这是 runtime 接入前的离线检查里程碑，不是真实 sim2sim 已接通。**

示例：

```bash
PYTHONPATH=src python apps/replay_trajectory.py \
  --config configs/system/local_dev.yaml \
  --text "walk forward slowly" \
  --waypoints '[[0,0,0,0],[2,3,0,4]]' \
  --replay-id demo_walk
```

该脚本会走完整 `FloodNet → bridge → MakeTrackingEasy → G1ReferenceChunk` 链路，
并将所有中间结果导出为 inspectable artifact bundle。导出的 bundle 包含：

- `command.json` — 输入命令与轨迹
- `human_chunk.npz` — FloodNet 生成的 human motion (263D)
- `reference_chunk.npz` — retarget 后的 G1 reference (root/dof/local body)
- `metadata.json` — 形状、名字、耗时等诊断信息

## 11. HTTP / WebSocket 用法

### 11.1 创建 session

```bash
curl -X POST http://127.0.0.1:8080/sessions
```

返回：

```json
{"session_id":"..."}
```

### 11.2 推送命令

```bash
curl -X POST http://127.0.0.1:8080/sessions/<session_id>/commands \
  -H "Content-Type: application/json" \
  -d '{
    "text": "walk forward slowly",
    "trajectory": {
      "waypoints": [
        {"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
        {"t": 1.0, "x": 1.0, "y": 0.0, "z": 0.0}
      ]
    },
    "transition_mode": "append"
  }'
```

### 11.3 查看状态

```bash
curl http://127.0.0.1:8080/sessions/<session_id>/status
```

返回字段包括：

- `phase`
- `buffer_frames`
- `sim_time`
- `latest_chunk_id`
- `planner_latency_ms`
- `retarget_latency_ms`
- `runtime_latency_ms`
- `falls`
- `errors`
- `metadata`

### 11.4 导出状态 bundle

```bash
curl -X POST http://127.0.0.1:8080/sessions/<session_id>/export
```

当前会在 artifact 根目录下生成 `status.json`。

### 11.5 连接 WebSocket

```text
ws://127.0.0.1:8080/ws/<session_id>
```

当前连接建立后会先收到一条状态消息。后续 command / reset / stop 操作会收到广播事件。

## 12. Artifact 约定

当前 artifact 由 [artifact_store.py](./src/text2humanoid/infra/artifact_store.py) 管理。

目录结构：

```text
artifacts/
└── <session_id>/
    └── status.json
```

当前能力比较基础：

- 保存 JSON
- 保存 NPZ
- 导出状态 bundle

建议后续扩展为：

- `commands.jsonl`
- `human_chunk_*.npz`
- `nmr_input_*.npz`
- `reference_chunk_*.npz`
- `runtime_status.jsonl`
- `preview.mp4`

## 13. 测试

当前测试位于 [tests](./tests)。

覆盖内容：

- 契约对象
- `263D -> 140D` 桥接
- reference adapter
- reference buffer
- trajectory source / canonical trajectory
- 离线 replay bundle
- file backend / floodnet source
- planner-native streaming
- fixed text + fixed trajectory demo smoke

运行命令：

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

当前测试通过状态：

- `142 passed`

## 14. 依赖与安装

`Text2Humanoid` 自身在 [pyproject.toml](./pyproject.toml) 中声明的直接依赖较少：

- `fastapi`
- `uvicorn`
- `pydantic`
- `numpy`
- `PyYAML`

但真正运行时还依赖外部 repo 已经具备可用环境，尤其是：

- `FloodNet` 的模型加载依赖
- `MakeTrackingEasy` 的推理依赖
- `motion_tracking` 的配置和未来 runtime 依赖

建议当前阶段的使用方式是：

- 不急着把所有环境重新统一打包
- 先在你现有的工作环境中运行
- 等真实 `motion_tracking` source plugin 接上后，再决定最终环境管理策略

## 15. Demo Runbook

### 15.1 默认主线：Socket Bridge（推荐）

- 在线 TCP bridge，Text2Humanoid `SocketBackend` → motion_tracking `SocketFloodNetSource`
- 不需要文件系统中转
- 单机单 GPU

**配对配置：**
- Text2Humanoid 侧：[demo_socket.yaml](./configs/system/demo_socket.yaml) — `runtime.backend: socket`
- motion_tracking 侧：[tracking_socket_floodnet.yaml](../motion_tracking/sim2real/config/tracking_socket_floodnet.yaml) — `motion_source: socket_floodnet`

**最小启动顺序：**

```bash
# Terminal 1: 启动 Text2Humanoid API server
cd /path/to/Text2Humanoid
PYTHONPATH=src python apps/api_server.py --config configs/system/demo_socket.yaml

# Terminal 2: 创建 session 并推送命令
curl -X POST http://127.0.0.1:8080/sessions
# → {"session_id": "abc123..."}

curl -X POST http://127.0.0.1:8080/sessions/abc123/commands \
  -H "Content-Type: application/json" \
  -d '{
    "text": "walk forward slowly",
    "trajectory": {
      "waypoints": [
        {"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
        {"t": 5.0, "x": 5.0, "y": 0.0, "z": 0.0}
      ]
    }
  }'

# Terminal 3: 启动 motion_tracking runtime（socket consumer）
cd /path/to/motion_tracking
PYTHONPATH=sim2real/src python sim2real/src/deploy.py --sim2sim \
  --tracking-config config/tracking_socket_floodnet.yaml
```

### 15.2 Fallback / Debug 路径：File-based

- 写 NPZ 到文件系统 → motion_tracking `FloodNetMotionSource` 轮询消费
- 适合离线调试、单步验证、不需要在线 bridge 的场景

**配对配置：**
- Text2Humanoid 侧：[demo_fixed.yaml](./configs/system/demo_fixed.yaml) — `runtime.backend: floodnet_file`
- motion_tracking 侧：[tracking_floodnet.yaml](../motion_tracking/sim2real/config/tracking_floodnet.yaml) — `motion_source: floodnet`

**最小启动顺序：**

```bash
# Terminal 1: 启动 Text2Humanoid API server
cd /path/to/Text2Humanoid
PYTHONPATH=src python apps/api_server.py --config configs/system/demo_fixed.yaml

# Terminal 2: 创建 session 并推送命令
curl -X POST http://127.0.0.1:8080/sessions
# → {"session_id": "abc123..."}

curl -X POST http://127.0.0.1:8080/sessions/abc123/commands \
  -H "Content-Type: application/json" \
  -d '{
    "text": "walk forward slowly",
    "trajectory": {
      "waypoints": [
        {"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
        {"t": 5.0, "x": 5.0, "y": 0.0, "z": 0.0}
      ]
    }
  }'

# 启动后台 refill（自动持续补 chunk）
curl -X POST http://127.0.0.1:8080/sessions/abc123/refill/start

# Terminal 3: 启动 motion_tracking runtime（file consumer）
cd /path/to/motion_tracking
uv run src/deploy.py --sim2sim \
  --tracking-config config/tracking_floodnet.yaml
```

### 15.3 最小验证步骤

**Socket 主线验证：**

1. 确认 `motion_tracking` 终端日志中出现 `[SocketFloodNetSource] Connected`。
2. 检查 session 状态：
   ```bash
   curl http://127.0.0.1:8080/sessions/abc123/status
   # → phase: running, buffer_frames > 0
   ```
3. 停止 session 后验证状态：
   ```bash
   curl -X POST http://127.0.0.1:8080/sessions/abc123/stop
   ```

**File fallback 验证：**

1. 检查 session 输出目录是否有多个 chunk：
   ```bash
   ls artifacts/demo/floodnet_clips/abc123/chunk_*.npz
   ```
2. 检查 chunk_index.json：
   ```bash
   cat artifacts/demo/floodnet_clips/abc123/chunk_index.json
   ```
3. 检查 stream 生命周期：
   ```bash
   cat artifacts/demo/floodnet_clips/abc123/stream_status.json
   # → {"phase": "running"}
   ```
4. 停止 session 后验证 `done`：
   ```bash
   curl -X POST http://127.0.0.1:8080/sessions/abc123/stop
   cat artifacts/demo/floodnet_clips/abc123/stream_status.json
   # → {"phase": "done"}
   ```

### 15.4 配置说明

**Socket 主线** [demo_socket.yaml](./configs/system/demo_socket.yaml)：

| 参数 | 值 | 说明 |
|------|-----|------|
| `planner.chunk_frames` | 60 | 每次生成 ~3s 动作 |
| `runtime.backend` | `socket` | TCP 在线发送 |
| `runtime.socket_port` | 15555 | TCP 监听端口 |
| `runtime.high_watermark_frames` | 80 | buffer 高水位 |

**File fallback** [demo_fixed.yaml](./configs/system/demo_fixed.yaml)：

| 参数 | 值 | 说明 |
|------|-----|------|
| `planner.chunk_frames` | 60 | 每次生成 ~3s 动作 |
| `runtime.backend` | `floodnet_file` | 写 NPZ 到文件系统 |
| `runtime.floodnet_output_dir` | `./artifacts/demo/floodnet_clips` | NPZ 输出目录 |
| `runtime.low_watermark_frames` | 20 | buffer 低于此触发 refill |

---

## 16. 已知难点和风险

这是后续开发时必须持续记住的风险列表。

### 16.1 最大算法风险：`263D -> 140D`

这不是简单字段映射，而是跨动作表示域的桥接。即使代码正确，也可能语义不对、分布不对、效果不稳。

### 16.2 `G1ReferenceChunk` 语义必须和 `motion_tracking` 数据集一致

如果 DOF 顺序、local body 定义、quaternion 顺序、body_names 有任何偏差，tracker 可能“能跑”，但行为会明显劣化。

### 16.3 在线切换文本 + 轨迹是高风险动作

即使上游生成正常，hard switch 也容易导致下游 reference 不连续，所以 buffer 的 overlap / cross-fade 很关键。

### 16.4 单机单 GPU 资源争用

后续一旦接真 runtime，上游 planner 和下游仿真/推理竞争资源会成为时延尖峰来源。

### 16.5 在线 runtime bridge 已接通，socket 已是默认 demo 主线

- 默认 demo 已切换到 socket bridge 主线（`demo_socket.yaml` + `tracking_socket_floodnet.yaml`）
- file-based 路径（`demo_fixed.yaml` + `tracking_floodnet.yaml`）继续保留为 fallback / debug
- 两条路径在配置、文档和 smoke 层面已对齐

当前剩余风险不再是“哪条路径是主线”，而是：

- TCP socket 桥接在极端网络条件下尚未压测
- 在线 bridge 的长稳运行尚未经过长时间考验
- consumer 侧 `timeout / disconnect` 语义已经有最小真实 smoke，但还没有经过更长时间压力验证

## 17. 下一步建议

当前里程碑：稳定的多次在线 command / trajectory 序列最小语义已闭环，最小平滑 `CROSSFADE` 路径也已经接通。运行中的 session 已能连续接受三条及以上 command，并在 crossfade 路径上暴露最小 overlap 窗口观测。runtime 主线方面，socket bridge 已提升为默认 demo 主线：`demo_socket.yaml` + `tracking_socket_floodnet.yaml`，同时 `demo_fixed.yaml` + `tracking_floodnet.yaml` 保留为 file-based fallback。

下一步推荐顺序：

1. 启动独立的三段式 pipeline integration testing
2. 先落地 `Stage A: 文本 + 轨迹 -> FloodNet 输出验证`
3. 保持现有 source contract 和两条 runtime 路径稳定
4. 再按 `Stage B -> Stage C` 逐段推进

不要在刚完成 socket 主线提升后立刻跳进”大而全”的真实仿真测试。

## 18. 与 `motion_tracking` 的 patch 说明

相关文档在：

- [floodnet_source_spec.md](./patches/motion_tracking/floodnet_source_spec.md)
- [expected_runtime_patch.md](./patches/motion_tracking/expected_runtime_patch.md)

这两份文档的角色是：

- 明确未来 `motion_tracking` 应该暴露什么 source 接口
- 约束 `Text2Humanoid` 和 runtime 之间的桥接边界

真正执行时，建议只对 `motion_tracking` 做最小 patch，而不是把它整仓吸收到这里。

## 19. 一句话总结

`Text2Humanoid` 的本质不是第四个模型仓库，而是一个把 `FloodNet`、`MakeTrackingEasy`、`motion_tracking` 三段系统稳定串起来的在线编排层。当前版本已经把多次在线切换、平滑 crossfade、最小在线 runtime bridge、socket 默认 demo 主线，以及 socket 生命周期边界的最小 smoke 跑通了，file-based fallback 路径继续可用。下一步的主战场不再是继续补局部 runtime 细节，而是正式启动分阶段 pipeline integration testing，并先从 Stage A 开始。
