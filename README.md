# Text2Humanoid

Text2Humanoid 是一个用于文本驱动人形机器人动作生成的研究 demo 总控工程。

当前项目已经跑通以下主线：

```text
FloodDiffusion
  -> MakeTrackingEasy
  -> BFM-Zero tracking_online
  -> MuJoCo G1
```

当前有两条可用路径：

```text
稳定验证路径:
  text -> full motion artifact -> retarget -> BFM-Zero chunk -> replay

实验流式路径:
  text stream -> 263D chunks -> retarget windows -> BFM-Zero paced frame buffer
```

从系统目标上看，Text2Humanoid 不是“生成一个完整 motion 文件然后播放”的离线工具，而是一个面向在线 demo 的流式 orchestration layer。当前 streaming 已经打通为 research prototype，但 buffer watermark、contract 命名和 retarget chunk 边界仍在收敛中：

```text
text prompt / text stream
  -> motion chunk generation
  -> chunk-level G1 retargeting
  -> runtime future buffer
  -> continuous BFM-Zero frame stream
  -> MuJoCo G1 motion tracking
```

当前部分实现仍会通过落盘 artifact 验证中间结果，例如 `motion_263.npz`、`motion_140.npy`、`bfmzero_chunk.npz`。这些离线路径主要用于 debug 和边界验证，不是最终架构目标。

当前推荐入口是本地 Web 控制台：

```text
apps/demo_console_server.py
```

---

## 1. 项目定位

Text2Humanoid 是一个 research demo / orchestration layer。

它负责：

* 管理 demo session 和运行状态；
* 接收文本 prompt 或文本更新；
* 调用 generation backend 生成 263D human motion chunk；
* 将 263D chunk 转换成 MakeTrackingEasy 需要的 140D chunk；
* 调用 MakeTrackingEasy retarget 成 Unitree G1 motion chunk；
* 将 G1 motion chunk 转换成 BFM-Zero `tracking_online` 可消费的 ZMQ frame stream；
* 维护 runtime future buffer，尽量保证下游连续播放；
* 启动和控制 MuJoCo simulation / BFM-Zero policy；
* 保存中间产物、诊断文件和 debug bundle。

它不负责：

* 训练 FloodDiffusion；
* 训练 FloodNet；
* 训练 MakeTrackingEasy；
* 训练 BFM-Zero；
* 重写机器人控制策略；
* 做 sim2real；
* 做生产级实时机器人系统。

当前阶段的目标是：让研究伙伴能够理解并复现完整数据链路，使用 Web Console 跑通流式 demo，并方便后续继续改 generation、retarget、runtime 或 demo orchestration。

---

## 2. 当前主线

当前已经跑通的主线是：

```text
Text prompt
  -> FloodDiffusion
  -> HumanML3D / 263D motion chunk
  -> 263D to 140D bridge
  -> MakeTrackingEasy
  -> G1 root + 29 DoF motion chunk
  -> BFM-Zero future buffer
  -> BFM-Zero ZMQ frame stream
  -> BFM-Zero tracking_online
  -> MuJoCo G1
```

当前实现状态：

```text
offline validation:
  apps/run_text_to_bfmzero.py
  apps/replay_bfmzero_chunk.py
  apps/launch_full_demo.py

streaming prototype:
  FloodDiffusionStreamingBackend
  StreamingTextToBFMZeroRunner
  StreamingRetargetBridge
  StreamingBFMZeroPublisher / BFMZeroFrameBuffer
  Demo Console Start Stream / Update Text / Stop Stream
```

当前主入口：

```text
apps/demo_console_server.py
```

当前主 runtime：

```text
BFM-Zero tracking_online + ZMQ
```

历史上保留过 `motion_tracking` socket/file backend 作为 runtime fallback，但当前主线已经切换到 BFM-Zero。`motion_tracking` 相关代码不再作为推荐路径维护。

未来计划中，FloodNet 会作为新的 generation backend 接入，用来替换当前的 FloodDiffusion backend。只要它继续输出同样语义的 `humanml3d_263` chunk，下游 retarget 和 runtime 不需要重写。

---

## 3. 总体数据链路

Text2Humanoid 的目标数据链路按 chunk 流动，而不是按完整 motion 文件流动。当前代码中主要使用 `GeneratedMotion`、`NMRInputChunk`、`RobotMotion` 和 `BFMZeroMotionChunk`；文档里的 `*Chunk` 命名表示后续 contract 收敛方向：

```text
Text Prompt / Text Stream
  |
  v
Demo Console / Session Orchestrator
  |
  v
GenerationBackend.stream_chunks()
  当前实现: FloodDiffusion / FloodDiffusion streaming wrapper
  未来实现: FloodNetBackend
  |
  v
GeneratedMotion / GeneratedMotionChunk
  representation = humanml3d_263
  motion.shape   = (chunk_T, 263)
  fps            = 20
  |
  v
HumanML263ToNMR140Bridge.convert_chunk()
  |
  v
NMRInputChunk / RetargetInputChunk
  representation = nmr_smplx_140
  motion.shape   = (chunk_T, 140)
  fps            = 30
  |
  v
MakeTrackingEasyBackend.retarget_chunk()
  |
  v
RobotMotion / RobotMotionChunk
  robot          = unitree_g1
  root + 29 DoF
  root_quat      = wxyz
  fps            = 30
  |
  v
G1MotionToBFMZeroInputBridge.convert_chunk()
  |
  v
BFMZeroMotionChunk
  joint_pos / joint_vel
  root_pos / root_quat
  root_lin_vel_w / root_ang_vel_w
  fps            = 50
  |
  v
Runtime Future Buffer
  |
  v
Streaming Publisher / BFMZeroZmqSink
  |
  v
BFM-Zero tracking_online
  |
  v
MuJoCo G1
```

几个关键数据表示：

| 阶段            | 数据对象               | 表示               | shape / 内容            | FPS |
| --------------- | ---------------------- | ------------------ | ----------------------- | --- |
| generation 输出 | `GeneratedMotion`      | `humanml3d_263`    | `(chunk_T, 263)`        | 20  |
| retarget 输入   | `NMRInputChunk`        | `nmr_smplx_140`    | `(chunk_T, 140)`        | 30  |
| robot motion    | `RobotMotion`          | `g1_root_dof`      | root + 29 DoF           | 30  |
| BFM-Zero 输入   | `BFMZeroMotionChunk`   | frame stream chunk | root + joint + velocity | 50  |

其中：

* `263D` 是 FloodDiffusion / HumanML3D 风格的人体动作表示；
* `140D` 是 MakeTrackingEasy / NMR retarget model 需要的 SMPL-X motion input；
* `29 DoF` 是 Unitree G1 的关节自由度；
* `50 FPS` 是推给 BFM-Zero `tracking_online` 前的 runtime 频率；
* `chunk` 是 generation、retarget 和 runtime 之间的最小流式处理单元；
* `future buffer` 是 runtime 侧维护的未来帧缓存，用来减少播放断流。

---

## 4. 如何启动 Demo Console

当前推荐从 Web Demo Console 启动：

```bash
cd /home/lai/大学/text2motion/Text2Humanoid
PYTHONPATH=src python3 apps/demo_console_server.py --host 127.0.0.1 --port 8090
```

浏览器打开：

```text
http://127.0.0.1:8090/
```

Demo Console 主要提供两组功能。

Simulation / Policy 控制：

```text
Start Sim      # 启动 MuJoCo sim 和 BFM-Zero policy
Put Down       # 发送 sim key 9
Init Robot     # 发送 policy key i
Start Motion   # 发送 policy key [
Enable Policy  # 发送 policy key ]
Stop App       # 清理 demo 相关进程
```

Streaming Motion：

```text
输入 prompt
Start Stream
Update Text
Stop Stream
查看 streaming metrics
查看 Live FloodDiffusion 263D Preview
```

离线验证入口仍保留在 CLI 中：

```bash
PYTHONPATH=src python3 apps/run_text_to_bfmzero.py \
  --text "walk forward" \
  --out-prefix assets/saved/demo_walk \
  --frames 300 \
  --generation-steps 150 \
  --dry-run
```

如需渲染中间态视频，额外加：

```text
--inspect
```

Web streaming 路径对应的是：

```text
准备 session
持续生成 263D chunks
按窗口 retarget chunk
append 到 BFM-Zero paced frame buffer
以 50Hz 推送给 BFM-Zero
```

如果需要清理 demo 相关进程，可以运行：

```bash
PYTHONPATH=src python3 apps/app_stop.py
```

---

## 5. 代码结构

仓库一级结构大致如下：

```text
Text2Humanoid/
├── apps/        # 稳定运行入口
├── configs/     # 系统、generation、retarget、runtime 配置
├── patches/     # 外部项目 patch / spec 说明
├── src/         # 核心源码
├── tests/       # 测试
├── tools/       # 诊断、可视化、replay 工具
├── README.md
└── pyproject.toml
```

`src/text2humanoid/` 是核心代码：

```text
src/text2humanoid/
├── contracts/      # 跨模块数据契约
├── generation/     # 文本到 263D motion chunk
├── retarget/       # 263D -> 140D -> G1 retarget
├── runtime/        # BFM-Zero 协议、future buffer、ZMQ 推流
├── orchestrator/   # pipeline / session / chunk 调度
├── demo/           # Web console 和进程管理
├── infra/          # 路径、配置、日志、artifact、进程管理
├── api/            # HTTP / WebSocket API
├── visualization/  # 可视化辅助
└── interfaces.py   # backend / bridge / sink 协议接口
```

建议理解方式：

```text
contracts     定义跨模块传输什么
generation    负责从文本生成 263D motion chunk
retarget      负责从 263D chunk 得到 G1 motion chunk
runtime       负责把 G1 motion chunk 连续推给 BFM-Zero
orchestrator  负责 session 和 chunk pipeline 调度
demo          负责本地控制台和进程管理
```

---

## 6. 流式链路与 Buffer 设计

Text2Humanoid 的核心不是离线生成一个完整 motion 文件后播放，而是维护一个 session 级的 streaming chunk pipeline。

系统可以理解成多个异步生产者 / 消费者之间的 buffer 链路：

```text
Text / Prompt Updates
        |
        v
Session Orchestrator
        |
        v
Generation Request Queue
        |
        v
263D Motion Buffer
        |
        v
Retarget Queue
        |
        v
G1 Motion Buffer
        |
        v
Runtime Future Buffer
        |
        v
BFM-Zero ZMQ Sender at 50Hz
        |
        v
BFM-Zero tracking_online -> MuJoCo G1
```

其中最重要的问题是：上游生成模型输出 20 FPS motion，而下游 BFM-Zero tracking 控制按 50 FPS 消费 reference frame。二者不是通过固定帧数直接对齐，而是通过相同时间长度的 chunk 对齐。

### 6.1 核心问题：20 FPS 生成如何接 50 FPS tracking

系统中不同阶段使用不同 FPS：

```text
FloodDiffusion / generation: 20 FPS
MakeTrackingEasy retarget:   30 FPS
BFM-Zero tracking_online:    50 FPS
```

这并不表示 BFM-Zero 直接消费 20 FPS 的 generation 输出。实际链路是：

```text
GeneratedMotionChunk
  20 FPS, humanml3d_263
  duration = D seconds
        |
        | resample / convert by time
        v
RetargetInputChunk
  30 FPS, nmr_smplx_140
  duration = D seconds
        |
        | MakeTrackingEasy retarget
        v
RobotMotionChunk
  30 FPS, G1 root + dof
  duration = D seconds
        |
        | resample by time
        v
BFMZeroMotionChunk
  50 FPS, root + dof + velocities
  duration = D seconds
        |
        | send one frame every 1/50 second
        v
BFM-Zero tracking_online
```

也就是说：

```text
20 FPS generation 只决定上游动作采样率
30 FPS retarget 是 MakeTrackingEasy 的输入 / 输出工作频率
50 FPS runtime 是 BFM-Zero 的发送和控制频率
```

中间通过时间轴重采样对齐。

### 6.2 用时间长度定义 chunk，而不是用固定帧数

README 和代码设计中不建议只说 chunk_size = 20，因为“20 帧”在哪个 FPS 下含义不同。

更推荐使用：

```text
chunk_duration_sec = 1.0
generation_fps = 20
retarget_fps = 30
runtime_fps = 50
```

然后各阶段帧数由时间长度派生：

```text
generation_frames = chunk_duration_sec * 20
retarget_frames   = chunk_duration_sec * 30
runtime_frames    = chunk_duration_sec * 50
```

例如，当 chunk_duration_sec = 1.0 时：

```text
GeneratedMotionChunk: 20 frames @ 20 FPS
RetargetInputChunk:  30 frames @ 30 FPS
RobotMotionChunk:    30 frames @ 30 FPS
BFMZeroMotionChunk:  50 frames @ 50 FPS
```

它们帧数不同，但都表示同一段 1 秒的动作。

如果 chunk_duration_sec = 2.0：

```text
GeneratedMotionChunk: 40 frames @ 20 FPS
RetargetInputChunk:  60 frames @ 30 FPS
BFMZeroMotionChunk:  100 frames @ 50 FPS
```

因此，系统真正对齐的是时间，不是原始帧数。

### 6.3 单个 chunk 的数据流

以 chunk_duration_sec = 1.0 为例，一个 chunk 在系统中的流动如下：

```text
[Generation Request]
  prompt = "walk forward"
  duration = 1.0s
        |
        v
[GeneratedMotionChunk]
  representation = humanml3d_263
  shape = (20, 263)
  fps = 20
  duration = 1.0s
        |
        v
[RetargetInputChunk]
  representation = nmr_smplx_140
  shape = (30, 140)
  fps = 30
  duration = 1.0s
        |
        v
[RobotMotionChunk]
  robot = unitree_g1
  root_pos = (30, 3)
  root_quat = (30, 4)
  dof_pos = (30, 29)
  fps = 30
  duration = 1.0s
        |
        v
[BFMZeroMotionChunk]
  joint_pos = (50, 29)
  joint_vel = (50, 29)
  root_pos = (50, 3)
  root_quat = (50, 4)
  root_lin_vel_w = (50, 3)
  root_ang_vel_w = (50, 3)
  fps = 50
  duration = 1.0s
        |
        v
[Runtime Future Buffer]
  append 50 frames
        |
        v
[BFM-Zero Sender]
  send 1 frame every 20 ms
```

在这个过程中，generation / retarget / runtime 各层只关心自己的输入输出契约。

generation 不需要知道 BFM-Zero 的 joint order；retarget 不需要知道 ZMQ 发送频率；runtime 不需要知道文本 prompt 如何被编码。

### 6.4 连续播放时的 future buffer

如果每次都等一个 chunk 播放完，再生成下一个 chunk，系统很容易断流。

例如：

```text
chunk_duration = 1.0s
generation + retarget 耗时 = 3.0s
BFM-Zero 1.0s 就播完当前 chunk
后面 2.0s 没有 reference frame
```

因此，真正的 streaming 设计不是：

```text
generate chunk
-> retarget chunk
-> play chunk
-> generate next chunk
```

而应该是：

```text
runtime 播放当前 future buffer
同时 generation / retarget 准备后续 chunk
当 buffer 低于阈值时提前请求 next chunk
新 chunk 完成后 append 到 future buffer
```

runtime future buffer 可以理解成：

```text
已准备但尚未播放的 BFM-Zero frames
```

例如：

```text
t = 0.0s:
  buffer = 100 frames  # 2 秒 future motion

t = 0.5s:
  BFM-Zero 已消费 25 frames
  buffer = 75 frames

如果 low_watermark = 50 frames:
  orchestrator 触发 generation / retarget 下一个 chunk

新 chunk 完成:
  append 50 frames
  buffer 回到约 100 frames
```

BFM-Zero sender 始终以 50Hz 消费：

```text
1 frame / 20 ms
50 frames / second
```

上游模块只需要在 buffer 被耗尽之前补上新的 future frames。

### 6.5 Generation Buffer

generation 层是 263D motion chunk 的生产者。

输入：

```text
PromptCommand / GenerateRequest
```

输出：

```text
GeneratedMotionChunk
  representation = humanml3d_263
  motion.shape = (chunk_T, 263)
  fps = 20
  chunk_id
  start_time
  prompt_id
```

generation buffer 可以理解成：

```text
Prompt / text update
  -> generation request queue
  -> generated 263D chunk queue
```

它解决的问题是：

generation 可能比 playback 慢；
同一个 prompt 可能需要连续生成多个 chunk；
新 prompt 可能在旧 prompt 的 chunk 还没播放完时到来；
生成失败时需要让 session fallback / stop，而不是让 runtime 直接崩掉。

当前实现中，generate_chunk() 可以作为单 chunk 生成能力；stream_chunks() 是目标接口，用于连续产出 GeneratedMotionChunk。

离线保存 motion_263.npz 只是验证方式。真正稳定的接口是：

GenerationBackend.stream_chunks(request)
  -> Iterator[GeneratedMotionChunk]

generation 侧 buffer 不关心 BFM-Zero 的 50Hz，也不关心 G1 joint order。它只保证产出语义正确、时间戳连续的 263D chunk。

### 6.6 Retarget Buffer

retarget 层是 263D chunk 的消费者，也是 G1 motion chunk 的生产者。

输入队列：

```text
GeneratedMotionChunk queue
```

输出队列：

```text
RobotMotionChunk / BFMZeroMotionChunk queue
```

retarget buffer 的链路是：

```text
GeneratedMotionChunk(20 FPS, 263D)
  -> 263D to 140D bridge
  -> RetargetInputChunk(30 FPS, 140D)
  -> MakeTrackingEasy retarget
  -> RobotMotionChunk(30 FPS, root + 29 DoF)
```

retarget buffer 需要处理的问题：

20 FPS 到 30 FPS 的 chunk 重采样；
chunk start_time / duration 对齐；
root position 在 chunk 间连续；
root orientation 在 chunk 间连续；
MakeTrackingEasy 是否需要上下文窗口；
预测结果在 chunk 边界是否跳变；
retarget 耗时是否会导致 runtime buffer 低水位。

因此 retarget 层通常不应该只处理完全孤立的 chunk，而应该保留必要的历史上下文，例如：

```text
previous chunk tail
previous root pose
previous root yaw
previous dof tail
```

这些状态用于：

velocity 计算；
chunk 边界连续性检查；
overlap / crossfade；
root trajectory 对齐；
retarget 输出 sanity check。

retarget buffer 不负责最终按 50Hz 发送。它输出的是 G1 motion 语义正确的 robot chunk。

### 6.7 Runtime Future Buffer

runtime 层是固定频率消费者。BFM-Zero tracking_online 需要持续接收 frame stream，因此 runtime 不能等待上游临时生成。

runtime 的核心结构是 future buffer：

```text
RobotMotionChunk / BFMZeroMotionChunk
  -> append to future buffer
  -> 50Hz playhead consumes frames
  -> BFMZeroZmqSink sends MotionFrameMessage
```

future buffer 维护：

```text
frame_start
frame_idx
playhead
buffered_frames
buffered_seconds
low_watermark
end_flag
```

BFM-Zero 每帧需要：

```text
frame_idx
joint_pos
joint_vel
root_pos
root_quat
root_lin_vel_w
root_ang_vel_w
```
flags

因此进入 future buffer 前，需要完成：

G1 motion 重采样到 50 FPS；
joint order 转成 BFM-Zero / Isaac order；
root quaternion 转成 wxyz；
在 50 FPS 轨迹上重新计算 joint_vel；
在 50 FPS 轨迹上重新计算 root_lin_vel_w；
用 quaternion 差分计算 root_ang_vel_w；
生成连续 frame_idx。

future buffer 的目标是：

BFM-Zero 以固定频率消费；
上游 generation / retarget 可以异步补充；
如果 buffer 低于阈值，orchestrator 触发下一 chunk；
如果 buffer 空了，系统应该进入明确的 fallback / stop 状态，而不是发送脏数据。
### 6.8 Chunk 边界与连续性

流式系统中，单个 chunk 看起来正常，不代表连续播放正常。

需要重点检查：

time continuity:
  chunk[i].start_time + chunk[i].duration ~= chunk[i+1].start_time

root continuity:
  root_pos tail/head 是否连续
  root_yaw tail/head 是否跳变
  root_quat consecutive dot 是否接近 1

joint continuity:
  dof tail/head 是否跳变
  joint_vel 是否在边界出现尖峰

frame continuity:
  BFM-Zero frame_idx 是否连续
  50Hz 重采样后是否丢帧或重复帧

buffer continuity:
  future buffer 是否 underrun
  append chunk 时是否覆盖尚未播放的 frame

后续如果需要改善 chunk 边界，可以考虑：

overlap；
crossfade；
保留上一 chunk tail 作为上下文；
root trajectory 对齐；
yaw smoothing；
delayed replace。
### 6.9 文本切换时的 buffer 策略

文本切换不是立即替换当前播放帧，而应该通过 buffer 策略处理。

推荐语义：

current_prompt: 正在播放 / 已生成 future chunk 的文本
pending_prompt: 用户新提交但尚未生效的文本
switch_boundary: 允许切换的 chunk 边界

基础策略：

1. 当前 frame 正在由 BFM-Zero 消费，不能直接替换
2. 已经进入安全播放窗口的 buffer 不修改
3. 新 prompt 先进入 pending_prompt
4. orchestrator 在下一个允许的 chunk boundary 使用新 prompt 请求 generation
5. 新 chunk retarget 完成后 append 到 future buffer
6. 必要时做 crossfade / delayed replace

第一版可以只支持 chunk boundary 切换，不承诺无缝。

### 6.10 离线 artifact 在流式链路中的角色

虽然系统目标是 streaming，但当前工程中仍会保存中间 artifact：

motion_263.npz
motion_140.npy
mte_result.npz
bfmzero_chunk.npz
diagnostics.json
inspect videos

这些文件的作用是：

验证某个边界的数据是否正确；
重放某个 chunk；
对比不同 retarget 策略；
复现实验中的异常；
避免每次调试都重新跑重模型。

因此 README 中出现 artifact，并不表示系统目标是离线 pipeline。它们是 streaming pipeline 的 debug checkpoint。


## 7. 后续计划

短期：

* 清理 `apps/` 和 `tools/` 边界；
* 将稳定入口和开发工具分开；
* 整理 `artifacts/`、`assets/saved/`、`assets/video/` 输出目录；
* 增强 artifact metadata；
* 增加 BFM-Zero ZMQ mock subscriber test；
* 增加 RobotMotionChunk / BFMZeroMotionChunk contract test；
* 明确 chunk id、start time、frame index 的连续性约束。

中期：

* 把 FloodNet 接成新的 `GenerationBackend`；
* 保持 `GeneratedMotionChunk(humanml3d_263)` 作为下游稳定边界；
* 改进 chunk 级 streaming；
* 支持 chunk boundary 文本切换；
* 强化 retarget 诊断和 root orientation 质量控制；
* 完善 future buffer low-watermark 调度。

长期：

* 支持更稳定的在线生成；
* 支持更自然的文本切换；
* 支持 trajectory / waypoint 条件；
* 支持更干净的远程 generation backend；
* 为 sim2real 做更严格的数据契约和 runtime 检查。
