# 弹性分布式训练协调组件（本地 IPC 版）

只用 Go 标准库 + 仓库已有依赖，在**单机上用 unix domain socket 多进程**模拟一个
弹性分布式训练集群：一个协调者（coordinator）+ 若干训练 worker。
不引入任何外部服务、真实凭据或网络依赖。

核心目标：**无论怎么分片、worker 顺序如何变化、成员中途如何增减、进程如何中断恢复，
最终参数与优化器状态都和单机顺序基准（`nn.RunBaseline`）一致**，并且各种失败原因
（分片非法/对不齐、掉队 vs 死亡、半截/坏魔数/版本不符/损坏检查点）都能用稳定的
错误类型与错误码明确区分。

## 目录结构

| 包 | 职责 |
|---|---|
| `shard` | 参数/数据分片区间规划、连续性与非空保证、多 worker 方案对齐校验 |
| `nn` | 确定性合成数据集、线性模型、梯度、SGD+momentum、单机基准 |
| `collective` | 纯函数集合通信原语：reduce-scatter / allgather / allreduce（canonical rank 顺序） |
| `proto` | worker↔coordinator 的 JSON 消息定义 |
| `transport` | unix domain socket 上的换行分隔 JSON RPC，客户端断线自动重连 |
| `checkpoint` | magic+版本+长度+JSON+SHA256，临时文件 + `rename` 原子提交 |
| `coordinator` | 成员/era 管理、步骤台账、幂等去重、慢/死判定、检查点、优雅停机、RPC 服务 |
| `worker` | 与传输无关的计算核心 `Core` + 走 socket 的客户端 `Worker` |
| `cmd/etcd` | 协调者进程（elastic-training-coordinator-daemon） |
| `cmd/etcworker` | worker 进程 |

## 1. 分片与集合通信语义

### 分片（`shard.Plan`）

- 长度为 `n` 的向量切成 `c` 个**连续、非空、首尾相接、恰好覆盖**的半开区间
  `[start,end)`；余数 `n%c` 依次分给前 `r` 片（长度 `q+1`），其余为 `q`。
  这样任何合法 `1 <= c <= n` 都得到确定性的唯一切分。
- 非法分片数会被**明确拒绝**并返回可 `errors.Is` 判定的原因：
  - `shard.ErrInvalidShardCount`（`c<=0` 或 `c>n`，空片不允许）；
  - `shard.ErrMisaligned`（各 worker 的 `n`/`c` 不一致，或 rank↔worldSize 不一致）；
  - `shard.ErrRangeMismatch`（worker 声明的区间与规划中该片不一致）；
  - `shard.ErrIndexOutOfRange`（rank 越界）。
  - 用 `shard.KindOf(err)` 可拿到稳定的枚举原因。

### 集合通信（`collective`）

- `AllReduce`：等价 reduce-scatter + allgather，每 rank 得到完整求和向量。
- `ReduceScatter`：归约后每 rank 只拿自己负责区间的片段。
- `AllGather`：把各 rank 的片段按规划拼回完整向量。
- **canonical 顺序**：归约一律按 rank 升序 `0..world-1` 累加，与请求到达顺序无关；
  rank 按 worker ID 字典序分配，因此**worker 启动/提交顺序任意变化，浮点结果位级一致**。
- 组校验规则（全部有独立错误原因）：组非空（`ErrEmptyGroup`）、worldSize 合法
  （`ErrInvalidShardCount`）、成员数==worldSize 且 rank 不重复/不越界
  （`ErrRankMismatch`）、各成员区间与统一规划逐片一致且数据长度匹配
  （`ErrMisaligned`）。

### 训练步骤实际协议（coordinator 内实现）

每个 batch 的样本也按相同规划切给各 worker。一步分两个阶段：

1. **reduce-scatter**：每个 worker 用协调者下发的**全量当前参数**，在自己的数据
   分片上计算**完整局部梯度向量**，但按**整个 batch 的大小**归一化
   （`nn.Model.GradientN(batch, batchSize)`）。各 worker 的局部梯度按 canonical
   顺序求和后，按参数分片切给对应 rank。
   - 子 batch 梯度按全 batch 归一化是和单机对齐的关键：
     `(2/B)Σ_A + (2/B)Σ_B = (2/B)Σ_{A∪B}`，与单机整 batch 梯度逐项相等。
2. **allgather**：每个 rank 只在自己负责的参数片上做 SGD+momentum
   `v = m*v + g; p -= lr*v`，把更新后的**参数片和 velocity 片**回传；协调者按
   canonical 顺序拼回全量参数/velocity，**一次性原子提交**这一步并推进 step。

> 优化器 velocity 由协调者作为唯一权威保存并随 reduce 结果/检查点流转，worker 不
> 自持 momentum，因此 worker 重连或被重新分到不同片都不会用旧 momentum 污染结果。

## 2. 弹性成员与故障处理

- **成员与 rank**：worker 用稳定 ID `join`；rank = 当前成员 ID 字典序下标。
- **era（时代）**：每次成员集合变化（加入/退出/判死）`era += 1`。任何在途步骤的
  旧 era 消息一律以 `ErrStaleEra`（RPC 错误码 `stale_era`）拒绝，**整步作废、用
  当前参数在新 era 重新执行同一步**。因为作废发生在提交之前，所以：
  - 不会把部分 worker 的梯度写进参数（**不丢梯度、不部分提交**）；
  - 重开后所有梯度只算一次（**不重复累加**）。
- **分片重分配**：新 era 的下一步按新 worldSize 重新出 `shard.Plan`，样本片和
  参数片同时重切。
- **幂等去重（含完成后重连）**：协调者对每个 `(era,step,rank)` 记录是否已收到
  reduce-scatter / allgather 贡献。worker 重发同一贡献（网络重试、断线重连、甚至
  全部训练完成后又重连）只会拿回缓存结果或 `Done`，**绝不产生第二次贡献**。
  `join` 本身也幂等：已存在的 ID 重连只刷新心跳，不推进 era。
- **掉队 vs 死亡可区分**（`Coordinator.CheckTimeouts`）：
  - **死亡（DEAD）**：心跳超过 `heartbeat-timeout` → 摘除成员、推进 era、在途步作废。
  - **掉队（SLOW/straggler）**：心跳仍新鲜（在心跳超时内）但当前步骤等待超过
    `step-timeout` → 仅记录为慢节点，**不摘成员、不丢步骤**，等其追上。
  - 两者在事件日志与返回值中分别给出原因。常驻进程内还有看门狗周期性做心跳判死。
- worker 客户端在连接中断或收到 `stale_era` 时自动退避重连、重新 `join`/`next`。

## 3. 检查点与恢复

- 协调者周期性（`-ckpt-every`，也可由 worker 显式请求）把
  **参数 + 优化器 velocity + 成员列表 + era + step** 落盘。
- 文件布局：`magic(8)="ETCKPT01" | version(uint32 BE) | payloadLen(uint32 BE)
  | JSON(payload) | SHA256(payload)`，文件名 `ckpt-s<step>-e<era>.etc`。
- **原子写**：先写同名 `.tmp`、`fsync`，再 `rename` 到正式文件并 `fsync` 目录。
  崩溃只会留下 `.tmp` 或一个未写全的正式文件，绝不会出现“看似有效”的检查点。
- 启动时读取**最高 (step,era)** 的正式检查点，并按下列顺序给出**可区分**的失败
  （`errors.Is` + `checkpoint.KindOf`）：

  | 现象 | 错误 |
  |---|---|
  | 目录里没有任何检查点 | `ErrMissing`（视为全新启动） |
  | 只有遗留 `.tmp` / 头或载荷被截断 | `ErrPartialWrite`（半截写入） |
  | 头完整但魔数不是 `ETCKPT01` | `ErrBadMagic`（不是本组件文件） |
  | 魔数对但版本号不受支持 | `ErrVersionMismatch` |
  | 长度完整但 SHA256 不符 / JSON 无法解析 / 维度不符 | `ErrCorrupt` |

  除“完全没有检查点”外，**任何半截/损坏/版本不符都不会被静默当成全新启动，也不会
  被当成有效检查点继续**——`coordinator.New` 直接返回对应错误。
- 恢复后从检查点的 `step` 继续（已完成的步不重算），worker 用相同 ID 重新 join，
  最终结果与不中断一次性训练**位级一致**（有测试断言对齐单机基准）。

### 保留上限（`-ckpt-keep`，默认 5）

- 每次落盘后自动清理，只保留**最新 N 份**正式检查点（`coordinator.Config.CheckpointKeep`，
  `<=0` 表示不自动清理）。
- 不变量：**最新有效点永不被清理**（`keep` 会被钳制到 `>=1`），因此清理后重启
  一定能从最新点恢复，绝不会因为清理让下次启动看起来像全新作业。
- 只清理正式文件；遗留的 `.tmp` 半截文件保留，恢复时仍按上表规则判定。

### 回滚与回退

- **显式回滚**（`-resume-from <path>` / `Config.ResumeFrom`）：从指定的某个更早的
  有效检查点重新往下跑。指定文件**已被清理/缺失 → `ErrMissing`，半截 →
  `ErrPartialWrite`，损坏 → `ErrCorrupt`，版本不符 → `ErrVersionMismatch`**，
  全部明确拒绝，绝不静默从头开始。回滚后继续训练的结果同样对齐单机基准（有测试）。
- **明确回退**（`-resume-fallback` / `Config.ResumeFallback`，默认关闭）：最新检查点
  不可用时，从最新到最旧依次尝试，**明确采用第一个仍然可用的点**，并在事件日志里
  逐条记录被跳过的检查点及原因；一个都不可用则返回最相关的可区分错误。
  默认（严格模式）仍是直接拒绝最新的坏点。
- **兼容范围**：当前格式版本为 `checkpoint.FormatVersion = 1`；魔数 `ETCKPT01` 且
  版本为 1 的历史检查点都可以恢复/回滚，版本不符的一律拒绝（或按回退规则跳过）。

## 4. 优雅停机与续训

协调者是一个可长期挂在本地、随时能停、停了能接着跑的作业：

- **两条退出路径，殊途同归**：
  - 跑满 `-steps` 目标步数（`ExitCompleted`）；
  - 被外部叫停（`SIGINT`/`SIGTERM`，或任何客户端发 `stop` RPC /
    `Coordinator.RequestStop`）（`ExitStopped`）。
- **停机语义**：进入停机流程后立即**不再接收新工作**——新 worker 拒绝 join，
  `next` 返回 `stopped`，在途的 `submit`/`allgather` 以稳定错误码
  `shutting_down` 拒绝。随后在**最后一个完整提交的步**上落一份最终检查点，
  关闭监听、断开连接、回收看门狗 goroutine，在有限时间内自行退出。
- **绝无半提交**：检查点只包含已完整提交的步；停机哪怕发生在某一步集合通信
  正做到一半，那一步的在途台账被整体丢弃，已收到的部分贡献**绝不写入**检查点，
  重启后该步从头重做（不丢步、不重复累加）。
- **如实报告**：进程退出前打印 `exit report: reason=completed|stopped step=N/M
  checkpoint=<path>`，明确这次是跑满了还是被中途停下；`Coordinator.ExitReport()`
  以结构化形式提供同一份信息。
- **立即重启续训**：退出会释放 socket 文件等本地资源，用**同一套参数、同一条
  socket 路径**即可马上重新拉起；worker 端默认自动重连（收到 `shutting_down` 或
  连接中断后退避重试），协调者重启后接着检查点的步继续跑。
  反复多轮“跑一段、停、重启”最终参数/优化器状态仍与不间断单机基准位级一致（有测试）。

## 5. 本地怎么验证

需要 Go 1.25+。不需要联网、不需要任何外部服务。

```bash
# 全部单元 + 集成 + 真实 socket 端到端测试（确定性，带成员/分片区间/判定日志）
go test ./... -v

# 竞态检测
go test -race ./...

# 构建两个进程
go build -o /tmp/etcd      ./cmd/etcd
go build -o /tmp/etcworker ./cmd/etcworker

# 终端 1：启动协调者（训练完成后自动退出）
rm -rf /tmp/etc-demo && mkdir -p /tmp/etc-demo
/tmp/etcd -socket /tmp/etc-demo/e.sock -ckpt-dir /tmp/etc-demo/ckpt \
  -steps 6 -batch 8 -nin 2 -samples 32 -ckpt-every 2

# 终端 2..N：启动若干 worker（可随时增删；演示分片数变化）
/tmp/etcworker -socket /tmp/etc-demo/e.sock -id wa
/tmp/etcworker -socket /tmp/etc-demo/e.sock -id wb
# 稍后再起一个，会触发 era 推进并从下一步起按 3 片重分配
/tmp/etcworker -socket /tmp/etc-demo/e.sock -id wc
```

协调者日志会持续打印：当前成员列表与 rank、每步样本/参数分片区间、reduce-scatter
与 allgather 的 canonical 完成与提交、检查点路径、以及慢/死/陈旧 era 等判定依据。

中断恢复演示：

```bash
# 跑到一半 kill -9 掉协调者，再用同一 ckpt 目录重启（worker 会自动重连接着跑）
/tmp/etcd -socket /tmp/etc-demo/e.sock -ckpt-dir /tmp/etc-demo/ckpt -steps 6 ...
# 想忽略旧检查点全新启动：加 -no-resume
```

优雅停机与续训演示：

```bash
# 终端 1 的协调者跑到一半时，给它发 SIGTERM（或 Ctrl-C）：
kill -TERM <etcd-pid>
# 日志会打印：stop requested -> 在最后一个完整提交的步上落最终检查点 ->
# exit report: reason=stopped step=N/6 checkpoint=...，随后在有限时间内自行退出。

# 用同一套参数、同一条 socket 路径立刻重新拉起，从停机点继续：
/tmp/etcd -socket /tmp/etc-demo/e.sock -ckpt-dir /tmp/etc-demo/ckpt -steps 6 ...
# 跑满后日志打印 exit report: reason=completed step=6/6 ...

# 检查点保留上限（默认 5）：目录里只留最新 5 份，最新有效点永不删除
/tmp/etcd ... -ckpt-keep 3

# 回滚到某个更早的有效检查点重新往下跑（文件缺失/损坏会被明确拒绝）：
/tmp/etcd ... -resume-from /tmp/etc-demo/ckpt/ckpt-s0000002-e0000003.etc

# 最新检查点损坏时，明确回退到仍然可用的更早点（日志记录每个被跳过的点）：
/tmp/etcd ... -resume-fallback
```

## 6. 测试覆盖一览

- `shard_test.go`：非法分片数、连续非空覆盖、切分/区间校验、多 worker 方案对不齐。
- `collective_test.go`：1/2/3/5 片的 allreduce 与 canonical 求和、打乱到达顺序、
  reduce-scatter+allgather 往返、空组/非法 worldSize/rank 重复/区间对不齐/长度不符。
- `nn_test.go`：数据集与基准的确定性、梯度维度。
- `checkpoint_test.go`：往返、选最新、缺失、半截 `.tmp`、截断正式文件、坏魔数、
  版本不符、校验和损坏、有效检查点优先于遗留 `.tmp`。
- `checkpoint/retention_test.go`：保留上限只留最新 N 份、最新有效点永不删除、
  被清理的点显式加载报 `ErrMissing`、`LoadUsable` 明确回退/全坏时报最新原因/
  空目录与只有 `.tmp` 的区分。
- `coordinator/elastic_test.go`：
  - 1/2/3 片且每步打乱 worker 顺序，结果对齐单机基准；
  - 训练中途加人/减人后分片重分配继续跑，对齐基准；
  - 集合通信中途掉节点，部分贡献随 era 作废、旧贡献被 `stale_era` 拒绝、补员后整步重做；
  - 完成后重连幂等，不产生重复贡献；
  - 慢节点（心跳新鲜）与死节点（心跳超时）被明确区分，慢节点不被摘除。
- `coordinator/recovery_test.go`：跑到一半用检查点重启接着跑并对齐基准；半截/坏魔数/
  版本不符/校验和损坏/维度不符在重启时分别被拒绝。
- `coordinator/lifecycle_test.go`：
  - 集合通信中途（reduce-scatter 只到 1/2）停机：在途步不半提交，最终检查点停在
    最后完整提交的步，重启续训对齐基准；
  - 退出报告如实区分 completed/stopped 且幂等；
  - 停机后 join/next/submit 全部被拒（`shutting_down`）；
  - 保留上限生效且重启不像全新作业；显式回滚到更早点继续跑对齐基准；
  - 显式回滚点缺失/损坏被可区分拒绝；`-resume-fallback` 明确回退到可用点并对齐基准。
- `coordinator/serve_lifecycle_test.go`：真实 `Serve`（unix socket）+ 真实 worker
  连续多轮“跑一段、停、用同一 socket 路径重启”，每轮断言协调者在有限时间内自然
  退出、socket 资源释放、退出报告正确，最终对齐单机基准且检查点数量不超过保留上限。
- `coordinator/e2e_socket_test.go`：真实 unix socket + 多 goroutine worker 的端到端，
  覆盖 IPC、心跳、并发与自动重连，完成后协调者干净退出并报告 completed。

所有测试默认走确定性路径（固定种子、可控假时钟），并用 `t.Logf` 打印成员列表、
分片区间与每一步的判定依据（`go test -v` 可见）。
