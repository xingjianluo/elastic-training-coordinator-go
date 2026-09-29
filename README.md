# Elastic Training Coordinator (Go)

一个只依赖 Go 标准库和本地系统能力的弹性分布式训练协调器示例。演示进程会启动一个
协调器进程和多个独立 worker 子进程，它们通过 **Unix-domain socket** 通信；不使用 TCP、
外部服务、真实凭据或网络资源。

默认训练任务是确定性的二次目标：

- 参数初始值：全 0。
- 目标值：`[1, 2, ..., d]`。
- 梯度：`g = params - target`。
- 优化器：带动量的 SGD，固定 `lr=0.05`、`mu=0.9`。
- 单机基准：`training.Baseline(dimension, steps, lr, mu)`。

所有分布式测试都把最终参数与这个单机基准逐元素比较，默认容差为 `1e-12`。

## 代码结构

```text
pkg/sharding     确定性分片计划与对齐校验
pkg/collective   allreduce、allgather、reduce-scatter
pkg/training     确定性模型、梯度和单机基准
pkg/membership   成员视图、成员顺序、slow/dead 判定
pkg/checkpoint   原子、带版本和校验和的检查点
pkg/cluster      Unix socket 协议、worker、协调器状态机
cmd/.../main.go  本地端到端演示
```

## 分片语义

`sharding.PlanFor(dimension, workers)` 先对 worker ID 排序，再使用固定边界：

```text
range_i = [ floor(i*d/n), floor((i+1)*d/n) )
```

因此：

1. worker 的发现顺序、map 遍历顺序不会影响分片位置；
2. 所有区间连续、无重叠、无空洞，完整覆盖 `[0,d)`；
3. worker 数量大于参数维度、worker 列表为空或存在重复 ID 时返回
   `ErrInvalidShardCount`；
4. 收到的计划缺区间、区间长度不对、成员不同或维度不同时返回
   `ErrMisalignedShards`。

协调器每次成员变化都会生成新的 `generation` 和新的分片计划，并在日志中打印：

- 当前成员列表；
- 每个 worker 的半开区间 `[start,end)`；
- 为什么建立或重建该计划。

## 集合通信语义

`pkg/collective` 提供三类同步操作：

- **allreduce(sum)**：对每个 worker 的同长度完整副本逐元素求和；
- **reduce-scatter(sum)**：先执行 allreduce(sum)，再让每个 worker 只保留自己负责的区间；
- **allgather**：按计划把各 worker 拥有的区间重新拼接成完整向量。

恒等关系：

```text
allgather(reduce-scatter(replicas, plan), plan) == allreduce(replicas, sum)
```

本项目暴露两条清晰的集合通信路径：

1. **库级数据并行路径**：`Contribute` 接收每个 worker 的完整副本，用 allreduce(sum) 求和，再除以当前成员数：

```text
applied_gradient = sum(local_gradients) / current_member_count
```

2. **本地 IPC worker 路径**：join 响应只下发该 worker 拥有的参数片，worker 也只回传该参数片的梯度；协调器用 allgather 按计划拼出完整梯度：

```text
applied_gradient = allgather(owned_gradient_shards, current_plan)
```

示例中每个参数位置由排序后拥有它的唯一 worker 计算，因此 allgather 的结果等于单机梯度。同一个 step 禁止混用完整副本和分片提交，避免错误地把两种集合操作叠加。成员变化时，未提交的梯度全部丢弃并在新 generation 重试，避免：

- 已退出 worker 的梯度被错误提交；
- 新加入 worker 被遗漏；
- 同一个 worker 在同一 `(generation, step)` 重复累加。

以下情况会返回不同错误，而不是静默产生错误结果：

- 无副本：`ErrEmptyReplica`；
- 副本长度不同：`ErrMisalignedShards`；
- allgather 缺少某个 worker 的分片：`ErrMisalignedShards`；
- 分片长度与计划区间不一致：`ErrMisalignedShards`；
- 非 `sum` 操作：`ErrUnsupportedOp`。

## 弹性成员与故障规则

每个请求都携带 worker ID、generation 和 step。协调器维护：

- `generation`：成员版本；
- `step`：尚未完成的下一步；
- `plan`：当前 generation 的确定性分片计划；
- 每个 generation/step 已接收的梯度集合。

### Join / 初始栅栏

- 初始成员数达到 `InitialWorkers` 前，worker 处于 pending；
- 达到阈值后建立 generation 0 和初始计划；
- join 响应返回当前 generation、step、计划和权威参数。

### 成员中途加入或退出

- join/leave 都会使 `generation` 加一；
- 当前 step 已收到但未提交的梯度全部清空；
- `step` 不前进，幸存者使用新计划和新成员数重试该 step；
- 使用旧 generation 或旧 step 发来的贡献返回 `generation_restart`；
- worker 收到该响应后按响应中的新计划重试，不本地猜测状态。

### 幂等与完成后重连

同一 worker 对同一 `(generation, step)` 的第二次贡献不会再次进入聚合，返回
`already_completed`。训练结束后的重连同样返回该码，响应包含最终参数但不会产生新贡献。

### Slow 与 dead 的区别

协调器使用两个时限：

- **StepTimeout / soft deadline**：当前 step 等待太久，但缺失 worker 仍有新鲜 heartbeat，
  判定为 `slow_worker` / `ErrSlowWorker`；
- **HeartbeatTimeout / hard deadline**：连接关闭或 heartbeat 静默超过硬超时，判定为
  `worker_dead` / `ErrWorkerDead`，然后删除成员并进入新 generation。

也就是说，“算得慢但还活着”和“连接已断或心跳消失”不会被合并成同一种失败。

## 检查点格式与恢复规则

检查点包含恢复训练所需的全部状态：

- 参数 `Params`；
- 优化器动量 `Momentum`；
- 目标 `Target`；
- `Generation`、`NextStep`、`TotalSteps`；
- 有序成员列表；
- 完成状态。

文件容器格式：

```text
magic("ETCK") | version(uint16) | payload_length(uint64) | gzip(JSON payload) | SHA-256(payload)
```

写入规则：

1. 写入同目录唯一临时文件；
2. flush + `fsync`；
3. 原子 `rename` 为正式检查点。

写到一半时通常只留下 `.tmp` 文件，正式路径仍指向上一份完整检查点。读取时严格区分：

| 情况 | 返回错误 |
| --- | --- |
| 文件不存在 | `ErrNotFound` |
| magic 不对，根本不是检查点 | `ErrInvalidCheckpoint` |
| 版本不受支持 | `ErrUnsupportedVersion` |
| 头部声明长度大于实际文件，文件截断 | `ErrPartial` |
| checksum 不匹配、gzip/JSON 损坏、状态长度不一致 | `ErrCorrupt` |

恢复后从 `NextStep` 继续，不重放已经提交的 step；成员和分片计划也从检查点重建。

## 本地验证

### 端到端演示

```bash
go run ./cmd/elastic-training-demo
```

演示会：

1. 在临时目录创建 Unix socket 和检查点文件；
2. 启动 3 个独立 worker 子进程；
3. 每个子进程只接收并计算自己的参数片，执行 4 步 allgather 训练；
4. 再启动一个 worker-0 子进程做完成后重连，验证幂等；
5. 打印成员、分片区间、generation、step 和与单机基准的最大误差。

### 全部测试

```bash
go test ./... -count=1 -v
```

带竞态检测：

```bash
go test ./... -race -count=1
```

### 测试覆盖矩阵

| 测试 | 覆盖内容 |
| --- | --- |
| `TestPlanOrderIndependentAndBalanced` | worker 输入顺序变化不改变分片 |
| `TestPlanRejectsInvalidShardCountAndMisalignment` | 非法分片数、重复 worker、分片错位 |
| `TestAllReduceAndReduceScatterAllGatherOrderIndependent` | allreduce 与 reduce-scatter+allgather 等价且顺序无关 |
| `TestCollectiveRejectsMisalignedInput` | 缺片、长度不一致、非法 op |
| `TestCoordinatorDeterministicAcrossShardCounts` | 1/2/4 个 worker 且每步打乱贡献顺序，结果等于单机基准 |
| `TestElasticJoinLeaveRetriesStepAndMatchesBaseline` | 中途退出、替代成员加入、generation 重试且不丢/不重梯度 |
| `TestSlowWorkerIsDistinguishedFromDeadWorker` | 有 heartbeat 的掉队节点判定为 slow |
| `TestHeartbeatTimeoutIsDeadAndRebuildsGeneration` | heartbeat 超时判定为 dead 并重建 generation |
| `TestDuplicateContributionDoesNotDoubleCount` | 同一步重复贡献不重复累加 |
| `TestCheckpointResumeContinuesAndMatchesBaseline` | 检查点恢复后继续训练并匹配基准 |
| `TestRejectsPartialCorruptVersionAndInvalidContainer` | 半截、损坏、版本错误、非法容器分别返回不同错误 |
| `TestContributeShardMatchesBaselineAndRejectsMisalignment` | 分片梯度 allgather 后等于单机基准 |
| `TestContributeShardRejectsWrongRangeAndLength` | 错误分片区间和错误分片长度返回对齐失败 |
| `TestUnixSocketWorkersAndCompletedReconnect` | 真实本地 Unix socket IPC、完成后重连幂等、race 检测 |
| `go run ./cmd/elastic-training-demo` | 3 个独立 worker 子进程，仅传输所属参数/梯度分片 |

测试使用 `-v` 时会输出成员列表、分片区间、提交/丢弃/重试/检查点判定依据。

## 设计限制

这是一个本地教学和验证用协调器：

- worker 数量不能超过参数维度；
- 当前 allreduce reduction 只实现 `sum`；
- 演示使用独立本地进程 + Unix socket 模拟多个 worker，不依赖外部集群；
- 检查点是单文件原子替换，适合本地单机验证，不是分布式共享存储提交协议。
