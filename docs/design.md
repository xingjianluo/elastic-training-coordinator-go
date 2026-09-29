# 设计说明

本项目的正确性边界是“同一确定性模型、同一优化器更新、成员变化只改变同步代数，不改变已提交
step 的数学含义”。

核心不变量：

1. 分片计划完全由 `(dimension, sorted(worker_ids))` 决定。
2. 每个 plan 的区间完整覆盖参数维度一次。
3. 只有当前 generation 的全部当前成员都贡献后，step 才能提交。
4. 成员变化会丢弃未提交梯度并递增 generation，已提交 step 不回滚。
5. `(generation, step, worker_id)` 的贡献最多生效一次。
6. 检查点只在 checksum 和状态校验通过后加载。
7. slow 与 dead 的判定依据分别是 step soft deadline 与 heartbeat hard deadline。

本地 IPC 使用 JSON 帧 over Unix-domain socket。协调器只向 worker 下发其拥有的参数片，
worker 只回传同一半开区间的梯度片并由协调器 allgather；库级 allreduce 路径则保留完整副本
数据并行语义。每个请求带序列号，worker 端单写者发送、单读者按序列号分发响应，从而允许等待
梯度提交期间继续发送 heartbeat。

演示二进制通过 `os/exec` 启动独立 worker 子进程，因此训练路径是本地进程间通信而不是仅在
单进程内用 goroutine 模拟。
