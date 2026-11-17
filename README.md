# 风机停机预警训练平台（可续跑）

用 FastAPI + NumPy + SQLite 实现的风机停机预警逻辑回归训练平台。训练由两个工作器
容器执行，每个批次都把权重、动量、损失和进度作为**一个原子断点**提交；工作器退出后
另一个工作器自动接管，并从最近断点继续。最终权重与不中断训练逐位一致。

## 组成

| 组件 | 说明 |
| --- | --- |
| `api` | FastAPI，提交任务、查询进度/断点/权重、Swagger 演示 |
| `worker-1` / `worker-2` | 同一镜像的 `python -m app.worker`，共享同一个 SQLite 文件 |
| `training-data` | Docker 卷，挂载到三个容器的 `/data` |

两个工作器共享**同一个** SQLite 数据库（`/data/training.db`），因此"检查租约并写入
断点"和"写入断点"是同一个 `BEGIN IMMEDIATE` 事务，不需要跨库协调。

## 训练约定

- 输入 8～64 条记录，每条是风机振动/电流等特征，标签 `0`=正常、`1`=停机预警；
  各特征列按提交样本标准化后追加偏置列，权重长度为 `特征数 + 1`。
- 轮数 2～5；按输入顺序每 8 条一批（最后一批可不足 8 条）。
- 逻辑回归（sigmoid + 二元交叉熵），SGD，学习率 `0.1`，动量 `0.9`，float64。
- 每批提交的断点包含：`step`、`training_epoch`、`batch_in_epoch`、`weights`、
  `momentum_velocity`、`batch_loss`、`loss`（全量交叉熵）、`progress`、提交时间、
  `worker_id`、`lease_epoch`。

## 启动

```bash
docker compose up -d --build
# Swagger: http://localhost:8000/docs
```

`worker-1` / `worker-2` 故意不设 `restart` 策略，便于演示接管。被 kill 的工作器用
`docker compose start worker-1` 恢复（它会以空闲状态重新加入，不会抢占已完成任务）。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/jobs` | 提交 `{"features": [[...]], "labels": [...], "epochs": 3}`，返回 201 与任务 |
| `GET` | `/jobs` | 最近任务列表 |
| `GET` | `/jobs/{job_id}` | 已确认步数、进度、损失、当前权重；完成时含 `final_weights`；含隔离状态 |
| `GET` | `/jobs/{job_id}/checkpoints` | 全部批次断点（可按 `after_step` 增量拉取） |
| `GET` | `/jobs/{job_id}/events` | 审计事件：创建、开始/从第几步续训、完成、丢失租约、核验通过、被隔离 |
| `POST` | `/jobs/{job_id}/verify` | 逐批重算核验全部断点；异常时隔离该任务（见下文） |
| `GET` | `/workers` | 工作器心跳与租约持有者 |
| `GET` | `/health` | 健康检查 |

`Swagger`（`/docs`）里直接 `POST /jobs` 提交即可看到训练开始；`GET /jobs/{id}` 与
`GET /jobs/{id}/checkpoints` 查看进度与每个断点的权重/动量/损失；`POST /jobs/{id}/verify`
只需任务 ID 即可在 Swagger 里直接执行核验。

## 断点核验与隔离

恢复历史训练库后，对任意任务执行 `POST /jobs/{job_id}/verify`：

1. **一致快照**：在一个读事务里取出任务行与全部断点（WAL 快照，不阻塞工作器写入）。
2. **逐批重放**：用任务里保存的样本与训练参数（学习率、动量、批大小），从零权重
   开始逐批重算权重、动量与损失，与每个断点逐一比对——即使最新断点与任务进度
   完全相符，中间批次的失真也会在它自己的步数上被发现。同时校验任务行当前状态
   与重放结果是否一致（含初始零状态、完成状态与步数的一致性）。
3. **原子结论**：在同一个 `BEGIN IMMEDIATE` 事务里复核进度并落地结论：
   - `ok`：确认进度在核验期间没有变化，断点可信、可用于续训，记录
     `checkpoint_verified` 事件；
   - `failed`：返回**首个**缺失或不一致的步数与原因（缺失、权重/动量/损失不符、
     出现进度之外的断点、任务行撕裂等），并在同一事务里**隔离**该任务
     （`quarantined=1`、删除租约、记录 `job_quarantined` 事件）。断点是只追加的，
     已被判定异常的历史不会因后续提交而"自愈"；
   - `stale`：核验期间任务进度发生了变化，本次不下结论（核验绝不把已过期的
     快照误报为通过），重试即可。

隔离后的行为：

- 工作器不再提交该任务的批次（提交路径在事务内检查隔离标志，迟到唤醒的旧工作器
  同样被 `JobQuarantined` 围栏挡下），也不会接管该任务（`claimable_jobs` 已过滤）；
- 任务状态、原有断点、事件仍可正常查询（`GET /jobs/{id}` 对隔离任务放宽了
  "最新断点必须等于进度"的硬校验，否则会查询不到被判定撕裂的任务）；
- 其他任务照常训练；已完成但被判异常的任务保留 `completed` 状态，但不再提供
  `final_weights`（结果不可信）。

从旧归档恢复的训练库缺少隔离相关列，所有进程启动时的 `init_db` 会幂等地
`ALTER TABLE` 补齐，无需手工迁移。

## 续跑与防回退机制

1. 工作器每批在**同一个事务**里写检查点、更新任务进度、刷新租约心跳。任何一项失败
   则整体回滚，不会出现"权重已变但进度没变"的中间态。
2. 租约保存在同一数据库的 `leases` 表。工作器每 `1.5s` 心跳；持有者心跳超过
   `6s`（`TRAINING_LEASE_SECONDS`）未更新，另一个工作器才可接管，接管时
   `epoch + 1`。
3. 每次提交前在事务内校验 `(worker_id, epoch)` 仍是租约持有者。被接管后仍存活（或
   迟到唤醒）的旧工作器校验失败，直接 `LeaseLost` 退出：既不能回退断点，也不能把
   别人的完成状态改写，更不能重复提交完成事件。
4. 完成状态由 `step == total_steps` 在提交事务内一次性置位；旧工作器无法再写入。
5. SQLite 使用 WAL + `synchronous=FULL`，已提交的断点在进程崩溃/容器重启后仍然存在。

## 演示与验证

```bash
# 完整演示（Docker）：提交 -> 观察 -> kill 当前持有者 -> 接管续跑 -> 对比不中断训练
python3 scripts/demo.py

# 没有 Docker 时用本机进程演示同样流程
python3 scripts/demo.py --local

# 不依赖 Docker 的四项自检（含断点核验与隔离）
python3 scripts/selftest.py

# 单元/集成测试（含真实 uvicorn + 两个工作器进程、SIGKILL 接管、迟到工作器围栏、
# 断点核验/隔离/交错提交）
python3 -m pytest -q
```

演示脚本第 6 步会用同一份输入再训练一次（不中断），并断言两次的 `final_weights`、
`loss`、`momentum_velocity` 完全相等。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TRAINING_DATA_DIR` | `./data` | 数据库目录 |
| `TRAINING_DB_PATH` | `$TRAINING_DATA_DIR/training.db` | 共享数据库 |
| `TRAINING_LEASE_SECONDS` | `6` | 租约超时 |
| `TRAINING_HEARTBEAT_INTERVAL` | `1.0` | 心跳间隔（必须小于租约） |
| `TRAINING_POLL_INTERVAL` | `0.3` | 空闲轮询/批间间隔 |
| `TRAINING_BATCH_DELAY_SECONDS` | `0` | 模拟每批耗时，便于演示 |
| `TRAINING_DEMO_PAUSE_FILE` | 未设置 | 演示用：文件存在时在批间暂停（不心跳） |
| `TRAINING_DEMO_CRASH_AFTER` | 未设置 | 演示用：指定步数提交后 `os._exit` |

## 目录

```
app/main.py       API 与 Swagger（含 POST /jobs/{id}/verify）
app/worker.py     工作器主循环、租约续训（隔离任务按丢失租约处理）
app/store.py      任务状态机、批次提交、断点查询、断点核验与隔离
app/lease.py      租约获取/续约/围栏校验
app/model.py      逻辑回归、交叉熵、SGD + 动量
app/features.py   输入校验与标准化
app/db.py         SQLite 连接、表结构与旧库迁移
tests/            单元、API、核验、进程级集成测试
scripts/demo.py   Swagger 同款接口的端到端演示
scripts/selftest.py  无 Docker 自检
```
