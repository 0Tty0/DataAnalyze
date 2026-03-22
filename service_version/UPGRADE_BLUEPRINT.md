# DataAnalyze Service 升级改造蓝图（仅 `service_version`）

## 0. 约束与范围
- 本蓝图只覆盖 `service_version`，不修改 `lightweight`。
- 目标是把当前可运行原型升级为可上线的生产级数据分析 Agent 服务。
- 升级遵循：先安全与稳定，再并发与质量，再可观测与企业集成。

---

## 1. 当前架构快照（As-Is）

### 1.1 代码结构
- 入口：[start_service.py](D:\笔记\DeepAnalyze\service_version\start_service.py)
- API 服务：[service/app.py](D:\笔记\DeepAnalyze\service_version\service\app.py)
- Agent 核心：[deepanalyze_langgraph.py](D:\笔记\DeepAnalyze\service_version\deepanalyze_langgraph.py)
- 存储层：[service/storage.py](D:\笔记\DeepAnalyze\service_version\service\storage.py)
- 前端页面：[web/index.html](D:\笔记\DeepAnalyze\service_version\web\index.html)

### 1.2 主流程
- `FastAPI -> chat_completions -> DeepAnalyzeLangGraph.generate()`
- LangGraph 节点：`Planner -> Coder -> Executor -> Reporter`

### 1.3 关键问题
- `Executor` 使用裸 `exec()`，存在任意代码执行风险。
- `generate()` 中使用 `os.chdir()`，存在全局副作用与并发风险。
- 服务是同步阻塞调用，吞吐能力有限。
- 状态以进程内内存为主，重启后不可恢复。
- 缺少质量校验节点（无 Critic），缺少上下文压缩机制。
- 可观测性只有基础日志，无 trace / token / 成本汇总。

---

## 2. 目标架构（To-Be）

### 2.1 执行安全
- `Executor` 改为隔离执行（阶段式：`subprocess` -> `container`）。
- 强制超时、内存上限、CPU 限额、最小权限。

### 2.2 Agent 编排
- 流程升级为：`Planner -> Coder -> Executor -> Critic -> Reporter`
- 加入上下文压缩与可恢复 checkpoint。

### 2.3 服务能力
- API 与 Agent 全链路异步化（`AsyncOpenAI` + `ainvoke/astream_events`）。
- 支持前端流式输出（推荐 SSE）。

### 2.4 可观测与审计
- 节点级耗时、token 用量、成本估算、运行 trace。
- 结构化日志包含 `user_id`、`thread_id`、`run_id`。

---

## 3. 分阶段实施（优先级）

## P0（必须先完成）
### P0-1：执行器安全重构
- 文件：
  - [deepanalyze_langgraph.py](D:\笔记\DeepAnalyze\service_version\deepanalyze_langgraph.py)
- 改造：
  - 废弃 `exec(code_str, {})`。
  - 新增 `SandboxExecutor`（先用 `subprocess.run`，`cwd=workspace`，`timeout`）。
  - 预留容器执行接口（`DockerExecutor` 占位 + 配置开关）。
- 验收：
  - 死循环代码在超时后被终止。
  - 不再出现主进程级 `os.chdir()`。
  - 执行失败输出可追踪（stderr + return code）。

### P0-2：移除全局副作用
- 文件：
  - [deepanalyze_langgraph.py](D:\笔记\DeepAnalyze\service_version\deepanalyze_langgraph.py)
- 改造：
  - 删除 `generate()` 内的 `os.chdir(workspace)`。
  - 全部执行与文件写入显式传 `workspace` 路径。
- 验收：
  - 并发多个线程时互不污染 cwd。

### P0-3：最小持久化安全底座
- 文件：
  - [service/storage.py](D:\笔记\DeepAnalyze\service_version\service\storage.py)
  - [service/app.py](D:\笔记\DeepAnalyze\service_version\service\app.py)
- 改造：
  - 线程元数据改为持久存储（SQLite 首版）。
  - 启动时可加载最近线程索引。
- 验收：
  - 服务重启后可继续读取历史 thread、messages、reports。

## P1（质量与吞吐）
### P1-1：异步化
- 文件：
  - [deepanalyze_langgraph.py](D:\笔记\DeepAnalyze\service_version\deepanalyze_langgraph.py)
  - [service/app.py](D:\笔记\DeepAnalyze\service_version\service\app.py)
- 改造：
  - 引入 `AsyncOpenAI`。
  - 节点函数改 `async def`，主入口增加 `agenerate()`。
  - `chat_completions` 改为异步调用。
- 验收：
  - 并发请求吞吐明显提升。
  - 无阻塞型长尾请求拖垮 worker。

### P1-2：上下文压缩
- 文件：
  - [deepanalyze_langgraph.py](D:\笔记\DeepAnalyze\service_version\deepanalyze_langgraph.py)
- 改造：
  - 新增 `ContextManager`：按 token 预算裁剪，不再按魔法数字切片。
  - 引入“中段摘要 + 最近轮次保留”策略。
- 验收：
  - 长任务 20+ 轮不出现上下文爆炸。

### P1-3：Critic 节点
- 文件：
  - [deepanalyze_langgraph.py](D:\笔记\DeepAnalyze\service_version\deepanalyze_langgraph.py)
- 改造：
  - 增加 `critic` 节点与路由：`pass/retry/escalate`。
  - `Coder` 不再直接决定完成，必须通过 Critic。
- 验收：
  - 低质量报告可自动回退修正。

## P2（生产可观测与恢复）
### P2-1：Checkpointer
- 文件：
  - [deepanalyze_langgraph.py](D:\笔记\DeepAnalyze\service_version\deepanalyze_langgraph.py)
- 改造：
  - 使用 LangGraph checkpointer（SQLite 起步）。
  - 统一 `thread_id/run_id` 作为恢复键。
- 验收：
  - 中断后支持继续运行。

### P2-2：流式输出（SSE）
- 文件：
  - [service/app.py](D:\笔记\DeepAnalyze\service_version\service\app.py)
  - [web/index.html](D:\笔记\DeepAnalyze\service_version\web\index.html)
- 改造：
  - 增加 `/v1/chat/completions/stream` 或兼容现有 `stream=true`。
  - 前端实时展示节点进度与内容增量。
- 验收：
  - 用户可看到实时进展，非整轮阻塞。

### P2-3：观测与审计
- 文件：
  - [deepanalyze_langgraph.py](D:\笔记\DeepAnalyze\service_version\deepanalyze_langgraph.py)
  - [service/app.py](D:\笔记\DeepAnalyze\service_version\service\app.py)
- 改造：
  - 节点耗时日志、token 使用、成本估算汇总。
  - 结构化日志补齐 `user_id/thread_id/run_id/node`。
- 验收：
  - 单次会话可还原全链路执行轨迹。

## P3（能力增强）
- 多模型路由（按节点指定模型）。
- ToolNode（SQL 查询、外部 API、企业工具）。
- Artifact 管理（图表/CSV/报告统一登记与下载）。
- Human-in-the-loop（关键节点人工审批）。

---

## 4. 建议目录重构（`service_version` 内）

```text
service_version/
  service/
    app.py
    storage.py
    executor/
      sandbox_executor.py
      docker_executor.py
    agent/
      graph_agent.py
      context_manager.py
      critic.py
    observability/
      tracing.py
      metrics.py
```

---

## 5. 配置项建议（新增）
- `EXECUTOR_MODE=subprocess|docker`
- `EXEC_TIMEOUT_SEC=30`
- `EXEC_MEMORY_MB=512`
- `EXEC_CPU_LIMIT=0.5`
- `MAX_CONTEXT_TOKENS=8000`
- `ENABLE_STREAMING=true`
- `CHECKPOINT_DB=./workspace/checkpoints.db`

---

## 6. 每阶段交付标准
- 代码：通过基础回归（上传、对话、历史、报告下载）。
- 安全：执行器具备隔离 + 超时 + 资源限额。
- 性能：关键 API 响应与并发指标可量化。
- 可观测：单次请求有完整日志链路。

---

## 7. 回滚策略
- 每阶段独立分支与独立 tag。
- 关键开关配置化（`EXECUTOR_MODE`、`ENABLE_STREAMING`），可灰度切换。
- 发生故障时先切回上阶段稳定 tag，再做问题定位。

---

## 8. 下一步执行建议（建议立即开始）
1. 先做 `P0-1 + P0-2`：替换执行器并移除 `os.chdir()`。
2. 完成后做 `P1-1`：异步化 Agent 调用链。
3. 再上 `P1-2 + P1-3`：上下文压缩 + Critic 质量闸门。
