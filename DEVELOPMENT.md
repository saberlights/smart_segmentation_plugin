# 智能分段插件（smart_segmentation_plugin）开发文档

> 本文档面向维护者，记录插件的运行模型、Hook 链路、核心机制、关键设计决策与历史教训。
> 阅读源码前先读本文，能避免重复踩坑。

---

## 1. 插件定位

对 Maisaka 回复模型产出的**主回复文本**做 LLM 智能分段，并把分段结果**分条发送**到聊天流，
模拟人类“一句一句说话”的节奏，避免一大段文字刷屏。

- 插件 ID：`saberlights.smart-segmentation-plugin`
- 主源码：`plugin.py`
- 命令：`/smart_seg on|off|status` 运行时开关

---

## 2. 运行模型（最重要，先理解这个）

**插件运行在独立子进程（Runner）里，通过 RPC/IPC 与宿主进程通信。**

证据与含义：

- 宿主侧 `src/plugin_runtime/host/supervisor.py` 用 `asyncio.create_subprocess_exec` 拉起
  Runner 子进程，`_rpc_server` 建连握手；Hook 调用经 `supervisor.invoke_plugin("plugin.invoke_hook", ...)`
  跨进程派发（见 `hook_dispatcher.py` 的 `response_envelope` / `RPCError` / `ipc_socket_path`）。
- **致命推论：插件子进程里 `from src.xxx import 某单例` 拿到的是子进程自己 import 的一份全新对象，
  不是宿主进程里那份有数据的单例。** 任何想直接操作宿主内存单例（如
  `heartflow_manager.heartflow_chat_list`）的代码，在生产里都会拿到空对象、静默失效。
- 插件能把消息发出去，是因为走的是 **SDK 能力 `ctx.send.text(...)`**：它经 RPC 回到宿主进程，
  由宿主的 `src/plugin_runtime/capabilities/core.py::_cap_send_text` → 宿主 `send_service` 执行，
  发送与历史同步都发生在**宿主进程**。

> 一句话原则：**凡是要落到宿主状态（历史、发送）的操作，必须经 SDK/RPC 走回宿主进程，
> 绝不能在插件子进程里直接 import 宿主单例去改。**

参见记忆笔记 `prefer-plugin-side-fixes`（优先纯插件方案）与 `avoid-full-plugin-reads`
（plugin.py 很长，定位用 grep、改用 Edit/sed、验证用 grep+AST，别整文件读）。

---

## 3. 宿主侧关键背景：为什么会“重复回复”

这是插件存在的最棘手问题，也是分段架构的核心约束来源。

Maisaka 推理主循环（`src/maisaka/reasoning_engine.py::run_loop`）：

1. 一批消息进来，先跑一次 **Timing Gate**（决定要不要理；`continue` 后该轮内不再跑）。
2. 进入**内部轮次循环**（`MAX_INTERNAL_ROUNDS = 10`）：planner 调 `reply` 工具发回复。
3. **`reply` 工具不设 `pause_execution`**（只有 `no_action`/`finish`/`wait` 才设）。所以一次回复后
   循环 `tool_continue`，**立刻重跑 round-1 planner** 去“判断话说完没”。
4. round-1 planner 读 maisaka 内部历史（`_chat_history`）。**如果此时它只看到第一段，就会判定
   “还没说完”，再调一次 `reply`，导致同一条用户消息被回复两次。**

群聊 vs 私聊差异：私聊侧把这个内部轮次/Timing Gate 关掉了，所以 round-1 planner 不重跑、不复现；
群聊开着就会复现。**所以“重复回复”本质是：分段后，群聊 round-1 planner 在剩余段进入历史之前就抢跑了。**

- `reply` 工具发首段时用宿主进程内的活对象同步历史（`src/maisaka/builtin_tool/reply.py`，
  `sync_to_maisaka_history=True`），**首段 seg1 一定进历史**。
- 剩余段 seg2..N 由本插件负责。**只要 seg2..N 没能在 round-1 planner 之前进宿主历史，就会重复回复。**

---

## 4. Hook 链路（当前架构：即发即同步）

| Hook | mode | 作用 |
|---|---|---|
| `maisaka.replyer.after_response` | 普通 | 拿到模型回复后**预分段**，存入进程内缓存 `_prepared_segment_registry`。这是“该消息是否来自回复模型”的唯一可靠信号。 |
| `send_service.after_build_message` | 普通 | 消费预分段缓存：把出站消息改写为 **seg1**，把 **seg2..N** 登记到 `_pending_follow_up_segments`。 |
| `send_service.after_send` | **BLOCKING** | **核心**。seg1 发送成功后触发；在本阻塞窗口内**同步发完** seg2..N，每段 `ctx.send.text(sync_to_maisaka_history=True)` 经 RPC 在宿主侧落历史。 |
| `chat.command.before_execute` / `after_execute` | 普通 | 命令执行期间标记聊天流，避免命令回执被误分段（`_active_command_streams` + 短保护窗口）。 |

辅助状态：
- `_prepared_segment_registry`：预分段缓存（TTL 60s）。
- `_pending_follow_up_segments`：待补发段登记（TTL 60s）。
- `_stream_resend_guards` + `_guard_stream_resend()`：补发期间关闭二次分段，并让补发段自身的
  `after_send` 直接放行（防重入）。

---

## 5. 核心机制：即发即同步（send-and-sync-immediately）

### 5.1 为什么这样设计

要根治“重复回复”，必须保证 **reply 工具返回前，seg2..N 已全部进入宿主 maisaka 历史**。
唯一可靠的跨进程落历史途径是 `ctx.send.text(sync_to_maisaka_history=True)`。于是：

- 把 `after_send` 注册为 **`mode=HookMode.BLOCKING`**：宿主发送链路会 `await` 它，
  reply 工具因此被阻塞到所有分段发完才返回。
- 在该窗口内**同步 `await` 连发** seg2..N，**不做段间 sleep**。
- reply 返回时 N 段已全部落历史 → round-1 planner 看到完整回复 → 不再重复回复。

### 5.2 超时处理

`send_service.after_send` 的宿主默认超时 `default_timeout_ms=5000`。连发多段会超过它，
被 cancel 就会丢尾段。解决：**handler 级 `timeout_ms` 优先于 hook spec 默认值**
（见 `hook_dispatcher.py::_resolve_timeout_ms`：`if entry.timeout_ms > 0: return entry.timeout_ms`）。
因此设：

```python
_AFTER_SEND_FOLLOW_UP_TIMEOUT_MS = 30_000
```

放宽到 30s，覆盖多段连发，避免 cancel 丢段。

### 5.3 代价（已知权衡）

即发即同步**放弃了逐段打字延迟节奏**：所有段几乎连续发出，reply 工具会被阻塞
≈“段数 × 单段 RPC 往返”的时长（正常 1–3s）后才返回。这是为彻底消除重复回复做的取舍。

> `max_segments` 越大，BLOCKING 阻塞越久。若平台发送较慢，注意观察 reply 返回延迟，
> 必要时下调 `max_segments`。

---

## 6. 历史教训（务必读，避免回退到旧坑）

这些是已经踩过、并已修正的坑，源码注释里也保留了说明：

1. **OBSERVE 是 fire-and-forget。** 旧实现把 `after_send` 注册为 `OBSERVE`，宿主用
   `asyncio.create_task` 后台跑、不被发送链路等待（`hook_dispatcher.py` 第 17 行注释 +
   `_schedule_observe_handler`）。于是补发与 round-1 planner 赛跑，planner 常抢先 → 重复回复。
   **结论：要让发送链路等待补发完成，必须用 `BLOCKING`，不能用 `OBSERVE`。**

2. **后台 task + 段间 sleep 是重复回复的直接成因。** reply 早早返回、剩余段还在 sleep，
   历史只有 seg1。**已删除**整套后台 task 体系（`_active_follow_up_tasks*`、idle event、
   `_track/_drain/_wait` 等）与 `_calculate_send_delay`。

3. **presync 跨进程从未生效（已删除）。** 曾有
   `_presync_follow_up_segments_to_maisaka_history`，企图在插件子进程里直接
   `import heartflow_manager` 并 `heartflow_chat_list.get()` 抢先写历史。但插件在子进程，
   拿到的是空单例，`return False` 必然触发，**生产里从未成功过**。它的 docstring 描述的
   “后台 sleep 导致只看到首段”在即发即同步下也已不成立。**已彻底删除。**

4. **`chat.receive.before_process` 阻塞不在内部轮次链路上（已删除）。** 曾想用它阻塞
   planner，但 round-1 planner 重跑是推理循环内部 `continue`，根本不走入站处理链路，拦不住。
   （注：更早还挂过 `maisaka.planner.before_request`，但宿主会把整个 planner prompt 序列化进
   IPC 帧，撑爆 16MB 帧上限，所以也废弃。）

5. **字节码缓存会加载旧逻辑。** 改完源码若仍报“旧行号 / 旧签名”的错（traceback 行号与磁盘
   文件对不上），多半是 `__pycache__/plugin.cpython-*.pyc` 陈旧。删掉 `__pycache__` 下的
   `plugin.cpython-*.pyc` 让运行器按当前源码重新编译。

---

## 7. 配置项

定义在 `SegmentationSectionConfig`（改配置只改模板并升版本号，不动实际 bot 配置）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | 是否启用分段 |
| `model` | "" | 分段模型，空则用宿主默认 |
| `style` | natural | 分段风格：natural / conservative / active |
| `min_length` | 15 | 启用分段的最小文本长度 |
| `max_segments` | 8 | 最大分段数（即发即同步下直接影响 reply 阻塞时长） |
| `temperature` | 0.3 | 分段模型温度 |
| `max_tokens` | 600 | 分段模型最大输出 token |
| `delay_base` / `delay_per_char` / `delay_max` | — | **历史延迟参数；即发即同步后已不生效（死数据）**，保留仅为配置向后兼容。 |

---

## 8. 已知遗留 / TODO

- `delay_base` / `delay_per_char` / `delay_max`：配置字段、`_get_segmentation_runtime_settings`
  的读取、`after_build` 登记 pending 时写入的这三项，都是**即发即同步后没人读的死数据**。
  建议后续单独 `chore` 提交清理（连带决定 config 字段是否删除并升版本号）。

---

## 9. 维护这份代码的工作约定

- **不要整文件读 `plugin.py`**（很长，浪费上下文）。定位用 `grep -n`，看局部用 `sed -n 'A,Bp'`，
  改用 Edit（精确 old_string）或 `sed -i 'A,Bd'`，验证用 `grep -c` + `python3 -c "import ast; ast.parse(...)"`。
- 多步删除/重构后，务必用 `grep` 复核“是否真的落盘 / 是否产生重复定义 / 是否还有对已删符号的引用”——
  对话中断容易导致 Edit 没真正写入，留下半成品。
- 优先纯插件方案；确需改宿主主程序时先申请许可。
- 提交遵循 Conventional Commits（中文），原子提交，不加 AI 署名。
