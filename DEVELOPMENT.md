# 智能分段插件（smart_segmentation_plugin）开发文档

> 本文档面向维护者，记录插件的运行模型、Hook 链路、当前纯插件折中、分段结果处理与历史教训。

---

## 1. 插件定位

插件只处理 Maisaka 回复模型产出的主回复文本：先让 LLM 选择自然分段边界，再把有效结果分条发送，模拟人类连续聊天的节奏。

- 插件 ID：`saberlights.smart-segmentation-plugin`
- 当前版本：`1.1.0`
- 主源码：`plugin.py`
- 命令：`/smart_seg on|off|status`

插件提示分段模型只选择发送边界，并对模型结果做宽松的内容保真校验：只比对文字字词（NFKC 折叠全半角、忽略标点/空白/大小写），字词被改写或增删时丢弃该结果并重试/回退。命令回执、memory/expression 文本或其他插件主动发送的文本仍不会进入分段链路。

---

## 2. 运行模型

插件运行在独立 Runner 子进程，通过 RPC/IPC 与宿主进程通信。

- Hook 由宿主跨进程派发到 Runner。
- 插件子进程直接 `import src.xxx` 得到的是子进程自己的单例，不是宿主正在使用的活对象。
- 发送和 Maisaka 历史同步必须经 `ctx.send.text(...)` 回到宿主；不能在插件进程里直接修改 `heartflow_manager` 等宿主单例。

因此，所有需要改变宿主状态的操作必须使用 SDK 能力。当前后续段通过：

```python
await self.ctx.send.text(
    segment,
    stream_id,
    typing=typing_enabled,
    sync_to_maisaka_history=True,
    maisaka_source_kind="guided_reply",
)
```

---

## 3. 宿主约束与当前折中

理想实现应由宿主在原生 `reply` 工具内部暴露“最终文本已确定、逐段发送”钩子，使所有分段与原生回复共享同一发送和历史同步时序。当前宿主没有这个精确 seam，只修改插件无法获得同等保证。

插件能观察到的关键时点是：

1. `maisaka.replyer.after_response`：回复模型刚产出文本，适合预分段。
2. `send_service.after_build_message`：可以把本次出站正文替换为首段。
3. `send_service.after_send`：首段平台发送结束，但宿主在该 Hook 返回后仍有存储和 Maisaka 历史同步工作。

当前采用以下折中：

- `after_send` 使用 `OBSERVE`，只快速创建并登记后台补发任务，不让原发送链受固定 Hook 超时约束。
- 后台任务先让出一次事件循环；启用 `typing_enabled` 时，第一个后续段再由宿主执行真实打字等待。这给首段原发送链完成存储和历史同步留下时间。
- 后续段顺序 `await` 发送，并逐段设置 `sync_to_maisaka_history=True`。
- `maisaka.planner.before_request` 为同一聊天流设置内部续轮闸门：等待补发任务完成，并修补已经提前构建、只包含首段的 prompt。
- `chat.receive.before_process` 为每个聊天流设置等待门：同一流仍有补发任务时，延后处理后来收到的新消息，避免新旧内容交错。

这是插件侧的时序屏障，不改变宿主首段的历史写入顺序：后台尾段仍按真实发送顺序进入历史；planner Hook 额外修补本轮已经序列化的消息窗口。若补发超过 Hook 超时，插件会放弃等待并保留可见错误日志。

---

## 4. Hook 链路

| Hook | mode | 作用 |
| --- | --- | --- |
| `maisaka.replyer.after_response` | `BLOCKING` | 在 25 秒 handler 窗口内预分段，合法的 JSON 数组结果写入 `_prepared_segment_registry`。 |
| `maisaka.reply.before_post_process` | `BLOCKING` | 精确命中预分段缓存时跳过本次宿主文本后处理，避免出站文本变化导致缓存失配。 |
| `send_service.after_build_message` | `BLOCKING` | 消费缓存，把出站消息改为首段，并登记剩余段。发送链上不再调用 LLM。 |
| `send_service.after_send` | `OBSERVE` | 首段发送成功后快速创建后台补发任务。 |
| `maisaka.planner.before_request` | `BLOCKING` | 等待同流补发完成，并将已构建 prompt 中的首段修补为完整分段。 |
| `chat.receive.before_process` | `BLOCKING` | 等待同流补发任务结束，再放行新的入站消息。 |
| `chat.command.before_execute` / `after_execute` | `BLOCKING` | 标记命令作用域及短保护窗口，避免命令回执被误分段。 |

主要状态：

- `_prepared_segment_registry`：按 `(stream_id, normalized_text_hash)` 保存预分段结果，TTL 60 秒。
- `_pending_follow_up_segments`：首段构建完成后，等待 `after_send` 消费的后续段。
- `_active_follow_up_tasks` / `_active_follow_up_tasks_by_stream`：持有后台 Task 强引用并按流追踪。
- `_follow_up_idle_events_by_stream`：供同流入站等待门判断补发是否结束。
- `_planner_follow_up_entries_by_stream`：记录同流回复的完整分段和完成事件，供 planner 闸门消费。
- `_stream_resend_guards`：补发期间禁止再次进入分段链，避免递归补发。
- `_active_command_streams` / `_recent_command_stream_expiries`：隔离命令回执。

---

## 5. 端到端时序

```text
replyer 产出原文
  → after_response 调用分段 LLM
  → 规范化并合并超出上限的模型分段
  → 缓存有效分段
  → before_post_process 精确命中后跳过宿主正文改写
  → after_build 把原消息替换为 seg1，登记 seg2..N
  → 宿主发送 seg1
  → after_send OBSERVE 创建后台 Task 并快速返回
  → 宿主继续完成 seg1 的存储与历史同步
  → 后台发送 seg2：宿主先按 typing_speed 等待，再发送并同步历史
  → 后台依次发送 seg3..N
  → planner.before_request 等待补发完成，并把 prompt 中的 seg1 修补为 seg1..N
```

如果此时同一聊天流收到新消息：

```text
chat.receive.before_process
  → 发现该流仍有活跃补发 Task
  → 等待流空闲事件
  → 补发结束后放行新消息
```

不同聊天流互不等待。

---

## 6. 分段结果处理

分段提示词沿用早期的真人聊天模仿方式，通过长短不均的具体对话示例校准发送节奏。不要按字符数给模型指定目标段数，也不要把逗号或抽象的“意图变化”默认当作切点；相关内容应留在同一条消息中。配置项 `max_segments` 只作为防止刷屏的硬上限，语义上不适合切分时模型应返回单段。

提示词要求分段模型只选择边界，不要创作或删除内容。插件处理顺序为：

1. 返回值必须是非空 JSON 数组（响应先剥离 `<think>`/`<thinking>` 推理标签，再做数组切片提取，支持大写围栏与围栏内前置说明文字）。
2. 空段会被移除。
3. 超出 `max_segments` 时保留前 `max_segments - 1` 段，并把模型返回的所有尾段拼回最后一段。
4. 修复模型误拆的括号内容，并按括号边界拆出完整动作段。
5. 内容保真校验：分段拼接后与原文只比对文字字词（忽略标点、空白、全半角、大小写）；标点微调、括号宽度转换、独立省略号被省略均放行，字词改写/增删则丢弃结果。
6. JSON 结构/解析异常或保真校验不通过时拒绝结果，触发同模型重试，仍失败则原发送链继续发送原文。

分段 LLM 调用的 `max_tokens` 按原文长度动态扩容（`max(配置值, 原文长度 × 2 + 160)`），避免长文的 JSON 输出被固定值截断。

运行时只把 `max_segments` 的下限限制为 1，不设置额外上限；具体段数以插件配置为准。

---

## 7. 模拟打字与发送节奏

插件不再维护自己的延时公式。`segmentation.typing_enabled = true` 时，所有后续段首次发送都传 `typing=True`，由宿主 `send_service` 调用当前 MaiBot 的 `calculate_typing_time`，并读取：

```toml
[response_post_process]
typing_speed = 1.0
```

这项等待不依赖 `enable_response_post_process` 是否开启。`typing_enabled = false` 时插件为所有后续段传 `typing=False`，但仍保持顺序发送和 Maisaka 历史同步。普通文本在 `typing_speed = 0` 时关闭常规等待；数值越大，等待越久。宿主当前对“单个中文字符”有提前返回的固定等待分支，不受 `typing_speed` 控制，插件不复制或修补这项上游语义。

发送时机是“输入当前后续段，再发送当前段”：

```text
seg1 已由原链发送
等待 typing_time(seg2) → 发送 seg2
等待 typing_time(seg3) → 发送 seg3
...
发送 segN 后不额外等待
```

旧配置键 `delay_base`、`delay_per_char`、`delay_max` 已从实现删除。旧配置文件即使仍包含这些键也会被忽略，不能与宿主模拟打字叠加。

---

## 8. 超时与失败语义

- `maisaka.replyer.after_response` 的 handler 超时为 25 秒。首次分段请求超过 6 秒时并发重试同一已配置模型，两个请求共用 20 秒内部总预算；任一请求先返回有效结果即采用并取消另一个，全部失败或超时才保留原文。
- `send_service.after_send` 只创建后台任务，不承担完整打字等待；`maisaka.planner.before_request` 负责等待同流补发完成。
- 某个后续段发送失败时停止发送剩余段并记录具体段号，不伪装成功。
- 插件卸载时取消并 drain 仍在运行的补发任务，避免 Task 泄漏。
- 同流入站等待有独立超时保护；它用于避免交错，不应被描述为同一 reply planner 的严格屏障。

---

## 9. 历史教训

1. **不要跨进程直接改宿主单例。** 插件子进程中的 `heartflow_manager` 不是宿主活对象；历史同步必须走 SDK/RPC。
2. **长时间 `BLOCKING after_send` 不适合模拟打字。** 宿主会对 Hook 施加固定超时，累计等待会导致取消和丢尾段。
3. **裸 `OBSERVE` 后台发送也不够。** 必须持有 Task 强引用、处理卸载，并按聊天流建立等待门，否则新消息容易与旧补发交错。
4. **入站门有明确边界。** `chat.receive.before_process` 只约束后来到达的消息；同一轮内部续轮由 `maisaka.planner.before_request` 处理。
5. **planner Hook 只修补轻量消息载荷。** 它不重新调用分段 LLM，也不传输宿主运行时对象；分段仍在 replyer 文本 Hook 完成。
6. **发送链不能再调用分段 LLM。** 只消费 replyer 阶段的精确缓存，避免慢模型占住发送链，也避免误切非主回复文本。
7. **缓存匹配必须面对宿主后处理。** 精确命中缓存时跳过本次宿主正文后处理，避免错别字或其他改写使 hash 永久 miss。
8. **字节码缓存可能加载旧逻辑。** 若 traceback 行号与磁盘源码不一致，检查并清理对应 `__pycache__/plugin.cpython-*.pyc`。

---

## 10. 配置项

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 是否启用智能分段。 |
| `model` | `""` | 分段模型；空值交给宿主默认解析。 |
| `style` | `natural` | `natural` / `conservative` / `active`。 |
| `min_length` | `15` | 低于此字符数时不分段。 |
| `max_segments` | `8` | 最大分段数；可在插件配置中自定义，运行时不设置额外上限。 |
| `temperature` | `0.3` | 分段模型温度。 |
| `max_tokens` | `600` | 分段模型最大输出 token。 |
| `typing_enabled` | `true` | 是否为后续分段启用宿主模拟打字等待。 |

插件只控制是否启用模拟打字；具体速度仍属于宿主配置 `response_post_process.typing_speed`。

---

## 11. 维护约定

- 修改函数、配置字段或 Hook 名时，分别检查直接调用、字符串字面量、类型引用、动态导入和重导出。
- 核心逻辑变更必须同步更新具体业务断言测试；删除被测逻辑后，测试必须失败。
- 多步编辑后重新读取目标区域，并运行 `pytest` 验证。
- 优先纯插件方案；若需要宿主原生 reply 精确时序保证，应明确提出宿主 seam，而不是在插件文档中宣称已经具备。
- 提交遵循 Conventional Commits（中文），保持原子修改，不加 AI 署名。
