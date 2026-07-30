# 智能分段插件 (Smart Segmentation Plugin)

使用 LLM 在语义自然的位置切分回复，把一段完整文本拆成多条连续消息，让 Bot 的发言节奏更像真人聊天，而不是一次性整段发出。

## 适合什么场景

- 希望回复看起来更像微信/QQ 的连续聊天，而不是公告式长段文本
- 想保留原文语义，只调整发送节奏
- 使用较小模型完成低成本分段任务

## 核心能力

- 自然切分：以真实网友按下发送键的口语意图为边界，不做机械平均分句
- 动态段数：根据文本长度和聊天风格提供软建议，极短消息不强拆，`max_segments` 只作为配置上限
- 节奏模拟：后续分段复用 MaiBot 本体的模拟打字速度，按真实打字节奏逐条发送
- 提示约束：要求模型只选择自然发送边界并尽量保持原意，插件不再二次校验措辞和标点
- 风格切换：`natural` / `conservative` / `active`
- 模型直连：`model` 支持 task 名、`[[models]].name`、`[[models]].model_identifier`
- 结果直用：合法的模型分段结果不再做字符级原文等价校验，避免细微文本差异导致分段失效
- 安全回退：LLM 超时、返回异常或 JSON 解析失败时，自动回退为原文直发

## 快速开始

### 1. 关闭宿主内置分段

必须先禁用宿主自带的 `response_splitter`，否则会和本插件重复切分。

编辑 `config/bot_config.toml`：

```toml
[response_splitter]
enable = false
```

### 2. 启用插件

编辑 `plugins/smart_segmentation_plugin/config.toml`：

```toml
[plugin]
config_version = "1.1.0"
version = "1.1.0"
enabled = true

[segmentation]
enabled = true
model = "doubao1.6"
style = "natural"
min_length = 8
max_segments = 8
typing_enabled = true
```

`typing_enabled = true` 时，后续分段的发送间隔由 MaiBot 本体配置控制：

```toml
[response_post_process]
typing_speed = 1.0
```

无需开启 `enable_response_post_process`，模拟打字仍会生效。将 `typing_enabled` 设为 `false` 会直接关闭插件后续段的模拟打字等待，但不影响顺序发送和历史同步。`typing_speed = 0` 会关闭普通文本的常规模拟打字等待；数值越大，等待越久。MaiBot 本体目前对“单个中文字符”保留固定等待的特殊分支，不受该值控制。

### 3. 准备模型

推荐使用响应快、价格低的小模型，例如 `gpt-4o-mini`、`qwen-plus`、`deepseek` 一类。这个任务只做文本切分，通常不需要大模型。

`model` 的解析顺序可以理解为：

1. 直接写 task 名，例如 `utils`
2. 直接写宿主 `model_config.toml` 里的 `[[models]].name`
3. 直接写 `[[models]].model_identifier`
4. 留空则交给宿主默认模型链路

## 配置说明

| 配置项 | 说明 | 建议值 |
| --- | --- | --- |
| `plugin.enabled` | 插件总开关 | `true` |
| `segmentation.enabled` | 智能分段开关 | `true` |
| `segmentation.model` | 分段所用模型，可填 task 名、模型名或模型标识 | 小模型即可 |
| `segmentation.style` | 切分风格：`natural` / `conservative` / `active` | `natural` |
| `segmentation.min_length` | 文本长度低于该值时不切分 | `8` 起步 |
| `segmentation.max_segments` | 单次回复最大切分段数，以 `config.toml` 配置为准 | `8`–`16` |
| `segmentation.typing_enabled` | 是否为后续分段启用宿主模拟打字等待 | `true` |

旧版的 `segmentation.delay_base`、`segmentation.delay_per_char`、`segmentation.delay_max` 已废弃；即使旧配置文件仍保留这些键，当前版本也会忽略它们。

### 风格差异

- `natural`：最像日常聊天，长短不均，适合作为默认风格
- `conservative`：尽量少切，单条更完整，适合偏稳重人设
- `active`：更容易拆成短句连发，适合活泼人设

插件会根据文本长度和风格向分段模型提供一个软段数区间，但不会要求模型凑满。语义上不适合分段时可以少于建议值，最终只受 `max_segments` 上限约束。

## 效果示例

原文：

```text
我今天去了那个新开的咖啡店，环境还不错。点了一杯拿铁，味道一般般吧，没有之前那家好喝。对了你上次推荐的那本书我看完了，超好看！
```

可能的分段结果：

```text
1. 我今天去了那个新开的咖啡店，环境还不错
2. 点了一杯拿铁，味道一般般吧，没有之前那家好喝
3. 对了你上次推荐的那本书我看完了
4. 超好看！
```

这个结果体现的是：

- 句号被去掉，但情绪标点被保留
- 短反应句可以单独成段
- 相关语义不会被逗号强行拆碎

## 命令

| 命令 | 说明 |
| --- | --- |
| `/smart_seg` | 直接切换当前运行时开关 |
| `/smart_seg on` | 开启智能分段 |
| `/smart_seg off` | 关闭智能分段 |
| `/smart_seg status` | 查看运行时状态和配置状态 |

命令只影响运行时开关，不会回写 `config.toml`。

## 工作流程

1. 在 `maisaka.replyer.after_response` 阶段调用分段 LLM，解析并规范化模型返回的 JSON 数组。首个请求超过 6 秒时会并发重试同一模型，内部总预算 20 秒；该 Hook 最多等待 25 秒，失败则保留原文
2. 将有效结果登记到 `(stream_id, 归一化文本 hash) -> [分段]` 的进程内缓存（默认 60 秒 TTL）
3. 在 `send_service.after_build_message` 阶段消费缓存：原发送链只发送首段，剩余分段进入待补发登记；未命中缓存时直接放行，不在发送链上再次调用 LLM
4. `send_service.after_send` 以 `OBSERVE` 模式快速创建后台任务，后续段通过 `ctx.send.text(..., typing=typing_enabled, sync_to_maisaka_history=True)` 顺序发送
5. `maisaka.planner.before_request` 会等待同一流的补发完成，并修补可能已经提前构建、只包含首段的 planner prompt，避免同一条用户消息被重复调用 `reply`
6. 同一聊天流收到新消息时，`chat.receive.before_process` 会等待该流的后台补发结束，避免新消息与旧回复交错；重入保护则避免补发内容再次被分段

这种设计的关键点是：**只信任 `maisaka.replyer.after_response` 产出的预分段缓存**。命令回执、memory/expression 文本和其他插件发送的文本不会进入缓存，因此不会被误切。首段仍走宿主原始发送链，引用和平台发送行为得以保留；启用模拟打字时，后台首个后续段的等待也给宿主留出完成首段历史同步的时间。

这是只修改插件时的工程折中，不是宿主原生 `reply` 工具内部的精确分段钩子。插件通过 planner Hook 等待并修补当前续轮的 prompt；若补发超过 Hook 超时，仍会记录失败并让宿主继续。

## 故障排查

### 插件没生效

按顺序检查：

1. `config.toml` 里 `plugin.enabled = true`
2. `config.toml` 里 `segmentation.enabled = true`
3. 宿主 `config/bot_config.toml` 里 `response_splitter.enable = false`
4. 日志中是否出现 `replyer.after_response 阶段预切分` 或 `命中 replyer 预分段缓存` 的记录
5. 如果只看到 `跳过：聊天流处于命令回执保护窗口`，说明 bot 刚执行完一条命令，命令后 1 秒内的回复会跳过分段
6. 如果回复模型出文没经过 `maisaka.replyer.after_response`（例如走了非 maisaka 的回复路径），插件会**直接放行原消息不分段** —— 这是设计如此，避免对非回复模型的输出（命令回执、memory/expression 文本等）误分段

### 切得太碎或太密

- 调大 `min_length`
- 调小 `max_segments`
- 把 `style` 改成 `conservative`
- 在宿主 `response_post_process.typing_speed` 的 `0` 到 `2` 范围内适当调大数值

### 切分不够明显

- 把 `style` 改成 `active`
- 调低 `min_length`
- 调大 `max_segments`

### JSON 解析失败

插件会自动回退为原文发送。合法的 JSON 数组会直接作为分段结果使用；若频繁解析失败，优先更换 JSON 输出更稳定的模型。

## 兼容性

- 插件版本：`1.1.0`
- 宿主最低版本：`1.0.0`
- SDK 支持范围：`1.0.0` - `2.99.99`

## 许可证

`GPL-v3.0-or-later`

## 作者

- 久远
- 仓库地址：<https://github.com/saberlights/smart_segmentation_plugin>
