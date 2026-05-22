# 智能分段插件 (Smart Segmentation Plugin)

使用 LLM 在语义自然的位置切分回复，把一段完整文本拆成多条连续消息，让 Bot 的发言节奏更像真人聊天，而不是一次性整段发出。

## 适合什么场景

- 希望回复看起来更像微信/QQ 的连续聊天，而不是公告式长段文本
- 想保留原文语义，只调整发送节奏
- 使用较小模型完成低成本分段任务

## 核心能力

- 自然切分：按语义停顿拆分，不做机械平均分句
- 节奏模拟：支持短句单发、正文连发，以及按字数变化的发送间隔
- 末尾清理：自动去掉句号，保留问号、感叹号、省略号等情绪标点
- 风格切换：`natural` / `conservative` / `active`
- 模型直连：`model` 支持 task 名、`[[models]].name`、`[[models]].model_identifier`
- 安全回退：LLM 返回异常或 JSON 解析失败时，自动回退为原文直发

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
config_version = "1.0.0"
enabled = true

[segmentation]
enabled = true
model = "doubao1.6"
style = "natural"
min_length = 8
max_segments = 8
delay_base = 0.35
delay_per_char = 0.015
delay_max = 1.2
```

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
| `segmentation.max_segments` | 单次回复最大切分段数，避免刷屏 | `8` |
| `segmentation.delay_base` | 每段发送的基础延迟，单位秒 | `0.35` |
| `segmentation.delay_per_char` | 按文本长度附加延迟，单位秒/字符 | `0.015` |
| `segmentation.delay_max` | 单段最大发送延迟，单位秒 | `1.2` |

### 风格差异

- `natural`：最像日常聊天，长短不均，适合作为默认风格
- `conservative`：尽量少切，单条更完整，适合偏稳重人设
- `active`：更容易拆成短句连发，适合活泼人设

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

1. 在 `maisaka.replyer.after_response` 阶段拿到回复模型刚出的回复文本后立刻调用分段 LLM 切分，并把结果登记到一份 `(stream_id, 归一化文本 hash) -> [分段]` 的进程内缓存（默认 60 秒 TTL）
2. 在 `send_service.after_build_message` 阶段只按 `(stream_id, 出站文本 hash)` 去缓存里找；命中即零 LLM 调用、首段替换原消息后直接发送；**未命中直接放行原消息，发送链上不会再做任何 LLM 兜底**
3. 在 `send_service.after_send` 阶段按延迟节奏补发剩余段落
4. 使用重入保护，避免插件自己补发的消息再次被切分

这种设计的关键点是：**只信任 `maisaka.replyer.after_response` 这一个入口的产出**，因为它是宿主里唯一会在回复模型出文时触发的钩子。其他所有路径（命令回执、memory 模块文本、expression 模块广播、其他插件 `ctx.send.text` 的输出）都不会进入缓存，因此完全不会被错误地切碎。首条消息仍然走宿主原始发送链路，引用、会话上下文和平台发送行为不会被破坏；并且发送链上没有同步 LLM 调用，慢模型不会再静默吃掉分段。

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
- 增大 `delay_base` 或 `delay_per_char`

### 切分不够明显

- 把 `style` 改成 `active`
- 调低 `min_length`
- 调大 `max_segments`

### JSON 解析失败

插件会自动回退为原文发送，不会阻塞主流程。若频繁出现，优先更换 JSON 输出更稳定的模型。

## 兼容性

- 插件版本：`1.0.0`
- 宿主最低版本：`1.0.0`
- SDK 支持范围：`1.0.0` - `2.99.99`

## 许可证

`GPL-v3.0-or-later`

## 作者

- 久远
- 仓库地址：<https://github.com/saberlights/smart_segmentation_plugin>
