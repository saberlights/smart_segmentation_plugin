# 智能分段插件 (Smart Segmentation Plugin)

## 📝 功能说明

使用 LLM 模拟真人聊天的发消息节奏，将回复智能拆分成多条消息，去掉末尾句号、保留情绪标点，让 bot 的回复像真人在微信里一条一条发出来。

### 主要特点

- ✅ **真人节奏**：模拟手机聊天分条发送，短反应词单独一条，相关内容合并
- ✅ **去句号**：自动去掉每条消息末尾的句号，保留感叹号、问号、省略号等情绪标点
- ✅ **长短不均**：消息长短自然不均匀，避免机械的均匀切分
- ✅ **三种风格**：natural / conservative / active，适配不同人设
- ✅ **小模型友好**：任务简单，14B+ 的模型即可稳定运行

## 🚀 使用方法

### 1. 禁用内置分段（必须！）

编辑 `config/bot_config.toml`：

```toml
[response_splitter]
enable = false  # 必须禁用内置分段，否则会冲突
```

### 2. 启用插件

编辑 `plugins/smart_segmentation_plugin/config.toml`：

```toml
[plugin]
enabled = true

[segmentation]
enabled = true
style = "natural"    # 切分风格
min_length = 15      # 最小切分长度
max_segments = 8     # 最大段数
```

### 3. 配置模型

在 `config/model_config.toml` 中配置模型。插件注册默认使用 `utils` 任务配置；如果你在插件配置文件里显式填写 `model`，则优先使用该值：

```toml
[segmentation]
model = "gpt-4o-mini"  # 留空则使用 utils 任务配置
```

推荐使用小模型（gpt-4o-mini、qwen-plus、deepseek 等），分段任务对模型能力要求不高。

## ⚙️ 配置说明

### 切分风格 (style)

- **natural**（推荐）：像和朋友微信聊天一样自然地分条发送，有的消息短有的长，节奏随意
- **conservative**：偏沉稳的发消息风格，一条消息说比较完整的内容，不会频繁发短消息
- **active**：活泼的发消息风格，喜欢发短消息连击，反应词和正文分开发

### 最小长度 (min_length)

小于此长度的文本不会切分，默认 15 字符。

### 最大段数 (max_segments)

切分段数上限，避免长文刷屏，默认 8 段。

## 📊 效果示例

### 示例 1：日常闲聊

**原文：**
```
我今天去了那个新开的咖啡店，环境还不错。点了一杯拿铁，味道一般般吧，没有之前那家好喝。对了你上次推荐的那本书我看完了，超好看！
```

**分条结果：**
```
消息1: "我今天去了那个新开的咖啡店，环境还不错"
消息2: "点了一杯拿铁，味道一般般吧，没有之前那家好喝"
消息3: "对了你上次推荐的那本书我看完了"
消息4: "超好看！"
```

### 示例 2：短反应 + 正文

**原文：**
```
哈哈真的吗，那太好了！我还以为你不喜欢呢。下次我们一起去看电影吧，最近有个新片子挺有意思的。
```

**分条结果：**
```
消息1: "哈哈真的吗"
消息2: "那太好了！我还以为你不喜欢呢"
消息3: "下次我们一起去看电影吧，最近有个新片子挺有意思的"
```

**特点：**
- ✅ 去掉了末尾句号
- ✅ 保留了「！」情绪标点
- ✅ 短反应「哈哈真的吗」单独一条
- ✅ 相关内容没有被逗号过度拆分

## 🔧 命令

| 命令 | 说明 |
|------|------|
| `/smart_seg` | 切换开关状态 |
| `/smart_seg on` | 开启智能分段 |
| `/smart_seg off` | 关闭智能分段 |
| `/smart_seg status` | 查看当前状态 |

## 🔧 工作原理

1. **AFTER_LLM 阶段**：拦截 LLM 生成的文本
2. **智能分析**：使用 LLM 模拟真人聊天节奏，决定分条方式
3. **标点处理**：去掉末尾句号，保留情绪标点
4. **添加分隔符**：用 `|||SPLIT|||` 标记切分点
5. **Monkey Patch**：替换 `process_llm_response()` 识别分隔符并切分
6. **分批发送**：每个片段作为独立消息发送

## ⚠️ 注意事项

1. **必须禁用内置分段**：在 `config/bot_config.toml` 中设置 `response_splitter.enable = false`
2. **模型选择**：14B+ 的小模型即可，推荐 gpt-4o-mini、qwen-plus 等
3. **段数限制**：默认最多 8 段，可通过 `max_segments` 调整

## 🐛 故障排查

### 插件未生效

1. 检查 `config.toml` 中 `enabled = true`
2. 确认已禁用内置 `response_splitter`（在 `config/bot_config.toml` 中）
3. 查看日志是否有 `✅ 已 patch process_llm_response` 信息

### 切分效果不理想

- 调整 `style` 参数：`natural`（推荐）、`conservative`（少切分）、`active`（多切分）
- 调整 `max_segments` 控制最大段数

### JSON 解析失败

- 小模型偶尔输出格式不规范的 JSON，插件会自动 fallback 到原文不切分
- 如果频繁失败，考虑换一个 JSON 输出更稳定的模型

## 📄 许可证

GPL-v3.0-or-later

## 👤 作者

久远
