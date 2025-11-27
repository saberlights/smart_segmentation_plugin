# 智能分段插件 (Smart Segmentation Plugin)

## 功能说明

使用 LLM 智能切分回复文本，在语义自然停顿处进行分割，保持原文内容不变，使回复更加灵活自然。

### 主要特性
- **智能语义切分**: 基于 LLM 分析语义结构，在自然停顿处切分
- **内容完整性**: 仅进行切分操作，不修改原始内容
- **情绪标点保留**: 保留感叹号、问号、省略号等情绪相关标点
- **灵活配置**: 支持多种切分风格和段数限制
- **高性能**: 使用轻量级模型确保快速响应

## 使用方法

### 1. 禁用内置分段功能

编辑 `config/bot_config.toml`：

```toml
[response_splitter]
enable = false
```

### 2. 插件配置

插件配置文件位于 `plugins/smart_segmentation_plugin/config.toml`：

```toml
[plugin]
enabled = true
config_version = "1.2.0"

[segmentation]
enabled = true
style = "natural"
min_length = 20
max_segments = 8

[model_config]
model_name = "utils_small"
temperature = 0.3
max_tokens = 1024

[user_llm]
base_url = "https://api.example.com/v1"
api_key = "your-api-key"
model_name = "gpt-4-turbo"
temperature = 0.3
max_tokens = 1024
timeout_seconds = 10
```

### 3. 模型配置

确保在 `config/model_config.toml` 中配置了 `utils_small` 模型：

```toml
[model_task_config]
utils_small = ["gpt-4o-mini", "qwen-plus"]
```

## 配置参数

### segmentation 配置
- `enabled`: 是否启用智能分段功能
- `style`: 切分风格 (`natural`, `conservative`, `active`)
- `min_length`: 最小切分长度（字符数）
- `max_segments`: 最大切分段数

### model_config 配置
- `model_name`: 模型名称，可使用系统内置模型或设置为 `"user_llm"` 启用自定义模型
- `temperature`: 生成温度，范围 0.0-1.0
- `max_tokens`: 最大生成 token 数

### user_llm 配置（可选）
当 `model_config.model_name = "user_llm"` 时生效：
- `base_url`: 自定义 LLM API 基础 URL（必需）
- `api_key`: API 密钥（可选）
- `model_name`: 自定义模型名称（必需）
- `temperature`: 生成温度，范围 0.0-1.0
- `max_tokens`: 最大生成 token 数
- `timeout_seconds`: API 调用超时时间（秒）

## 工作原理

1. 在 `AFTER_LLM` 阶段拦截 LLM 生成的文本
2. 使用指定模型分析语义结构，确定切分位置
3. 删除切分点的普通标点（逗号、句号），保留情绪标点
4. 使用 `|||SPLIT|||` 标记切分点
5. 通过 Monkey Patch 替换 `process_llm_response()` 函数
6. 将切分后的文本作为独立消息分批发送

## 注意事项

1. 必须禁用内置分段功能以避免冲突
2. 建议使用轻量级模型（如 gpt-4o-mini、qwen-plus）以获得最佳性能
3. 默认最大段数为 8，可根据需要调整

## 故障排查

### 插件未生效
- 确认 `config.toml` 中 `enabled = true`
- 确认已禁用内置 `response_splitter`
- 检查日志中是否有 patch 成功的信息

### 切分效果问题
- 调整 `style` 参数：`natural`（默认）、`conservative`（保守）、`active`（活跃）
- 调整 `max_segments` 控制输出段数

## 许可证

GPL-v3.0-or-later

## 作者

久远