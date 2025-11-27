"""
智能分段插件 - 使用 LLM 智能切分回复文本
"""
import json
import aiohttp
from typing import List, Tuple, Type

from src.plugin_system import (
    BasePlugin,
    register_plugin,
    BaseEventHandler,
    EventType,
    MaiMessages,
    ConfigField,
    ComponentInfo,
    llm_api,
)
from src.common.logger import get_logger

logger = get_logger("smart_segmentation")

# ============================================================================
# Monkey Patch
# ============================================================================

_original_process_llm_response = None
_patch_applied = False

def patched_process_llm_response(text: str, enable_splitter: bool = True, enable_chinese_typo: bool = True) -> list[str]:
    """识别智能分段分隔符并切分，否则使用原函数"""
    if "|||SPLIT|||" in text:
        logger.debug("检测到智能分段分隔符")
        try:
            return [s.strip() for s in text.split("|||SPLIT|||") if s.strip()]
        except Exception as e:
            logger.error(f"分段处理失败: {e}")
            return [text]

    return _original_process_llm_response(text, enable_splitter, enable_chinese_typo) if _original_process_llm_response else [text]

def apply_patch():
    """应用 monkey patch"""
    global _original_process_llm_response, _patch_applied

    if _patch_applied:
        return

    try:
        from src.chat.utils import utils
        from src.plugin_system.apis import generator_api

        _original_process_llm_response = utils.process_llm_response
        utils.process_llm_response = patched_process_llm_response
        generator_api.process_llm_response = patched_process_llm_response

        _patch_applied = True
        logger.info("✅ 已 patch process_llm_response")
    except Exception as e:
        logger.error(f"❌ Patch 失败: {e}")

apply_patch()

# ============================================================================
# Event Handler
# ============================================================================

class SmartSegmentationHandler(BaseEventHandler):
    """AFTER_LLM 阶段使用 LLM 智能切分文本"""

    event_type = EventType.AFTER_LLM
    handler_name = "smart_segmentation_handler"
    handler_description = "使用LLM智能切分回复文本"
    intercept_message = True
    weight = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._available_models = None

    async def _call_model(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int
    ) -> Tuple[bool, str, str, str]:
        """调用指定模型"""
        # 导入仅用于系统内置模型的模块
        from src.config.api_ada_configs import TaskConfig

        # 检查是否使用自定义user_llm
        if model_name == "user_llm":
            base_url = self.get_config("user_llm.base_url", "")
            api_key = self.get_config("user_llm.api_key", "")
            custom_model_name = self.get_config("user_llm.model_name", "")
            custom_temperature = self.get_config("user_llm.temperature", temperature)
            custom_max_tokens = self.get_config("user_llm.max_tokens", max_tokens)
            timeout_seconds = self.get_config("user_llm.timeout_seconds", 10)

            if not base_url or not custom_model_name:
                logger.error("user_llm配置不完整：缺少base_url或model_name")
                return False, "", "user_llm配置不完整，请检查配置文件", model_name

            try:
                logger.info(f"使用自定义user_llm模型: {custom_model_name} ({base_url})")

                # 构建OpenAI兼容的API请求
                api_url = f"{base_url.rstrip('/')}/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                }
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                payload = {
                    "model": custom_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": custom_temperature,
                    "max_tokens": custom_max_tokens,
                }

                # 执行HTTP请求
                timeout = aiohttp.ClientTimeout(total=timeout_seconds)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(api_url, json=payload, headers=headers) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"API调用失败，状态码: {response.status}, 错误: {error_text}")
                            return False, "", f"API调用失败: {response.status} - {error_text[:200]}", model_name

                        response_data = await response.json()

                        # 提取生成的文本
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            generated_text = response_data["choices"][0]["message"]["content"]
                            return True, generated_text, "", f"user_llm:{custom_model_name}"
                        else:
                            logger.error(f"API响应格式错误: {response_data}")
                            return False, "", "API响应格式错误，缺少choices字段", model_name

            except aiohttp.ClientTimeout:
                logger.error(f"user_llm模型调用超时 ({timeout_seconds}秒)")
                return False, "", f"模型调用超时 ({timeout_seconds}秒)", model_name
            except aiohttp.ClientError as e:
                logger.error(f"user_llm模型网络错误: {e}", exc_info=True)
                return False, "", f"网络错误: {str(e)}", model_name
            except Exception as e:
                logger.error(f"user_llm模型调用失败: {e}", exc_info=True)
                return False, "", str(e), model_name

        # 处理系统内置模型
        available_models = llm_api.get_available_models()

        # 如果model_name是任务配置名称（如"utils_small"），需要解析实际模型
        if model_name in ["utils", "utils_small", "replyer", "vlm", "voice", "tool_use", "planner", "embedding", "lpmm_entity_extract", "lpmm_rdf_build", "lpmm_qa"]:
            from src.config.config import model_config
            if hasattr(model_config.model_task_config, model_name):
                task_config = getattr(model_config.model_task_config, model_name)
                if task_config.model_list:
                    actual_model_name = task_config.model_list[0]
                    if actual_model_name in available_models:
                        logger.debug(f"使用任务模型 '{model_name}': {actual_model_name}")
                        temp = temperature if temperature is not None else task_config.temperature
                        tokens = max_tokens if max_tokens is not None else task_config.max_tokens
                        return await llm_api.generate_with_model(
                            prompt=prompt,
                            model_config=available_models[actual_model_name],
                            request_type="smart_segmentation",
                            temperature=temp,
                            max_tokens=tokens
                        )
                    else:
                        logger.warning(f"任务模型 '{model_name}' 配置的模型 '{actual_model_name}' 不可用")
                else:
                    logger.warning(f"任务模型 '{model_name}' 未配置模型列表")

        # 直接使用模型名称（已注册的模型）
        if model_name in available_models:
            logger.debug(f"使用指定模型: {model_name}")
            return await llm_api.generate_with_model(
                prompt=prompt,
                model_config=available_models[model_name],
                request_type="smart_segmentation",
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            logger.error(f"指定的模型 '{model_name}' 不可用。可用模型: {list(available_models.keys())}")
            logger.info("提示：请在 config/model_config.toml 中配置有效的模型")

            # 尝试回退到默认模型
            fallback_models = ["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku"]
            for fallback_name in fallback_models:
                if fallback_name in available_models:
                    logger.info(f"回退到默认模型: {fallback_name}")
                    return await llm_api.generate_with_model(
                        prompt=prompt,
                        model_config=available_models[fallback_name],
                        request_type="smart_segmentation"
                    )

            return False, "", f"模型 '{model_name}' 不可用且无回退选项", model_name

    async def execute(self, message: MaiMessages | None) -> Tuple[bool, bool, str | None, None, MaiMessages | None]:
        """执行智能分段"""
        if not message or not message.llm_response_content:
            logger.debug("消息为空或无LLM响应内容，跳过处理")
            return True, True, "无内容", None, message

        if not self.get_config("segmentation.enabled", True):
            logger.debug("智能分段功能未启用，跳过处理")
            return True, True, "未启用", None, message

        original = message.llm_response_content
        min_length = self.get_config("segmentation.min_length", 20)
        max_segments = self.get_config("segmentation.max_segments", 8)

        if len(original) < min_length:
            logger.debug(f"文本长度({len(original)}字)小于最小长度({min_length}字)，跳过分段")
            return True, True, "文本太短", None, message

        style = self.get_config("segmentation.style", "natural")
        style_guides = {
            "natural": "在自然停顿的地方切分，一个完整想法作为一段。",
            "conservative": "尽量少切分，只在明显话题转换处切分。",
            "active": "切分更细致，在语气转换、情绪变化处也可切分。"
        }

        prompt = f"""将文本切分成多段，模拟真人发消息节奏。

{style_guides.get(style, style_guides["natural"])}

规则：
- 在语义完整、自然停顿处切分
- 最多切分成 {max_segments} 段
- 删除切分点的逗号、句号、顿号
- 保留感叹号、问号、省略号、波浪号等情绪标点
- 其他内容不变

原文：{original}

返回 JSON 数组：["片段1", "片段2"]"""

        try:
            # 获取模型配置
            model_name = self.get_config("model_config.model_name", "utils_small")
            temperature = self.get_config("model_config.temperature", 0.3)
            max_tokens = self.get_config("model_config.max_tokens", 1024)

            success, result, reasoning, model_used = await self._call_model(
                prompt, model_name, temperature, max_tokens
            )

            if success:
                result = result.strip()
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()

                try:
                    segments = json.loads(result)

                    if not isinstance(segments, list) or not segments:
                        raise ValueError("JSON格式错误：返回的不是有效的数组")

                    if len(segments) > max_segments:
                        segments = segments[:max_segments]
                        logger.warning(f"分段数量({len(segments)})超过最大限制({max_segments})，已截断")

                    message.modify_llm_response_content("|||SPLIT|||".join(segments))
                    logger.info(f"智能切分为 {len(segments)} 段 (使用模型: {model_used})")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {e}, 原始响应: {result[:200]}...")
                    return True, True, "JSON解析失败", None, message
            else:
                logger.error(f"LLM调用失败: {reasoning}")
                return True, True, "LLM调用失败", None, message

        except Exception as e:
            logger.error(f"智能切分过程中发生异常: {e}", exc_info=True)
            return True, True, f"异常: {str(e)}", None, message

        return True, True, "完成", None, message

# ============================================================================
# Plugin Registration
# ============================================================================

@register_plugin
class SmartSegmentationPlugin(BasePlugin):
    """智能分段插件"""

    plugin_name: str = "smart_segmentation_plugin"
    enable_plugin: bool = True
    dependencies: List[str] = []
    python_dependencies: List[str] = []
    config_file_name: str = "config.toml"

    config_section_descriptions = {
        "plugin": "插件基本信息",
        "segmentation": "智能分段配置",
        "model_config": "模型配置",
        "user_llm": "自定义LLM配置（可选）"
    }

    config_schema: dict = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
            "config_version": ConfigField(type=str, default="1.2.0", description="配置文件版本"),
        },
        "segmentation": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用智能分段功能。设为 false 可临时禁用插件功能。"),
            "style": ConfigField(
                type=str,
                default="natural",
                description="""切分风格，控制文本切分的粒度。

# 可选值说明：
# - "natural"（推荐）：在自然停顿处切分，模拟真人发消息节奏
# - "conservative"：保守风格，尽量少切分，只在明显话题转换处切分
# - "active"：活跃风格，切分更细致，在语气转换、情绪变化处也可切分
""",
                choices=["natural", "conservative", "active"]
            ),
            "min_length": ConfigField(type=int, default=20, description="启用分段的最小文本长度（字符数）。小于此长度的文本不会进行智能分段，直接原样输出。"),
            "max_segments": ConfigField(type=int, default=8, description="最大切分段数。避免长文本切分过多导致刷屏，超过此数量会自动截断。"),
        },
        "model_config": {
            "model_name": ConfigField(
                type=str,
                default="utils_small",
                description="""指定用于智能分段的模型名称。

# 使用方式：
# 1. 使用系统内置模型（推荐）：
#    - "utils_small": 系统小模型（默认）
#    - "utils": 系统工具模型
#    - "replyer": 主回复模型
#
# 2. 使用自定义模型：
#    如果配置了 [user_llm] 节，可以在此处设置 model_name = "user_llm"
#    插件将直接使用 [user_llm] 节中配置的自定义模型
"""
            ),
            "temperature": ConfigField(
                type=float,
                default=0.3,
                description="模型生成温度。值范围 0.0-1.0，值越低输出越确定，越高越随机。"
            ),
            "max_tokens": ConfigField(
                type=int,
                default=1024,
                description="模型生成的最大token数。控制输出长度。"
            ),
        },
        "user_llm": {
            "base_url": ConfigField(
                type=str,
                default="",
                description="自定义LLM API的基础URL。",
                example="https://api.example.com/v1"
            ),
            "api_key": ConfigField(
                type=str,
                default="",
                description="自定义LLM API的密钥。",
                example="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            ),
            "model_name": ConfigField(
                type=str,
                default="",
                description="自定义LLM的模型名称。"
            ),
            "temperature": ConfigField(
                type=float,
                default=0.3,
                description="模型生成温度。值范围 0.0-1.0。"
            ),
            "max_tokens": ConfigField(
                type=int,
                default=1024,
                description="模型生成的最大token数。"
            ),
            "timeout_seconds": ConfigField(
                type=int,
                default=10,
                description="API调用超时时间（秒）。"
            ),
        }
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        return [
            (SmartSegmentationHandler.get_handler_info(), SmartSegmentationHandler),
        ]
