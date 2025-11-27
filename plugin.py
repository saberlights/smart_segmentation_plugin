"""
智能分段插件 - 使用 LLM 智能切分回复文本
"""
import json
from typing import List, Tuple, Type

from src.plugin_system import (
    BasePlugin,
    register_plugin,
    BaseEventHandler,
    EventType,
    MaiMessages,
    ConfigField,
    ComponentInfo,
)
from src.llm_models.utils_model import LLMRequest
from src.config.config import model_config
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
        return [s.strip() for s in text.split("|||SPLIT|||") if s.strip()]

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
        self.segmentation_llm = None

    def _init_llm(self):
        """延迟初始化 LLM"""
        if self.segmentation_llm is None:
            self.segmentation_llm = LLMRequest(
                model_set=model_config.model_task_config.utils_small,
                request_type="smart_segmentation"
            )

    async def execute(self, message: MaiMessages | None) -> Tuple[bool, bool, str | None, None, MaiMessages | None]:
        """执行智能分段"""
        if not message or not message.llm_response_content:
            return True, True, "无内容", None, message

        if not self.get_config("segmentation.enabled", True):
            return True, True, "未启用", None, message

        original = message.llm_response_content
        min_length = self.get_config("segmentation.min_length", 20)
        max_segments = self.get_config("segmentation.max_segments", 8)

        if len(original) < min_length:
            logger.debug(f"文本太短({len(original)}字)")
            return True, True, "文本太短", None, message

        self._init_llm()

        style = self.get_config("segmentation.style", "natural")
        style_guides = {
            "natural": "在话题转换、语气变化等重要停顿处切分。一个完整的意思放在一段里，不要过度切分。",
            "conservative": "尽量少切分，只在明显话题转换处切分。多个句子可以放在同一段。",
            "active": "切分更细致，在语气转换、情绪变化处也可切分，但仍要保持语义完整。"
        }

        prompt = f"""将文本切分成多段，模拟真人发消息节奏。

{style_guides.get(style, style_guides["natural"])}

重要规则：
- 不要在每个标点都切分！只在重要的语义停顿处切分
- 相关的句子应该保持在同一段
- 最多切分成 {max_segments} 段
- 切分时可以删除切分点的逗号、句号、顿号
- 保留感叹号、问号、省略号、波浪号等情绪标点
- 保持原文内容和语气不变

原文：{original}

返回 JSON 数组：["片段1", "片段2"]

示例：
原文："今天天气不错，阳光明媚。我们去公园玩吧！"
好的切分：["今天天气不错，阳光明媚", "我们去公园玩吧！"]
不好的切分：["今天天气不错", "阳光明媚", "我们去公园玩吧！"]（过度切分）"""

        try:
            result, _ = await self.segmentation_llm.generate_response_async(prompt)

            result = result.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            segments = json.loads(result)

            if not isinstance(segments, list) or not segments:
                raise ValueError("JSON 格式错误")

            message.modify_llm_response_content("|||SPLIT|||".join(segments))
            logger.info(f"智能切分为 {len(segments)} 段")

        except Exception as e:
            logger.error(f"智能切分失败: {e}")

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
        "segmentation": "智能分段配置"
    }

    config_schema: dict = {
        "plugin": {
            "name": ConfigField(type=str, default="smart_segmentation_plugin", description="插件名称"),
            "version": ConfigField(type=str, default="1.0.0", description="插件版本"),
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
        },
        "segmentation": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用智能分段"),
            "style": ConfigField(
                type=str,
                default="natural",
                description="切分风格：natural(自然), conservative(保守), active(活跃)"
            ),
            "min_length": ConfigField(type=int, default=20, description="启用分段的最小文本长度"),
            "max_segments": ConfigField(type=int, default=8, description="最大切分段数"),
        }
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        return [
            (SmartSegmentationHandler.get_handler_info(), SmartSegmentationHandler),
        ]
