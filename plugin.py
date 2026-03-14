"""
智能分段插件 - 使用 LLM 智能切分回复文本
"""
import json
from typing import List, Tuple, Type

from src.plugin_system import (
    BasePlugin,
    BaseCommand,
    register_plugin,
    BaseEventHandler,
    EventType,
    MaiMessages,
    ConfigField,
    ComponentInfo,
    CommandInfo,
    send_api,
)
from src.llm_models.utils_model import LLMRequest
from src.config.config import model_config
from src.config.api_ada_configs import TaskConfig
from src.common.logger import get_logger

logger = get_logger("smart_segmentation")

# 运行时开关状态（独立于配置文件，用于命令动态控制）
_runtime_enabled = True

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
            model_name = self.get_config("segmentation.model", "")
            if model_name:
                # 使用配置中指定的模型
                task_config = TaskConfig(model_list=[model_name])
            else:
                # 默认使用 utils 任务配置
                task_config = model_config.model_task_config.utils
            self.segmentation_llm = LLMRequest(
                model_set=task_config,
                request_type="smart_segmentation"
            )

    async def execute(self, message: MaiMessages | None) -> Tuple[bool, bool, str | None, None, MaiMessages | None]:
        """执行智能分段"""
        if not message or not message.llm_response_content:
            return True, True, "无内容", None, message

        if not self.get_config("segmentation.enabled", True):
            return True, True, "未启用", None, message

        if not _runtime_enabled:
            return True, True, "已通过命令关闭", None, message

        original = message.llm_response_content
        min_length = self.get_config("segmentation.min_length", 20)
        max_segments = self.get_config("segmentation.max_segments", 8)

        if len(original) < min_length:
            logger.debug(f"文本太短({len(original)}字)")
            return True, True, "文本太短", None, message

        self._init_llm()

        style = self.get_config("segmentation.style", "natural")
        style_guides = {
            "natural": "像和朋友微信聊天一样自然地分条发送。有的消息短有的长，节奏随意。",
            "conservative": "偏沉稳的发消息风格，一条消息说比较完整的内容，不会频繁发短消息。",
            "active": "活泼的发消息风格，喜欢发短消息连击，反应词和正文分开发。"
        }

        prompt = f"""你正在模拟一个人用手机聊天。下面是 ta 想说的内容，请把它分成几条消息，就像真人会怎么一条一条发出来那样。

{style_guides.get(style, style_guides["natural"])}

规则：
- 去掉每条消息末尾的句号「。」，真人聊天很少用句号结尾
- 保留感叹号、问号、省略号、波浪号等有情绪的标点
- 不要每个逗号都拆开，相关的内容放在一条里
- 消息长短可以不均匀
- 最多分成 {max_segments} 条

原文：{original}

返回 JSON 数组，如 ["消息1", "消息2"]

示例：
原文："我今天去了那个新开的咖啡店，环境还不错。点了一杯拿铁，味道一般般吧，没有之前那家好喝。对了你上次推荐的那本书我看完了，超好看！"
分条：["我今天去了那个新开的咖啡店，环境还不错", "点了一杯拿铁，味道一般般吧，没有之前那家好喝", "对了你上次推荐的那本书我看完了", "超好看！"]

原文："哈哈真的吗，那太好了！我还以为你不喜欢呢。下次我们一起去看电影吧，最近有个新片子挺有意思的。"
分条：["哈哈真的吗", "那太好了！我还以为你不喜欢呢", "下次我们一起去看电影吧，最近有个新片子挺有意思的"]

原文："嗯...这个问题有点复杂，我想想怎么说。简单来说就是你需要先把环境配好，然后再安装依赖。如果还有问题可以再问我。"
分条：["嗯...这个问题有点复杂", "我想想怎么说", "简单来说就是你需要先把环境配好，然后再安装依赖", "如果还有问题可以再问我"]"""

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
# Command
# ============================================================================

class SmartSegmentationCommand(BaseCommand):
    """通过命令开关智能分段"""

    command_name: str = "smart_seg"
    command_description: str = "开关智能分段功能"
    command_pattern: str = r"(?P<seg_cmd>^/smart_seg(\s+(on|off|status))?\s*$)"

    async def execute(self) -> Tuple[bool, str, int]:
        global _runtime_enabled

        cmd_text = self.matched_groups.get("seg_cmd", "").strip()
        parts = cmd_text.split()

        if len(parts) == 1:
            # 无参数，切换状态
            _runtime_enabled = not _runtime_enabled
            state = "开启" if _runtime_enabled else "关闭"
            await self.send_text(f"智能分段已{state}")
            logger.info(f"智能分段已通过命令切换为: {state}")
            return True, f"智能分段已{state}", 2

        action = parts[1]
        if action == "on":
            _runtime_enabled = True
            await self.send_text("智能分段已开启")
            logger.info("智能分段已通过命令开启")
            return True, "智能分段已开启", 2
        elif action == "off":
            _runtime_enabled = False
            await self.send_text("智能分段已关闭")
            logger.info("智能分段已通过命令关闭")
            return True, "智能分段已关闭", 2
        elif action == "status":
            state = "开启" if _runtime_enabled else "关闭"
            await self.send_text(f"智能分段当前状态: {state}")
            return True, f"状态: {state}", 2

        await self.send_text("用法: /smart_seg [on|off|status]")
        return False, "参数错误", 2


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
            "model": ConfigField(
                type=str,
                default="",
                description="使用的模型名称（需在model_config中配置），留空则使用utils任务配置"
            ),
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
            (SmartSegmentationCommand.get_command_info(), SmartSegmentationCommand),
        ]
