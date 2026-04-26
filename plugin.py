"""智能分段插件。"""

from collections.abc import Callable
from typing import Any

import asyncio
from contextlib import contextmanager
import json
import logging
import os
import random
import re
import time
import tomllib

from maibot_sdk import Command, EventHandler, MaiBotPlugin
from maibot_sdk.types import EventType
from src.config.model_configs import TaskConfig
from src.llm_models.utils_model import LLMOrchestrator

_CUSTOM_HOOK_HANDLER_ATTR = "__smart_segmentation_custom_hook_handler__"

try:
    from maibot_sdk import Field, HookHandler, PluginConfigBase
    from maibot_sdk.types import HookMode

    _SDK_HOOK_HANDLER_AVAILABLE = True
except ImportError:
    from pydantic import BaseModel, Field

    PluginConfigBase = BaseModel
    _SDK_HOOK_HANDLER_AVAILABLE = False

    class HookMode:
        """当前宿主缺失 HookMode 时的最小兼容定义。"""

        BLOCKING = "blocking"
        OBSERVE = "observe"

    def HookHandler(
        hook_name: str,
        name: str = "",
        description: str = "",
        mode: Any = None,
        **metadata: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """在缺失 SDK HookHandler 时，自行声明 host 可识别的 hook_handler 组件。"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            setattr(
                func,
                _CUSTOM_HOOK_HANDLER_ATTR,
                {
                    "hook": str(hook_name or "").strip(),
                    "name": str(name or hook_name.replace(".", "_") or func.__name__).strip(),
                    "description": description,
                    "mode": HookMode.OBSERVE if mode == HookMode.OBSERVE else HookMode.BLOCKING,
                    "metadata": dict(metadata),
                },
            )
            return func

        return decorator

logger = logging.getLogger("plugin.smart_segmentation")

_runtime_enabled = True
_pending_follow_up_segments: dict[str, dict[str, Any]] = {}
_stream_resend_guards: dict[str, int] = {}
_active_command_streams: dict[str, int] = {}
_recent_command_stream_expiries: dict[str, float] = {}
_SEGMENT_DELIMITER = "|||SPLIT|||"
_PREPARED_SEGMENTS_MARKER = "[smart_segmentation_plugin:segments]"
_PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY = "smart_segmentation_prepared_segments"
_REPLYER_SEGMENT_MARKERS = (
    "[smart_segmentation_plugin:replyer]",
    "[replyer]",
)
# 命令回执的同步发送链路已被 _active_command_streams 精确覆盖，这里的窗口只用来兜住
# 命令 hook 与 send_service 之间可能存在的轻微异步抖动（毫秒到秒级），过长会把命令后的
# 正常业务主回复一起误伤——曾出现 90s 窗口导致重启后首个 @ 回复不分段的问题。
_COMMAND_REPLY_GRACE_SECONDS = 2.0


class _PinnedTaskLLMOrchestrator(LLMOrchestrator):
    """用于在插件内固定使用指定模型列表的轻量调度器。"""

    def __init__(self, task_config: TaskConfig, request_type: str = "") -> None:
        self._pinned_task_config = task_config
        super().__init__(task_name="planner", request_type=request_type)

    def _get_task_config_or_raise(self) -> TaskConfig:
        return self._pinned_task_config

    def _refresh_task_config(self) -> TaskConfig:
        latest = self._pinned_task_config
        if latest is not self.model_for_task:
            self.model_for_task = latest
        if list(self.model_usage.keys()) != latest.model_list:
            self.model_usage = {model: self.model_usage.get(model, (0, 0, 0)) for model in latest.model_list}
        return self.model_for_task


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    name: str = Field(default="smart_segmentation_plugin", description="插件名称")
    config_version: str = Field(default="1.0.0", description="配置文件版本")
    version: str = Field(default="1.0.0", description="插件版本")
    enabled: bool = Field(default=True, description="是否启用插件")


class SegmentationSectionConfig(PluginConfigBase):
    """智能分段配置。"""

    enabled: bool = Field(default=True, description="是否启用智能分段")
    model: str = Field(default="", description="分段使用的模型名称，留空则使用宿主默认模型")
    style: str = Field(default="natural", description="分段风格：natural / conservative / active")
    min_length: int = Field(default=15, description="启用分段的最小文本长度")
    max_segments: int = Field(default=8, description="最大分段数量")
    temperature: float = Field(default=0.3, description="分段模型温度")
    max_tokens: int = Field(default=600, description="分段模型最大输出 token")
    delay_base: float = Field(default=0.35, description="分段消息基础发送间隔（秒）")
    delay_per_char: float = Field(default=0.015, description="按字数增加的发送间隔（秒）")
    delay_max: float = Field(default=1.2, description="单段消息最大发送间隔（秒）")


class SmartSegmentationConfig(PluginConfigBase):
    """插件完整配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    segmentation: SegmentationSectionConfig = Field(default_factory=SegmentationSectionConfig)


def _merge_config_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置，优先使用运行时覆盖值。"""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_config_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def _find_host_model_config_path() -> str:
    """定位宿主 model_config.toml。"""
    plugin_dir = os.path.dirname(__file__)
    host_root = os.path.abspath(os.path.join(plugin_dir, "..", ".."))
    return os.path.join(host_root, "config", "model_config.toml")


def _load_host_model_config_fallback() -> dict[str, Any]:
    """回退读取宿主 model_config.toml，用于把底层模型名映射到 task 名。"""
    model_config_path = _find_host_model_config_path()
    if not os.path.isfile(model_config_path):
        return {}

    try:
        with open(model_config_path, "rb") as config_file:
            config_data = tomllib.load(config_file)
        return config_data if isinstance(config_data, dict) else {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("读取宿主 model_config.toml 失败，无法做模型名映射: %s", exc)
        return {}


def _normalize_model_alias_candidates(configured_name: str, host_model_config: dict[str, Any]) -> list[str]:
    """把 task 名、模型别名和 model_identifier 归一成候选匹配值。"""
    normalized_name = str(configured_name or "").strip()
    if not normalized_name:
        return []

    candidate_names = [normalized_name]
    raw_models = host_model_config.get("models")
    if not isinstance(raw_models, list):
        return candidate_names

    for model_item in raw_models:
        if not isinstance(model_item, dict):
            continue
        model_alias = str(model_item.get("name", "") or "").strip()
        model_identifier = str(model_item.get("model_identifier", "") or "").strip()
        if normalized_name not in {model_alias, model_identifier}:
            continue
        if model_alias and model_alias not in candidate_names:
            candidate_names.append(model_alias)
        if model_identifier and model_identifier not in candidate_names:
            candidate_names.append(model_identifier)
    return candidate_names


def _resolve_model_alias_from_host_model_config(configured_name: str, host_model_config: dict[str, Any]) -> str:
    """把模型别名或 model_identifier 解析为宿主 models.name。"""
    normalized_name = str(configured_name or "").strip()
    if not normalized_name:
        return ""

    raw_models = host_model_config.get("models")
    if not isinstance(raw_models, list):
        return ""

    for model_item in raw_models:
        if not isinstance(model_item, dict):
            continue
        model_alias = str(model_item.get("name", "") or "").strip()
        model_identifier = str(model_item.get("model_identifier", "") or "").strip()
        if normalized_name in {model_alias, model_identifier}:
            return model_alias or normalized_name
    return ""


def _get_nested_config_value(config_data: dict[str, Any], key: str, default: Any = None) -> Any:
    """按点分路径读取配置。"""
    current: Any = config_data
    for part in str(key or "").split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _strip_thinking_content(text: str) -> str:
    """移除 thinking 标签及其内容，只保留最终可见正文。"""
    if not text:
        return ""

    cleaned_text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned_text = re.sub(r"</?thinking>", "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text.strip()


def _extract_json_array_text(raw_text: str) -> str:
    """从模型返回中提取 JSON 数组文本。"""
    result_text = str(raw_text or "").strip()
    if "```json" in result_text:
        return result_text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in result_text:
        return result_text.split("```", 1)[1].split("```", 1)[0].strip()

    start = result_text.find("[")
    end = result_text.rfind("]")
    if start != -1 and end != -1 and start < end:
        return result_text[start : end + 1]
    return result_text


def _encode_prepared_segments(segments: list[str]) -> str:
    """将已分好的主回复编码为内部传递文本。"""
    return _SEGMENT_DELIMITER.join(segment.strip() for segment in segments if segment.strip())


def _decode_prepared_segments(text: str) -> list[str]:
    """从内部传递文本中提取已准备好的分段。"""
    if _SEGMENT_DELIMITER not in text:
        return []
    return [segment.strip() for segment in text.split(_SEGMENT_DELIMITER) if segment.strip()]


def _build_prepared_segments_marker_text(segments: list[str]) -> str:
    """构建带有插件标记的预分段文本。"""
    encoded_segments = _encode_prepared_segments(segments)
    if not encoded_segments:
        return ""
    return f"{_PREPARED_SEGMENTS_MARKER}{encoded_segments}"


def _get_top_level_additional_data(message: dict[str, Any]) -> dict[str, Any]:
    """提取事件消息上的 additional_data。"""
    additional_data = message.get("additional_data")
    if not isinstance(additional_data, dict):
        return {}
    return additional_data


def _inject_prepared_segments_into_message(message: dict[str, Any], segments: list[str]) -> None:
    """将预分段结果同时写入文本标记与 additional_data，避免后续链路改写正文后丢失。"""
    encoded_segments = _encode_prepared_segments(segments)
    if not encoded_segments:
        return

    marked_text = f"{_PREPARED_SEGMENTS_MARKER}{encoded_segments}"
    message["llm_response_content"] = marked_text

    additional_data = message.get("additional_data")
    if not isinstance(additional_data, dict):
        additional_data = {}
        message["additional_data"] = additional_data
    additional_data[_PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY] = encoded_segments

    message_info = message.get("message_info")
    if not isinstance(message_info, dict):
        message_info = {}
        message["message_info"] = message_info
    additional_config = message_info.get("additional_config")
    if not isinstance(additional_config, dict):
        additional_config = {}
        message_info["additional_config"] = additional_config
    additional_config[_PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY] = encoded_segments


def _extract_marked_segments(text: str) -> tuple[list[str], str]:
    """从标记文本中解码分段，返回分段列表与命中的标记来源。"""
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return [], ""

    if normalized_text.startswith(_PREPARED_SEGMENTS_MARKER):
        payload = normalized_text[len(_PREPARED_SEGMENTS_MARKER) :].strip()
        segments = _decode_prepared_segments(payload)
        return segments, "prepared" if segments else ""

    for marker in _REPLYER_SEGMENT_MARKERS:
        if normalized_text.startswith(marker):
            payload = normalized_text[len(marker) :].strip()
            segments = _decode_prepared_segments(payload)
            return segments, "replyer" if segments else ""

    replyer_match = re.match(r"^\[replyer(?::[^\]]*)?\](.+)$", normalized_text, flags=re.IGNORECASE | re.DOTALL)
    if replyer_match:
        segments = _decode_prepared_segments(replyer_match.group(1).strip())
        return segments, "replyer" if segments else ""

    return [], ""


def _extract_prepared_segments_from_outbound_message(
    message: dict[str, Any],
    display_message: str = "",
) -> tuple[list[str], str]:
    """从发送链消息里提取预分段标记（含 replyer 标记兜底）。"""
    top_level_additional_data = _get_top_level_additional_data(message)
    encoded_segments = str(top_level_additional_data.get(_PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY, "") or "").strip()
    if encoded_segments:
        segments = _decode_prepared_segments(encoded_segments)
        if segments:
            return segments, "additional_data"

    additional_config = _get_outbound_additional_config(message)
    encoded_segments = str(additional_config.get(_PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY, "") or "").strip()
    if encoded_segments:
        segments = _decode_prepared_segments(encoded_segments)
        if segments:
            return segments, "additional_config"

    candidate_texts = [
        display_message,
        str(message.get("display_message", "") or ""),
        str(message.get("processed_plain_text", "") or ""),
        _extract_plain_text_outbound_message(message, display_message),
        str(additional_config.get("replyer_display_message", "") or ""),
        str(additional_config.get("replyer_text", "") or ""),
        str(additional_config.get("replyer_marker", "") or ""),
    ]
    for candidate_text in candidate_texts:
        segments, source = _extract_marked_segments(candidate_text)
        if segments:
            return segments, source
    return [], ""


def _normalize_segments(segments: Any, *, max_segments: int) -> list[str]:
    """规范化模型返回的分段结果。"""
    if not isinstance(segments, list):
        raise ValueError("模型返回的分段结果不是列表")

    normalized_segments = [str(segment).strip() for segment in segments if str(segment).strip()]
    if not normalized_segments:
        raise ValueError("模型返回的分段结果为空")

    return normalized_segments[:max_segments]


def _extract_plain_text_outbound_message(message: dict[str, Any], display_message: str = "") -> str:
    """从发送链消息中提取可安全分段的纯文本。"""
    raw_components = message.get("raw_message")
    if not isinstance(raw_components, list):
        return str(display_message or "").strip()

    parts: list[str] = []
    for component in raw_components:
        if not isinstance(component, dict):
            return ""

        component_type = str(component.get("type", "") or "").strip().lower()
        if component_type == "text":
            text = str(component.get("data", "") or "")
            if text:
                parts.append(text)
            continue

        # 只处理纯文本消息，避免打乱图片/语音/艾特等结构。
        return ""

    text = "".join(parts).strip()
    if text:
        return text
    return str(display_message or "").strip()


def _replace_outbound_text(message: dict[str, Any], new_text: str) -> dict[str, Any]:
    """将发送链消息改写为新的单条纯文本。"""
    updated_message = dict(message)
    updated_message["raw_message"] = [{"type": "text", "data": new_text}]
    updated_message["processed_plain_text"] = new_text
    updated_message["display_message"] = new_text
    message_info = updated_message.get("message_info")
    if isinstance(message_info, dict):
        updated_message["message_info"] = dict(message_info)
        additional_config = message_info.get("additional_config")
        if isinstance(additional_config, dict):
            sanitized_additional_config = dict(additional_config)
            sanitized_additional_config.pop(_PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY, None)
            updated_message["message_info"]["additional_config"] = sanitized_additional_config
    return updated_message


def _get_outbound_additional_config(message: dict[str, Any]) -> dict[str, Any]:
    """提取出站消息 additional_config。"""
    message_info = message.get("message_info")
    if not isinstance(message_info, dict):
        return {}

    additional_config = message_info.get("additional_config")
    if not isinstance(additional_config, dict):
        return {}
    return additional_config


def _should_apply_after_build_fallback(message: dict[str, Any], *, storage_message: bool = True) -> bool:
    """对大多数纯文本出站消息启用发送前兜底分段。"""
    additional_config = _get_outbound_additional_config(message)

    selected_expressions = additional_config.get("selected_expressions")
    if isinstance(selected_expressions, list) and selected_expressions:
        return True

    reply_to = str(message.get("reply_to", "") or "").strip()
    if reply_to:
        return True

    for marker_key in ("replyer_display_message", "replyer_text", "replyer_marker"):
        marker_value = str(additional_config.get(marker_key, "") or "").strip()
        if marker_value:
            return True

    # 大量插件命令帮助/进度/报错文本会显式标记为不入库，
    # 这类主动输出不应参与智能分段。
    if not storage_message:
        return False

    # 宿主链路近期出现了 reply_to / replyer 标记缺失的情况，
    # 这里回退到“只要是纯文本出站消息就尝试分段”，避免正常主回复被跳过。
    return True


def _normalize_pending_lookup_keys(*keys: Any) -> list[str]:
    """规范化待补发分段的查找键，并去重。"""
    normalized_keys: list[str] = []
    for key in keys:
        normalized_key = str(key or "").strip()
        if normalized_key and normalized_key not in normalized_keys:
            normalized_keys.append(normalized_key)
    return normalized_keys


def _build_follow_up_tracking_key(*, stream_id: str, timestamp: Any, visible_text: str) -> str:
    """构建不受平台回执 message_id 改写影响的稳定追踪键。"""
    normalized_stream_id = str(stream_id or "").strip()
    normalized_timestamp = str(timestamp or "").strip()
    normalized_text = _strip_thinking_content(str(visible_text or "").strip())
    if not normalized_stream_id or not normalized_timestamp or not normalized_text:
        return ""

    return "\x1f".join((normalized_stream_id, normalized_timestamp, normalized_text))


def _register_pending_follow_up_segments(*, lookup_keys: list[str], pending_data: dict[str, Any]) -> None:
    """用多种查找键登记待补发的分段消息。"""
    tracked_pending_data = dict(pending_data)
    tracked_pending_data["lookup_keys"] = list(lookup_keys)
    for lookup_key in lookup_keys:
        _pending_follow_up_segments[lookup_key] = tracked_pending_data


def _pop_pending_follow_up_segments(*lookup_keys: Any) -> dict[str, Any] | None:
    """按任一查找键提取并清理待补发分段。"""
    normalized_lookup_keys = _normalize_pending_lookup_keys(*lookup_keys)
    for lookup_key in normalized_lookup_keys:
        pending_data = _pending_follow_up_segments.get(lookup_key)
        if pending_data is None:
            continue

        raw_cleanup_keys = pending_data.get("lookup_keys")
        if isinstance(raw_cleanup_keys, list):
            cleanup_keys = _normalize_pending_lookup_keys(*raw_cleanup_keys)
        else:
            cleanup_keys = [lookup_key]

        for cleanup_key in cleanup_keys:
            _pending_follow_up_segments.pop(cleanup_key, None)
        return pending_data

    return None


def _resolve_pending_follow_up_segments(*, message_id: Any, tracking_key: Any) -> dict[str, Any] | None:
    """优先按当前 message_id 查找，未命中时再回退到稳定追踪键。"""
    normalized_message_id = str(message_id or "").strip()
    if normalized_message_id:
        pending_data = _pop_pending_follow_up_segments(normalized_message_id)
        if pending_data is not None:
            return pending_data

    normalized_tracking_key = str(tracking_key or "").strip()
    if normalized_tracking_key:
        pending_data = _pop_pending_follow_up_segments(normalized_tracking_key)
        if pending_data is not None and normalized_message_id:
            logger.info("智能分段待补发消息未命中当前 message_id，已回退到稳定追踪键")
        return pending_data

    return None


def _mark_command_stream_active(stream_id: Any) -> None:
    """标记当前聊天流正在执行命令。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return
    _active_command_streams[normalized_stream_id] = _active_command_streams.get(normalized_stream_id, 0) + 1
    _recent_command_stream_expiries[normalized_stream_id] = time.monotonic() + _COMMAND_REPLY_GRACE_SECONDS


def _mark_command_stream_inactive(stream_id: Any) -> None:
    """清理当前聊天流的命令执行标记。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return
    remaining = _active_command_streams.get(normalized_stream_id, 0) - 1
    if remaining > 0:
        _active_command_streams[normalized_stream_id] = remaining
    else:
        _active_command_streams.pop(normalized_stream_id, None)
    # 故意不在命令结束时续期：before_execute 设定的短窗口足够覆盖回执同步发送，
    # 再次续期会把命令结束后的正常业务主回复一起挡住。


def _is_command_stream_active(stream_id: Any) -> bool:
    """判断当前聊天流是否正处于命令执行中。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return False
    return _active_command_streams.get(normalized_stream_id, 0) > 0


def _get_command_stream_grace_remaining(stream_id: Any) -> float | None:
    """返回保护窗口剩余秒数；未命中返回 None。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return None

    expires_at = _recent_command_stream_expiries.get(normalized_stream_id)
    if expires_at is None:
        return None

    remaining = expires_at - time.monotonic()
    if remaining <= 0:
        _recent_command_stream_expiries.pop(normalized_stream_id, None)
        return None
    return remaining


def _is_stream_guarded(stream_id: str) -> bool:
    """判断当前流是否处于插件补发保护期。"""
    return _stream_resend_guards.get(stream_id, 0) > 0


@contextmanager
def _guard_stream_resend(stream_id: str):
    """在插件自行补发分段消息时临时关闭二次分段。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        yield
        return

    _stream_resend_guards[normalized_stream_id] = _stream_resend_guards.get(normalized_stream_id, 0) + 1
    try:
        yield
    finally:
        remaining = _stream_resend_guards.get(normalized_stream_id, 0) - 1
        if remaining > 0:
            _stream_resend_guards[normalized_stream_id] = remaining
        else:
            _stream_resend_guards.pop(normalized_stream_id, None)


def _collect_custom_hook_components(plugin_instance: MaiBotPlugin) -> list[dict[str, Any]]:
    """收集当前文件自定义声明的 hook_handler 组件。"""
    components: list[dict[str, Any]] = []
    for attr_name in dir(plugin_instance):
        try:
            attr = getattr(plugin_instance, attr_name)
        except Exception:
            continue
        if not callable(attr):
            continue

        hook_info = getattr(attr, _CUSTOM_HOOK_HANDLER_ATTR, None)
        if not isinstance(hook_info, dict):
            continue

        component_name = str(hook_info.get("name", "") or attr_name).strip()
        if not component_name:
            continue

        component_metadata = {
            "description": str(hook_info.get("description", "") or "").strip(),
            "enabled": True,
            "hook": str(hook_info.get("hook", "") or "").strip(),
            "mode": str(hook_info.get("mode", HookMode.BLOCKING) or HookMode.BLOCKING).strip().lower(),
            "handler_name": attr_name,
            **(hook_info.get("metadata") if isinstance(hook_info.get("metadata"), dict) else {}),
        }
        components.append(
            {
                "name": component_name,
                "type": "hook_handler",
                "metadata": component_metadata,
            }
        )
    return components


class SmartSegmentationPlugin(MaiBotPlugin):
    """使用 LLM 对主回复进行智能分段，并直接分条发送。"""

    config_model = SmartSegmentationConfig

    def get_components(self) -> list[dict[str, Any]]:
        """补充当前 SDK 未导出的 hook_handler 声明。"""
        components = list(super().get_components())
        if _SDK_HOOK_HANDLER_AVAILABLE:
            return components
        components.extend(_collect_custom_hook_components(self))
        return components

    async def on_load(self) -> None:
        """处理插件加载。"""
        _pending_follow_up_segments.clear()
        _stream_resend_guards.clear()
        _active_command_streams.clear()
        _recent_command_stream_expiries.clear()
        if not _SDK_HOOK_HANDLER_AVAILABLE:
            logger.info("当前 maibot_sdk 未导出 HookHandler，智能分段已启用内置 hook_handler 声明兼容")

    async def on_unload(self) -> None:
        """处理插件卸载。"""
        _pending_follow_up_segments.clear()
        _stream_resend_guards.clear()
        _active_command_streams.clear()
        _recent_command_stream_expiries.clear()

    def _load_local_config_fallback(self) -> dict[str, Any]:
        """回退读取插件目录下的 `config.toml`。"""
        config_path = os.path.join(os.path.dirname(__file__), "config.toml")
        if not os.path.isfile(config_path):
            return {}

        try:
            with open(config_path, "rb") as config_file:
                config_data = tomllib.load(config_file)
            return config_data if isinstance(config_data, dict) else {}
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.error("读取插件配置失败: %s", exc)
            return {}

    async def _get_plugin_config(self) -> dict[str, Any]:
        """获取插件完整配置。"""
        local_config = self._load_local_config_fallback()
        runtime_config = await self.ctx.config.get_all()
        if not isinstance(runtime_config, dict):
            return local_config
        return _merge_config_dicts(local_config, runtime_config)

    async def _get_config_value(self, key: str, default: Any = None) -> Any:
        """读取配置字段。"""
        plugin_config = await self._get_plugin_config()
        return _get_nested_config_value(plugin_config, key, default)

    @staticmethod
    def _normalize_task_model_list(raw_model_list: Any) -> list[str]:
        """规范化 task 下的模型列表。"""
        if not isinstance(raw_model_list, list):
            return []
        return [str(item).strip() for item in raw_model_list if str(item).strip()]

    @staticmethod
    def _extract_available_task_names_from_host_model_config() -> list[str]:
        """从宿主 model_config.toml 提取可用 task 名。"""
        host_model_config = _load_host_model_config_fallback()
        raw_task_config = host_model_config.get("model_task_config")
        if not isinstance(raw_task_config, dict):
            return []
        return [str(task_name).strip() for task_name in raw_task_config.keys() if str(task_name).strip()]

    async def _resolve_generation_model(self, configured_name: str) -> tuple[str, str]:
        """把配置值解析为宿主任务名或固定模型名。"""
        normalized_name = str(configured_name or "").strip()
        host_model_config = _load_host_model_config_fallback()
        normalized_tasks = self._extract_available_task_names_from_host_model_config()
        if not normalized_name:
            for preferred_task in ("utils", "replyer", "planner"):
                if preferred_task in normalized_tasks:
                    return ("task", preferred_task)
            return ("task", "")

        if normalized_name in normalized_tasks:
            return ("task", normalized_name)

        direct_model_name = _resolve_model_alias_from_host_model_config(normalized_name, host_model_config)
        if direct_model_name:
            logger.info("智能分段模型 `%s` 已解析为固定模型 `%s`", normalized_name, direct_model_name)
            return ("model", direct_model_name)

        candidate_names = _normalize_model_alias_candidates(normalized_name, host_model_config)
        raw_task_config = host_model_config.get("model_task_config")
        if isinstance(raw_task_config, dict):
            matched_tasks: list[tuple[str, str]] = []
            for task_name, task_config in raw_task_config.items():
                if not isinstance(task_config, dict):
                    continue
                task_model_list = self._normalize_task_model_list(task_config.get("model_list"))
                for candidate_name in candidate_names:
                    if candidate_name in task_model_list:
                        matched_tasks.append((str(task_name).strip(), candidate_name))
                        break

            if matched_tasks:
                resolved_task, matched_candidate = matched_tasks[0]
                if len(matched_tasks) > 1:
                    logger.warning(
                        "智能分段模型/标识 `%s` 命中多个宿主任务 %s，将优先使用 `%s`",
                        normalized_name,
                        [task_name for task_name, _ in matched_tasks],
                        resolved_task,
                    )
                logger.info(
                    "智能分段模型 `%s` 已映射到宿主任务 `%s` (匹配值: `%s`)",
                    normalized_name,
                    resolved_task,
                    matched_candidate,
                )
                return ("task", resolved_task)

        logger.warning(
            "智能分段配置的模型/任务 `%s` 未命中宿主可用 task，将回退默认模型",
            normalized_name,
        )
        return ("task", "")

    @staticmethod
    async def _generate_with_pinned_model(
        prompt: str,
        *,
        resolved_model_name: str,
        temperature: float,
        max_tokens: int,
        request_type: str,
    ) -> dict[str, Any]:
        """像 nai_pic_plugin 一样，直接固定到底层模型执行生成。"""
        orchestrator = _PinnedTaskLLMOrchestrator(
            TaskConfig(
                model_list=[resolved_model_name],
                max_tokens=max_tokens,
                temperature=temperature,
                slow_threshold=30.0,
                selection_strategy="random",
            ),
            request_type=request_type,
        )
        result = await orchestrator.generate_response_async(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "success": True,
            "response": result.response,
            "reasoning": result.reasoning,
            "model_name": result.model_name,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }

    @staticmethod
    def _build_segmentation_prompt(text: str, style: str, max_segments: int) -> str:
        """构建智能分段提示词。"""
        style_guides = {
            "natural": "像和朋友微信聊天一样自然地分条发送。有的消息短有的长，节奏随意。",
            "conservative": "偏沉稳的发消息风格，一条消息说比较完整的内容，不会频繁发短消息。",
            "active": "活泼的发消息风格，喜欢发短消息连击，反应词和正文分开发。",
        }

        return f"""你正在模拟一个人用手机聊天。下面是 ta 想说的内容，请把它分成几条消息，就像真人会怎么一条一条发出来那样。

{style_guides.get(style, style_guides["natural"])}

规则：
- 不要改写原意，不要补充新信息
- 去掉每条消息末尾的句号「。」
- 保留感叹号、问号、省略号、波浪号等有情绪的标点
- 不要每个逗号都拆开，相关的内容放在一条里
- 消息长短可以不均匀
- 最多分成 {max_segments} 条
- 如果不适合切分，就返回只包含原文的一项数组

原文：{text}

只返回 JSON 数组，如 ["消息1", "消息2"]"""

    async def _segment_text(
        self,
        text: str,
        *,
        style: str,
        model_name: str,
        max_segments: int,
        temperature: float,
        max_tokens: int,
    ) -> list[str] | None:
        """调用 LLM 对文本进行分段。"""
        prompt = self._build_segmentation_prompt(text, style, max_segments)
        target_kind, target_name = await self._resolve_generation_model(model_name)
        logger.info(
            "智能分段开始调用 LLM: configured_model=%r resolved_kind=%r resolved_target=%r",
            model_name,
            target_kind,
            target_name or "<default>",
        )

        try:
            if target_kind == "model" and target_name:
                raw_result = await self._generate_with_pinned_model(
                    prompt,
                    resolved_model_name=target_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_type="plugin.smart_segmentation.segment",
                )
            else:
                raw_result = await self.ctx.llm.generate(
                    prompt=prompt,
                    model=target_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            logger.error("智能分段 LLM 调用失败: %s", exc, exc_info=True)
            return None

        result = raw_result
        if isinstance(raw_result, dict) and isinstance(raw_result.get("result"), dict):
            result = raw_result.get("result") or {}

        if not result.get("success", False):
            logger.error("智能分段 LLM 返回失败: %s", result)
            return None

        actual_model_name = str(result.get("model", "") or result.get("model_name", "") or "").strip()
        if actual_model_name:
            logger.info(
                "智能分段 LLM 调用完成: configured_model=%r resolved_kind=%r resolved_target=%r actual_model=%r",
                model_name,
                target_kind,
                target_name or "<default>",
                actual_model_name,
            )

        response_text = str(result.get("response", "") or "").strip()
        if not response_text:
            logger.warning("智能分段 LLM 返回空结果: %r", result)
            return None

        try:
            json_text = _extract_json_array_text(response_text)
            segments = json.loads(json_text)
            return _normalize_segments(segments, max_segments=max_segments)
        except Exception as exc:
            logger.error("解析智能分段结果失败: %s, 原始返回: %r", exc, response_text)
            return None

    @staticmethod
    def _calculate_send_delay(segment: str, delay_base: float, delay_per_char: float, delay_max: float) -> float:
        """根据文本长度计算分条发送间隔。"""
        normalized_delay = delay_base + len(segment) * delay_per_char + random.uniform(0.0, 0.15)
        return max(0.0, min(delay_max, normalized_delay))

    async def _send_segments(
        self,
        stream_id: str,
        segments: list[str],
        *,
        delay_base: float,
        delay_per_char: float,
        delay_max: float,
        delay_before_first: bool = False,
    ) -> bool:
        """逐条发送分段结果。"""
        for index, segment in enumerate(segments):
            if index > 0 or delay_before_first:
                await asyncio.sleep(self._calculate_send_delay(segment, delay_base, delay_per_char, delay_max))

            send_ok = await self.ctx.send.text(segment, stream_id)
            if not send_ok:
                logger.error("发送分段消息失败，第 %s 段: %r", index + 1, segment)
                return False

        return True

    async def _get_segmentation_runtime_settings(self) -> dict[str, Any] | None:
        """读取并规范化运行时所需的分段配置。"""
        plugin_enabled = bool(await self._get_config_value("plugin.enabled", True))
        segmentation_enabled = bool(await self._get_config_value("segmentation.enabled", True))
        if not plugin_enabled or not segmentation_enabled or not _runtime_enabled:
            return None

        min_length_raw = await self._get_config_value("segmentation.min_length", 15)
        max_segments_raw = await self._get_config_value("segmentation.max_segments", 8)
        temperature_raw = await self._get_config_value("segmentation.temperature", 0.3)
        max_tokens_raw = await self._get_config_value("segmentation.max_tokens", 600)
        style = str(await self._get_config_value("segmentation.style", "natural") or "natural")
        model_name = str(await self._get_config_value("segmentation.model", "") or "")
        delay_base_raw = await self._get_config_value("segmentation.delay_base", 0.35)
        delay_per_char_raw = await self._get_config_value("segmentation.delay_per_char", 0.015)
        delay_max_raw = await self._get_config_value("segmentation.delay_max", 1.2)

        try:
            min_length = int(min_length_raw)
        except (TypeError, ValueError):
            min_length = 15
        try:
            max_segments = max(1, int(max_segments_raw))
        except (TypeError, ValueError):
            max_segments = 8
        try:
            temperature = float(temperature_raw)
        except (TypeError, ValueError):
            temperature = 0.3
        try:
            max_tokens = int(max_tokens_raw)
        except (TypeError, ValueError):
            max_tokens = 600
        try:
            delay_base = float(delay_base_raw)
        except (TypeError, ValueError):
            delay_base = 0.35
        try:
            delay_per_char = float(delay_per_char_raw)
        except (TypeError, ValueError):
            delay_per_char = 0.015
        try:
            delay_max = float(delay_max_raw)
        except (TypeError, ValueError):
            delay_max = 1.2

        return {
            "min_length": min_length,
            "max_segments": max_segments,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "style": style,
            "model_name": model_name,
            "delay_base": delay_base,
            "delay_per_char": delay_per_char,
            "delay_max": delay_max,
        }

    @EventHandler(
        "smart_segmentation_after_llm",
        description="在 AFTER_LLM 阶段预切分主回复并写入分段标记",
        event_type=EventType.AFTER_LLM,
        intercept_message=True,
        weight=120,
    )
    async def handle_smart_segmentation_after_llm(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """在 AFTER_LLM 阶段提前分段，仅写入内部标记。"""
        del kwargs

        if not isinstance(message, dict):
            return {"continue_processing": True}

        settings = await self._get_segmentation_runtime_settings()
        if settings is None:
            return {"continue_processing": True}

        response_content = str(message.get("llm_response_content", "") or "").strip()
        if not response_content:
            return {"continue_processing": True}

        if response_content.startswith(_PREPARED_SEGMENTS_MARKER):
            return {"continue_processing": True}

        visible_text = _strip_thinking_content(response_content)
        if not visible_text or len(visible_text) < settings["min_length"]:
            return {"continue_processing": True}

        segments = await self._segment_text(
            visible_text,
            style=settings["style"],
            model_name=settings["model_name"],
            max_segments=settings["max_segments"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
        )
        if not segments or len(segments) <= 1:
            return {"continue_processing": True}

        marked_text = _build_prepared_segments_marker_text(segments)
        if not marked_text:
            return {"continue_processing": True}

        _inject_prepared_segments_into_message(message, segments)
        logger.info("智能分段预切分完成，已写入分段标记，共 %s 段", len(segments))
        return {
            "continue_processing": True,
            "modified_message": message,
        }

    @HookHandler(
        "chat.command.before_execute",
        name="smart_segmentation_command_scope_enter",
        description="在命令执行期间标记当前聊天流，避免命令回执被智能分段",
    )
    async def handle_command_before_execute(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """在命令执行前标记聊天流。"""
        del kwargs

        if isinstance(message, dict):
            _mark_command_stream_active(message.get("session_id", ""))
        return {"action": "continue"}

    @HookHandler(
        "chat.command.after_execute",
        name="smart_segmentation_command_scope_leave",
        description="在命令执行结束后清理聊天流标记",
    )
    async def handle_command_after_execute(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """在命令执行结束后清理聊天流标记。"""
        del kwargs

        if isinstance(message, dict):
            _mark_command_stream_inactive(message.get("session_id", ""))
        return {"action": "continue"}

    @HookHandler(
        "send_service.after_build_message",
        name="smart_segmentation_after_build",
        description="在发送前消费预分段标记，首段保留原始发送链路",
    )
    async def handle_smart_segmentation_after_build(
        self,
        message: dict[str, Any] | None = None,
        stream_id: str = "",
        display_message: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """在发送前改写首段，并登记后续待补发的分段。"""
        updated_kwargs = dict(kwargs)
        updated_kwargs["message"] = message
        updated_kwargs["stream_id"] = stream_id
        updated_kwargs["display_message"] = display_message

        if not isinstance(message, dict):
            return {"action": "continue"}

        normalized_stream_id = str(stream_id or message.get("session_id", "") or "").strip()
        if not normalized_stream_id or _is_stream_guarded(normalized_stream_id):
            return {"action": "continue"}

        if _is_command_stream_active(normalized_stream_id):
            logger.debug("智能分段跳过发送前兜底：当前聊天流正在执行命令")
            return {"action": "continue"}
        grace_remaining = _get_command_stream_grace_remaining(normalized_stream_id)
        if grace_remaining is not None:
            logger.info(
                "智能分段跳过发送前兜底：当前聊天流处于命令回执保护窗口，剩余 %.2fs",
                grace_remaining,
            )
            return {"action": "continue"}

        settings = await self._get_segmentation_runtime_settings()
        if settings is None:
            return {"action": "continue"}

        segments, marker_source = _extract_prepared_segments_from_outbound_message(message, display_message)
        if not segments or len(segments) <= 1:
            if not _should_apply_after_build_fallback(
                message,
                storage_message=bool(updated_kwargs.get("storage_message", True)),
            ):
                logger.debug("智能分段跳过发送前兜底：当前消息不满足兜底条件")
                return {"action": "continue"}

            outbound_text = _strip_thinking_content(_extract_plain_text_outbound_message(message, display_message))
            if not outbound_text or len(outbound_text) < settings["min_length"]:
                logger.debug(
                    "智能分段跳过发送前兜底：文本为空或长度不足，length=%s min_length=%s",
                    len(outbound_text),
                    settings["min_length"],
                )
                return {"action": "continue"}

            segments = await self._segment_text(
                outbound_text,
                style=settings["style"],
                model_name=settings["model_name"],
                max_segments=settings["max_segments"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            )
            if not segments or len(segments) <= 1:
                return {"action": "continue"}
            marker_source = "after_build_fallback"

        first_segment = segments[0]
        follow_up_segments = segments[1:]
        message_id = str(message.get("message_id", "") or "").strip()
        tracking_key = _build_follow_up_tracking_key(
            stream_id=normalized_stream_id,
            timestamp=message.get("timestamp", ""),
            visible_text=first_segment,
        )
        lookup_keys = _normalize_pending_lookup_keys(message_id, tracking_key)
        if not follow_up_segments or not lookup_keys:
            return {"action": "continue"}

        updated_message = _replace_outbound_text(message, first_segment)
        updated_kwargs["message"] = updated_message
        updated_kwargs["display_message"] = first_segment

        _register_pending_follow_up_segments(
            lookup_keys=lookup_keys,
            pending_data={
                "stream_id": normalized_stream_id,
                "segments": follow_up_segments,
                "delay_base": settings["delay_base"],
                "delay_per_char": settings["delay_per_char"],
                "delay_max": settings["delay_max"],
            },
        )
        if marker_source == "replyer":
            logger.info("智能分段已从 replyer 兜底标记恢复，原始发送保留首段并登记 %s 条补发消息", len(follow_up_segments))
        elif marker_source == "additional_data":
            logger.info("智能分段已从 additional_data 恢复预分段，原始发送保留首段并登记 %s 条补发消息", len(follow_up_segments))
        elif marker_source == "additional_config":
            logger.info("智能分段已从 additional_config 恢复预分段，原始发送保留首段并登记 %s 条补发消息", len(follow_up_segments))
        elif marker_source == "after_build_fallback":
            logger.info("智能分段已在发送前直接分段，原始发送保留首段并登记 %s 条补发消息", len(follow_up_segments))
        else:
            logger.info("智能分段已消费预分段标记，原始发送保留首段并登记 %s 条补发消息", len(follow_up_segments))
        return {"action": "continue", "modified_kwargs": updated_kwargs}

    @HookHandler(
        "send_service.after_send",
        name="smart_segmentation_after_send",
        description="在首段发送成功后补发剩余智能分段消息",
        mode=HookMode.OBSERVE,
    )
    async def handle_smart_segmentation_after_send(
        self,
        message: dict[str, Any] | None = None,
        sent: bool = False,
        **kwargs: Any,
    ) -> None:
        """在首段发送成功后补发其余分段。"""
        del kwargs

        if not isinstance(message, dict):
            return None

        if not sent:
            return None

        message_id = str(message.get("message_id", "") or "").strip()
        normalized_stream_id = str(message.get("session_id", "") or "").strip()
        tracking_text = str(
            message.get("display_message", "")
            or message.get("processed_plain_text", "")
            or _extract_plain_text_outbound_message(message)
            or ""
        ).strip()
        tracking_key = _build_follow_up_tracking_key(
            stream_id=normalized_stream_id,
            timestamp=message.get("timestamp", ""),
            visible_text=tracking_text,
        )

        pending_data = _resolve_pending_follow_up_segments(message_id=message_id, tracking_key=tracking_key)
        if pending_data is None:
            return None

        stream_id = str(pending_data.get("stream_id", "") or message.get("session_id", "") or "").strip()
        segments = pending_data.get("segments")
        if not stream_id or not isinstance(segments, list) or not segments:
            return None

        try:
            delay_base = float(pending_data.get("delay_base", 0.35))
        except (TypeError, ValueError):
            delay_base = 0.35
        try:
            delay_per_char = float(pending_data.get("delay_per_char", 0.015))
        except (TypeError, ValueError):
            delay_per_char = 0.015
        try:
            delay_max = float(pending_data.get("delay_max", 1.2))
        except (TypeError, ValueError):
            delay_max = 1.2

        with _guard_stream_resend(stream_id):
            send_ok = await self._send_segments(
                stream_id,
                segments,
                delay_base=delay_base,
                delay_per_char=delay_per_char,
                delay_max=delay_max,
                delay_before_first=True,
            )

        if not send_ok:
            logger.error("智能分段补发失败，stream_id=%s message_id=%s", stream_id, message_id)
        return None

    @Command("smart_seg", description="开关智能分段功能", pattern=r"^/smart_seg(?:\s+(?P<action>on|off|status))?\s*$")
    async def handle_smart_seg(
        self,
        stream_id: str = "",
        matched_groups: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None, bool]:
        """通过命令控制运行时智能分段开关。"""
        del kwargs
        global _runtime_enabled

        action = str((matched_groups or {}).get("action", "") or "").strip().lower()
        if not action:
            _runtime_enabled = not _runtime_enabled
            state = "开启" if _runtime_enabled else "关闭"
            await self.ctx.send.text(f"智能分段已{state}", stream_id)
            logger.info("智能分段已通过命令切换为: %s", state)
            return True, f"智能分段已{state}", True

        if action == "on":
            _runtime_enabled = True
            await self.ctx.send.text("智能分段已开启", stream_id)
            logger.info("智能分段已通过命令开启")
            return True, "智能分段已开启", True

        if action == "off":
            _runtime_enabled = False
            await self.ctx.send.text("智能分段已关闭", stream_id)
            logger.info("智能分段已通过命令关闭")
            return True, "智能分段已关闭", True

        if action == "status":
            plugin_enabled = bool(await self._get_config_value("plugin.enabled", True))
            segmentation_enabled = bool(await self._get_config_value("segmentation.enabled", True))
            state = "开启" if _runtime_enabled else "关闭"
            config_state = "启用" if plugin_enabled and segmentation_enabled else "禁用"
            await self.ctx.send.text(f"智能分段当前状态: {state}，配置状态: {config_state}", stream_id)
            return True, f"状态: {state}, 配置: {config_state}", True

        await self.ctx.send.text("用法: /smart_seg [on|off|status]", stream_id)
        return False, "参数错误", True

    async def on_config_update(self, *args: Any, **kwargs: Any) -> None:
        """处理配置热重载事件。"""
        del args
        del kwargs


def create_plugin() -> SmartSegmentationPlugin:
    """创建智能分段插件实例。"""
    return SmartSegmentationPlugin()
