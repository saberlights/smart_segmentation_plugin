"""智能分段插件。"""

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import tomllib
import unicodedata

from maibot_sdk import Command, MaiBotPlugin
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
        hook: str,
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
                    "hook": str(hook or "").strip(),
                    "name": str(name or hook.replace(".", "_") or func.__name__).strip(),
                    "description": description,
                    "mode": HookMode.OBSERVE if mode == HookMode.OBSERVE else HookMode.BLOCKING,
                    "metadata": dict(metadata),
                },
            )
            return func

        return decorator

logger = logging.getLogger("plugin.smart_segmentation")

# === 运行时全局状态 ===
_runtime_enabled = True

# 早期路径预分段缓存：key = (stream_id, normalized_response_hash)
# 由 maisaka.replyer.after_response hook 写入，由 send_service.after_build_message 消费
_prepared_segment_registry: dict[tuple[str, str], dict[str, Any]] = {}
_PREPARED_SEGMENT_TTL_SECONDS = 60.0

# 首段直发后，剩余分段在 after_send 阶段补发
_pending_follow_up_segments: dict[str, dict[str, Any]] = {}
_PENDING_FOLLOW_UP_TTL_SECONDS = 60.0

# 补发必须脱离宿主 after_send 的固定超时，同时保留强引用，避免后台 Task 被回收。
_active_follow_up_tasks: set[asyncio.Task[Any]] = set()
_active_follow_up_tasks_by_stream: dict[str, set[asyncio.Task[Any]]] = {}
_follow_up_idle_events_by_stream: dict[str, asyncio.Event] = {}

# planner 可能在补发历史写入前构建好 prompt；按流保留一次性修补信息和完成屏障。
_planner_follow_up_entries_by_stream: dict[str, list[dict[str, Any]]] = {}

# 插件自身补发时关闭二次分段
_stream_resend_guards: dict[str, int] = {}

# 命令执行期间禁止把命令回执误判为主回复
_active_command_streams: dict[str, int] = {}
_active_command_stream_expiries: dict[str, float] = {}
_recent_command_stream_expiries: dict[str, float] = {}
# 仅兜住命令 hook 与 send_service 之间的轻微异步抖动。值过大会把命令后紧跟的正常主回复一起误伤
# (旧 90s 窗口就出现过重启后首个 @ 回复不分段的回归)，1.0s 已经足够覆盖 IPC 微抖动，
# 同时把误伤窗口比之前的 2.0s 缩短一半。
_COMMAND_REPLY_GRACE_SECONDS = 1.0
_ACTIVE_COMMAND_STREAM_TTL_SECONDS = 300.0

# maisaka 早期路径自己的 LLM 超时；这是当前唯一会做分段 LLM 调用的入口。
_REPLYER_SEGMENT_RETRY_DELAY_SECONDS = 6.0
_REPLYER_PREPARED_SEGMENT_TIMEOUT_SECONDS = 20.0
_REPLYER_HOOK_TIMEOUT_MS = 25_000
_DEFAULT_MAX_SEGMENTS = 8

# 新消息进入同一聊天流时，最多等待正在补发的分段两分钟，避免两轮消息交错。
_FOLLOW_UP_WAIT_TIMEOUT_MS = 120_000
_FOLLOW_UP_WAIT_TIMEOUT_SECONDS = (_FOLLOW_UP_WAIT_TIMEOUT_MS - 5_000) / 1000
_FIRST_SEND_OBSERVE_GRACE_SECONDS = 5.0
_SEND_SEGMENT_RPC_TIMEOUT_MS = _FOLLOW_UP_WAIT_TIMEOUT_MS
_FOLLOW_UP_SEND_MAX_ATTEMPTS = 2
_FOLLOW_UP_UNLOAD_GRACE_SECONDS = 1.0


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
    config_version: str = Field(default="1.1.0", description="配置文件版本")
    version: str = Field(default="1.1.0", description="插件版本")
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
    typing_enabled: bool = Field(default=True, description="是否为后续分段启用宿主模拟打字等待")


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


# === mtime 缓存：避免每个 hook 都同步读盘 ===

_host_model_config_cache: dict[str, Any] = {}
_host_model_config_cache_mtime: float | None = None

_local_plugin_config_cache: dict[str, Any] = {}
_local_plugin_config_cache_mtime: float | None = None


def _load_host_model_config_fallback() -> dict[str, Any]:
    """回退读取宿主 model_config.toml；按 mtime 缓存，文件未变就走内存。"""
    global _host_model_config_cache, _host_model_config_cache_mtime

    model_config_path = _find_host_model_config_path()
    if not os.path.isfile(model_config_path):
        return {}

    try:
        mtime = os.path.getmtime(model_config_path)
    except OSError as exc:
        logger.warning("读取宿主 model_config.toml mtime 失败: %s", exc)
        return _host_model_config_cache

    if _host_model_config_cache_mtime == mtime and _host_model_config_cache:
        return _host_model_config_cache

    try:
        with open(model_config_path, "rb") as config_file:
            config_data = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("读取宿主 model_config.toml 失败，无法做模型名映射: %s", exc)
        return _host_model_config_cache

    if not isinstance(config_data, dict):
        return {}

    _host_model_config_cache = config_data
    _host_model_config_cache_mtime = mtime
    return _host_model_config_cache


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
    """移除 thinking 标签及其内容，只保留最终可见正文。

    同时覆盖 ``<thinking>`` 与 doubao/DeepSeek 风格的 ``<think>`` 标签。
    """
    if not text:
        return ""

    cleaned_text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned_text = re.sub(r"</?think(?:ing)?>", "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text.strip()


def _extract_json_array_text(raw_text: str) -> str:
    """从模型返回中提取 JSON 数组文本。

    围栏提取后仍做一次数组切片：模型可能在围栏内先输出说明文字，
    或使用大写 ```JSON 围栏。
    """
    result_text = str(raw_text or "").strip()
    fence_index = result_text.lower().find("```json")
    if fence_index != -1:
        result_text = result_text[fence_index + len("```json") :].split("```", 1)[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```", 1)[1].split("```", 1)[0].strip()

    start = result_text.find("[")
    end = result_text.rfind("]")
    if start != -1 and end != -1 and start < end:
        return result_text[start : end + 1]
    return result_text


def _normalize_text_for_content_check(text: str) -> str:
    """归一化文本用于内容保真比对：NFKC 折叠全半角，仅保留文字/数字并忽略大小写。

    标点、空白、emoji 全部忽略——分段模型允许调整边界标点（去句号/逗号、
    全角括号转半角、省略独立的"……"），这类差异不应导致分段失效；
    但字词级的改写、增删必须被拦截。
    """
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(re.findall(r"\w+", normalized)).casefold()


def _segments_preserve_original_content(original_text: str, segments: list[str]) -> bool:
    """校验分段拼接后与原文的文字内容一致（忽略标点、空白、宽度与大小写）。"""
    return _normalize_text_for_content_check("".join(segments)) == _normalize_text_for_content_check(original_text)


# === 括号识别 ===
# 同时覆盖中英文括号；角色扮演动作描述常用全角"（）"。
_BRACKET_PAIRS: tuple[tuple[str, str], ...] = (
    ("（", "）"),
    ("(", ")"),
    ("【", "】"),
    ("[", "]"),
)


def _is_action_only_text(text: str) -> bool:
    """判断文本是否整体被一对括号包裹（典型的角色扮演动作/神态描述）。

    用于在分段前直接放行此类消息，避免把"（理理缓缓起身）"这种整段动作描述
    继续拆开发送。
    """
    stripped = str(text or "").strip()
    if len(stripped) < 2:
        return False

    for open_bracket, close_bracket in _BRACKET_PAIRS:
        if not stripped.startswith(open_bracket) or not stripped.endswith(close_bracket):
            continue
        depth = 0
        for index, char in enumerate(stripped):
            if char == open_bracket:
                depth += 1
            elif char == close_bracket:
                depth -= 1
                if depth == 0:
                    # 首个完整闭合点必须正好是末尾字符，否则中间已经先闭合再起新括号。
                    return index == len(stripped) - 1
    return False


def _has_unbalanced_brackets(text: str) -> bool:
    """判断文本中是否存在未闭合的括号。"""
    for open_bracket, close_bracket in _BRACKET_PAIRS:
        if text.count(open_bracket) != text.count(close_bracket):
            return True
    return False


def _merge_segments_balancing_brackets(segments: list[str]) -> list[str]:
    """合并被模型拆到不同段的括号对，保证每段的括号都是平衡的。

    背景：模型偶尔会把"（动作描述）"切到相邻段（如 ["a（理理", "起身）b"]），
    后续的"按括号边界拆段"必须建立在每段括号成对的前提下，否则会识别不到完整
    的括号块。这里按括号深度累计扫描，发现累计段还有未闭合的括号就和下一段合
    并，直到括号闭合。
    """
    if not segments:
        return list(segments)

    merged: list[str] = []
    buffer = ""
    for segment in segments:
        buffer = buffer + segment if buffer else segment
        if not _has_unbalanced_brackets(buffer):
            merged.append(buffer)
            buffer = ""

    if buffer:
        # 累计完所有段后括号仍未闭合：原文里大概率就缺一半括号，
        # 直接把剩余 buffer 作为一段输出，保留原文内容不丢失。
        merged.append(buffer)

    return merged


def _split_text_at_brackets(text: str) -> list[str]:
    """把单段文本按括号边界拆成片段，括号块本身作为独立片段保留。

    例如 ``"你好啊（理理缓缓起身）今天怎么样"`` 会被拆成
    ``["你好啊", "（理理缓缓起身）", "今天怎么样"]``。
    """
    if not text:
        return []

    parts: list[str] = []
    buffer: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        matched_pair: tuple[str, str] | None = None
        for open_bracket, close_bracket in _BRACKET_PAIRS:
            if char == open_bracket:
                matched_pair = (open_bracket, close_bracket)
                break

        if matched_pair is None:
            buffer.append(char)
            index += 1
            continue

        open_bracket, close_bracket = matched_pair
        depth = 1
        scan_index = index + 1
        while scan_index < len(text) and depth > 0:
            if text[scan_index] == open_bracket:
                depth += 1
            elif text[scan_index] == close_bracket:
                depth -= 1
            scan_index += 1

        if depth != 0:
            # 这段括号没有闭合，留给上层的不平衡处理，整体保留不拆。
            buffer.append(text[index:])
            index = len(text)
            break

        if buffer:
            parts.append("".join(buffer))
            buffer = []
        parts.append(text[index:scan_index])
        index = scan_index

    if buffer:
        parts.append("".join(buffer))

    return parts


def _split_segments_at_bracket_boundaries(segments: list[str], *, max_segments: int) -> list[str]:
    """把每段内的括号包裹内容拆成独立的消息段。

    场景：括号内的动作/神态描述（如"（理理缓缓起身）"）应该作为单独一条消息发送，
    而不是和括号外的正文连在一起。最终段数受 ``max_segments`` 限制：超出时把溢出
    的部分合并回最后一段，避免吃掉内容。
    """
    if not segments:
        return list(segments)

    result: list[str] = []
    for segment in segments:
        for part in _split_text_at_brackets(segment):
            stripped_part = part.strip()
            if stripped_part:
                result.append(stripped_part)

    if not result:
        return result

    if max_segments > 0 and len(result) > max_segments:
        head = result[: max_segments - 1]
        # 溢出的分段直接首尾拼接，保留原始字符顺序但不再细分。
        tail = "".join(result[max_segments - 1 :])
        result = head + [tail]

    return result


# === 预分段缓存 ===

def _normalize_response_text_for_key(text: str) -> str:
    """归一化用于查找预分段缓存的文本：剥 thinking + 折叠所有空白。"""
    cleaned = _strip_thinking_content(str(text or ""))
    if not cleaned:
        return ""
    return " ".join(cleaned.split())


def _hash_normalized_text(text: str) -> str:
    """对归一化后的文本做稳定哈希；空文本返回空串。"""
    normalized = _normalize_response_text_for_key(text)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _prune_expired_prepared_segments() -> None:
    """清理早期路径预分段缓存里所有已超时的条目。"""
    if not _prepared_segment_registry:
        return
    now = time.monotonic()
    expired_keys = [k for k, v in _prepared_segment_registry.items() if v.get("expires_at", 0.0) <= now]
    for key in expired_keys:
        _prepared_segment_registry.pop(key, None)


def _store_prepared_segments(stream_id: str, response_text: str, segments: list[str]) -> bool:
    """登记早期路径预分段；返回是否成功登记。"""
    normalized_stream_id = str(stream_id or "").strip()
    text_hash = _hash_normalized_text(response_text)
    if not normalized_stream_id or not text_hash or not segments:
        return False

    _prune_expired_prepared_segments()
    _prepared_segment_registry[(normalized_stream_id, text_hash)] = {
        "segments": list(segments),
        "expires_at": time.monotonic() + _PREPARED_SEGMENT_TTL_SECONDS,
    }
    return True


def _pop_prepared_segments(stream_id: str, outbound_text: str) -> list[str] | None:
    """命中即返回缓存里的分段并移除条目；未命中返回 None。"""
    _prune_expired_prepared_segments()
    normalized_stream_id = str(stream_id or "").strip()
    text_hash = _hash_normalized_text(outbound_text)
    if not normalized_stream_id or not text_hash:
        return None

    entry = _prepared_segment_registry.pop((normalized_stream_id, text_hash), None)
    if entry is None:
        return None
    segments = entry.get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    return list(segments)


def _has_prepared_segments(stream_id: str, response_text: str) -> bool:
    """检查回复是否存在精确匹配的预分段缓存，但不消费缓存。"""
    _prune_expired_prepared_segments()
    normalized_stream_id = str(stream_id or "").strip()
    text_hash = _hash_normalized_text(response_text)
    if not normalized_stream_id or not text_hash:
        return False
    return (normalized_stream_id, text_hash) in _prepared_segment_registry


def _normalize_segments(segments: Any, *, max_segments: int) -> list[str]:
    """规范化模型返回的分段结果。"""
    if not isinstance(segments, list):
        raise ValueError("模型返回的分段结果不是列表")

    normalized_segments = [str(segment).strip() for segment in segments if str(segment).strip()]
    if not normalized_segments:
        raise ValueError("模型返回的分段结果为空")

    if max_segments <= 0 or len(normalized_segments) <= max_segments:
        return normalized_segments

    return normalized_segments[: max_segments - 1] + ["".join(normalized_segments[max_segments - 1 :])]


def _render_mention_component_text(component: dict[str, Any]) -> str:
    """将真实艾特组件渲染为用于分段判断的可见文本。"""
    component_data = component.get("data")
    if isinstance(component_data, dict):
        target_text = ""
        for key in (
            "target_user_cardname",
            "target_user_nickname",
            "card",
            "nickname",
            "name",
            "target_user_id",
            "user_id",
            "qq",
            "id",
        ):
            target_text = str(component_data.get(key, "") or "").strip()
            if target_text:
                break
    else:
        target_text = str(component_data or "").strip()

    if not target_text:
        return ""
    if target_text.startswith("@"):
        return target_text
    return f"@{target_text}"


def _is_mention_component_type(component_type: str) -> bool:
    return component_type in {"at", "mention", "mention_bot"}


def _extract_plain_text_outbound_message(message: dict[str, Any], processed_plain_text: str = "") -> str:
    """从发送链消息中提取可安全分段的纯文本。

    宿主在 ``send_service.after_build_message`` Hook 里实际传入的参数名是
    ``processed_plain_text``；该参数仅作 raw_message 解析失败时的兜底回填。
    """
    raw_components = message.get("raw_message")
    if not isinstance(raw_components, list):
        return str(processed_plain_text or "").strip()

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

        if _is_mention_component_type(component_type):
            mention_text = _render_mention_component_text(component)
            if not mention_text:
                return ""
            parts.append(mention_text)
            continue

        if component_type == "reply":
            continue

        # 只处理文本、真实艾特与引用元数据，避免打乱图片/语音等结构。
        return ""

    text = "".join(parts).strip()
    if text:
        return text
    return str(processed_plain_text or "").strip()


def _extract_leading_mentions_and_text_body(message: dict[str, Any]) -> tuple[str, str] | None:
    """识别宿主 rich reply 生成的 ``AtComponent... + TextComponent...`` 结构。"""
    raw_components = message.get("raw_message")
    if not isinstance(raw_components, list):
        return None

    mention_parts: list[str] = []
    text_parts: list[str] = []
    text_started = False
    for component in raw_components:
        if not isinstance(component, dict):
            return None
        component_type = str(component.get("type", "") or "").strip().lower()
        if component_type == "reply":
            continue
        if _is_mention_component_type(component_type):
            if text_started:
                return None
            mention_text = _render_mention_component_text(component)
            if not mention_text:
                return None
            mention_parts.append(mention_text)
            continue
        if component_type == "text":
            text_started = True
            text_parts.append(str(component.get("data", "") or ""))
            continue
        return None

    mention_prefix = "".join(mention_parts)
    text_body = "".join(text_parts).strip()
    if not mention_prefix or not text_body:
        return None
    return mention_prefix, text_body


def _clone_message_component(component: dict[str, Any]) -> dict[str, Any]:
    cloned_component = dict(component)
    component_data = component.get("data")
    if isinstance(component_data, dict):
        cloned_component["data"] = dict(component_data)
    return cloned_component


def _build_replaced_outbound_raw_message(message: dict[str, Any], new_text: str) -> list[dict[str, Any]]:
    raw_components = message.get("raw_message")
    if not isinstance(raw_components, list):
        return [{"type": "text", "data": new_text}]

    replaced_components: list[dict[str, Any]] = []
    remaining_text = new_text
    for component in raw_components:
        if not isinstance(component, dict):
            return [{"type": "text", "data": new_text}]

        component_type = str(component.get("type", "") or "").strip().lower()
        if component_type == "text":
            continue

        if not _is_mention_component_type(component_type):
            return [{"type": "text", "data": new_text}]

        mention_text = _render_mention_component_text(component)
        if not mention_text:
            return [{"type": "text", "data": new_text}]

        mention_index = remaining_text.find(mention_text)
        if mention_index < 0:
            continue

        prefix_text = remaining_text[:mention_index]
        if prefix_text:
            replaced_components.append({"type": "text", "data": prefix_text})
        replaced_components.append(_clone_message_component(component))
        remaining_text = remaining_text[mention_index + len(mention_text) :]

    if not replaced_components:
        return [{"type": "text", "data": new_text}]

    if remaining_text:
        replaced_components.append({"type": "text", "data": remaining_text})
    return replaced_components


def _replace_outbound_text(message: dict[str, Any], new_text: str) -> dict[str, Any]:
    """将发送链消息改写为新的单条纯文本。"""
    updated_message = dict(message)
    updated_message["raw_message"] = _build_replaced_outbound_raw_message(message, new_text)
    updated_message["processed_plain_text"] = new_text
    # SessionMessage 字段里并没有 display_message，这里写一份只是方便调试/兼容旧链路读取。
    updated_message["display_message"] = new_text
    message_info = updated_message.get("message_info")
    if isinstance(message_info, dict):
        updated_message["message_info"] = dict(message_info)
    return updated_message


# === 待补发分段 ===

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


def _prune_expired_pending_follow_up() -> None:
    """清理 pending 注册表里超时未消费的条目，避免 send 失败时常驻内存。"""
    if not _pending_follow_up_segments:
        return
    now = time.monotonic()
    expired_keys: list[str] = []
    seen_owners: set[int] = set()
    for lookup_key, pending_data in list(_pending_follow_up_segments.items()):
        if id(pending_data) in seen_owners:
            continue
        seen_owners.add(id(pending_data))
        if pending_data.get("expires_at", 0.0) > now:
            continue
        raw_cleanup_keys = pending_data.get("lookup_keys")
        cleanup_keys = (
            _normalize_pending_lookup_keys(*raw_cleanup_keys)
            if isinstance(raw_cleanup_keys, list)
            else [lookup_key]
        )
        expired_keys.extend(cleanup_keys)
    for key in expired_keys:
        _pending_follow_up_segments.pop(key, None)


def _register_pending_follow_up_segments(*, lookup_keys: list[str], pending_data: dict[str, Any]) -> None:
    """用多种查找键登记待补发的分段消息。"""
    _prune_expired_pending_follow_up()
    tracked_pending_data = dict(pending_data)
    tracked_pending_data["lookup_keys"] = list(lookup_keys)
    tracked_pending_data.setdefault("expires_at", time.monotonic() + _PENDING_FOLLOW_UP_TTL_SECONDS)
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
        return pending_data

    return None


def _discard_pending_follow_up_for_planner_entry(entry: Any) -> None:
    """首段未进入 after_send 时，清掉该回复仍占用的 pending 查找键。"""
    if not isinstance(entry, dict):
        return

    cleanup_keys: list[str] = []
    for lookup_key, pending_data in list(_pending_follow_up_segments.items()):
        if pending_data.get("planner_entry") is not entry:
            continue
        raw_lookup_keys = pending_data.get("lookup_keys")
        if isinstance(raw_lookup_keys, list):
            cleanup_keys.extend(_normalize_pending_lookup_keys(*raw_lookup_keys))
        else:
            cleanup_keys.append(lookup_key)
    for cleanup_key in cleanup_keys:
        _pending_follow_up_segments.pop(cleanup_key, None)


def _prune_expired_planner_follow_up_entries() -> None:
    """清理所有聊天流中已过期的 Planner 补发状态。"""
    if not _planner_follow_up_entries_by_stream:
        return
    now = time.monotonic()
    for stream_id, entries in list(_planner_follow_up_entries_by_stream.items()):
        active_entries = [entry for entry in entries if entry.get("expires_at", 0.0) > now]
        if active_entries:
            _planner_follow_up_entries_by_stream[stream_id] = active_entries
        else:
            _planner_follow_up_entries_by_stream.pop(stream_id, None)


def _register_planner_follow_up_entry(*, stream_id: str, segments: list[str]) -> dict[str, Any]:
    """登记当前回复，供紧邻的 planner 请求等待并修补已构建 prompt。"""
    _prune_expired_planner_follow_up_entries()
    entry = {
        "segments": list(segments),
        "after_send_started": asyncio.Event(),
        "completed": asyncio.Event(),
        "send_ok": False,
        "expires_at": time.monotonic() + _PENDING_FOLLOW_UP_TTL_SECONDS,
    }
    _planner_follow_up_entries_by_stream.setdefault(stream_id, []).append(entry)
    return entry


def _mark_planner_follow_up_started(entry: Any) -> None:
    if not isinstance(entry, dict):
        return
    started = entry.get("after_send_started")
    if isinstance(started, asyncio.Event):
        started.set()


def _complete_planner_follow_up_entry(entry: Any, *, send_ok: bool) -> None:
    if not isinstance(entry, dict):
        return
    entry["send_ok"] = bool(send_ok)
    completed = entry.get("completed")
    if isinstance(completed, asyncio.Event):
        completed.set()


async def _wait_for_planner_follow_up_entry(entry: dict[str, Any], *, stream_id: str) -> None:
    """先等 after_send OBSERVE 到达，再等待实际补发完成。"""
    completed = entry.get("completed")
    if not isinstance(completed, asyncio.Event) or completed.is_set():
        return

    started = entry.get("after_send_started")
    if isinstance(started, asyncio.Event) and not started.is_set():
        try:
            await asyncio.wait_for(
                started.wait(),
                timeout=_FIRST_SEND_OBSERVE_GRACE_SECONDS,
            )
        except asyncio.TimeoutError:
            _discard_pending_follow_up_for_planner_entry(entry)
            _complete_planner_follow_up_entry(entry, send_ok=False)
            logger.error("首段发送未进入 after_send，已释放智能分段等待，stream_id=%s", stream_id)
            return

    if completed.is_set():
        return
    try:
        await asyncio.wait_for(
            completed.wait(),
            timeout=_FOLLOW_UP_WAIT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("等待智能分段补发完成超时，stream_id=%s", stream_id)


def _pop_planner_follow_up_entries(stream_id: Any) -> list[dict[str, Any]]:
    _prune_expired_planner_follow_up_entries()
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return []
    return _planner_follow_up_entries_by_stream.pop(normalized_stream_id, [])


def _planner_message_body(content: str) -> str:
    """提取 planner 中序列化聊天消息的正文。"""
    header, separator, body = str(content or "").partition("\n")
    if separator and header.lstrip().startswith("<message "):
        return body
    return ""


def _repair_planner_messages(messages: list[Any], entries: list[dict[str, Any]]) -> list[Any]:
    """把 prompt 中仅可见首段的自身消息替换为本次完整分段文本。"""
    repaired_messages = [dict(message) if isinstance(message, dict) else message for message in messages]
    for entry in entries:
        segments = entry.get("segments")
        if not entry.get("send_ok") or not isinstance(segments, list) or len(segments) <= 1:
            continue
        normalized_segments = [str(segment) for segment in segments]
        first_segment = normalized_segments[0]
        candidate_index = -1
        for index in range(len(repaired_messages) - 1, -1, -1):
            message = repaired_messages[index]
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and _planner_message_body(content) == first_segment:
                candidate_index = index
                break
        if candidate_index < 0:
            continue

        matched_tail_indices: list[int] = []
        next_index = candidate_index + 1
        for tail_segment in normalized_segments[1:]:
            if next_index >= len(repaired_messages):
                break
            next_message = repaired_messages[next_index]
            if not isinstance(next_message, dict) or next_message.get("role") != "user":
                break
            next_content = next_message.get("content")
            if not isinstance(next_content, str) or _planner_message_body(next_content) != tail_segment:
                break
            matched_tail_indices.append(next_index)
            next_index += 1

        if len(matched_tail_indices) == len(normalized_segments) - 1:
            continue

        candidate = repaired_messages[candidate_index]
        original_content = str(candidate.get("content") or "")
        header = original_content.split("\n", 1)[0]
        joined_segments = "\n".join(normalized_segments)
        candidate["content"] = f"{header}\n{joined_segments}"
        for index in reversed(matched_tail_indices):
            repaired_messages.pop(index)
    return repaired_messages


# === 后台补发任务 ===
def _get_follow_up_idle_event(stream_id: Any) -> asyncio.Event:
    """返回聊天流的补发空闲事件。"""
    normalized_stream_id = str(stream_id or "").strip()
    idle_event = _follow_up_idle_events_by_stream.get(normalized_stream_id)
    if idle_event is None:
        idle_event = asyncio.Event()
        idle_event.set()
        if normalized_stream_id:
            _follow_up_idle_events_by_stream[normalized_stream_id] = idle_event
    return idle_event


def _release_follow_up_idle_event(stream_id: Any) -> None:
    """释放并移除聊天流的补发空闲事件。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return
    idle_event = _follow_up_idle_events_by_stream.pop(normalized_stream_id, None)
    if idle_event is not None:
        idle_event.set()


def _get_active_follow_up_task_count(stream_id: Any) -> int:
    """返回聊天流仍在运行的补发任务数。"""
    normalized_stream_id = str(stream_id or "").strip()
    stream_tasks = _active_follow_up_tasks_by_stream.get(normalized_stream_id)
    if not normalized_stream_id or not stream_tasks:
        return 0

    pending_tasks = {task for task in stream_tasks if not task.done()}
    if pending_tasks:
        _active_follow_up_tasks_by_stream[normalized_stream_id] = pending_tasks
        return len(pending_tasks)

    _active_follow_up_tasks_by_stream.pop(normalized_stream_id, None)
    _release_follow_up_idle_event(normalized_stream_id)
    return 0


async def _wait_for_stream_follow_up_tasks(stream_id: Any) -> int:
    """等待聊天流的后台补发结束，并返回等待前的任务数。"""
    pending_task_count = _get_active_follow_up_task_count(stream_id)
    if pending_task_count > 0:
        await _get_follow_up_idle_event(stream_id).wait()
    return pending_task_count


def _track_follow_up_task(task: asyncio.Task[Any], *, stream_id: Any) -> None:
    """持有后台补发任务，并在任务结束时释放聊天流。"""
    _active_follow_up_tasks.add(task)
    normalized_stream_id = str(stream_id or "").strip()
    if normalized_stream_id:
        _active_follow_up_tasks_by_stream.setdefault(normalized_stream_id, set()).add(task)
        _get_follow_up_idle_event(normalized_stream_id).clear()

    def _cleanup(completed_task: asyncio.Task[Any]) -> None:
        _active_follow_up_tasks.discard(completed_task)
        if not normalized_stream_id:
            return
        stream_tasks = _active_follow_up_tasks_by_stream.get(normalized_stream_id)
        if stream_tasks is not None:
            stream_tasks.discard(completed_task)
            if stream_tasks:
                return
            _active_follow_up_tasks_by_stream.pop(normalized_stream_id, None)
        _release_follow_up_idle_event(normalized_stream_id)

    task.add_done_callback(_cleanup)


async def _drain_active_follow_up_tasks() -> None:
    """等待所有后台补发任务结束，供卸载流程和测试使用。"""
    pending_tasks = [task for task in _active_follow_up_tasks if not task.done()]
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)


# === 命令保护窗口 ===

def _prune_expired_recent_command_streams(*, now: float | None = None) -> float:
    """清理所有聊天流中过期的命令回执保护窗口。"""
    current_time = time.monotonic() if now is None else now
    expired_stream_ids = [
        stream_id
        for stream_id, expires_at in _recent_command_stream_expiries.items()
        if expires_at <= current_time
    ]
    for stream_id in expired_stream_ids:
        _recent_command_stream_expiries.pop(stream_id, None)
    return current_time


def _prune_expired_active_command_streams(*, now: float) -> None:
    """回收未收到 after_execute 的失联命令标记。"""
    expired_stream_ids = [
        stream_id
        for stream_id, expires_at in _active_command_stream_expiries.items()
        if expires_at <= now
    ]
    for stream_id in expired_stream_ids:
        _active_command_streams.pop(stream_id, None)
        _active_command_stream_expiries.pop(stream_id, None)
        logger.warning("命令执行标记超时，已恢复智能分段 stream_id=%s", stream_id)


def _mark_command_stream_active(stream_id: Any) -> None:
    """标记当前聊天流正在执行命令。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return
    now = _prune_expired_recent_command_streams()
    _prune_expired_active_command_streams(now=now)
    _active_command_streams[normalized_stream_id] = _active_command_streams.get(normalized_stream_id, 0) + 1
    _active_command_stream_expiries[normalized_stream_id] = now + _ACTIVE_COMMAND_STREAM_TTL_SECONDS
    _recent_command_stream_expiries[normalized_stream_id] = now + _COMMAND_REPLY_GRACE_SECONDS


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
        _active_command_stream_expiries.pop(normalized_stream_id, None)
    # 故意不在命令结束时续期：before_execute 设定的短窗口足够覆盖回执同步发送，
    # 再次续期会把命令结束后的正常业务主回复一起挡住。


def _is_command_stream_active(stream_id: Any) -> bool:
    """判断当前聊天流是否正处于命令执行中。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return False
    _prune_expired_active_command_streams(now=time.monotonic())
    return _active_command_streams.get(normalized_stream_id, 0) > 0


def _get_command_stream_grace_remaining(stream_id: Any) -> float | None:
    """返回保护窗口剩余秒数；未命中返回 None。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return None

    now = _prune_expired_recent_command_streams()
    expires_at = _recent_command_stream_expiries.get(normalized_stream_id)
    if expires_at is None:
        return None

    return expires_at - now


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
        _prepared_segment_registry.clear()
        _pending_follow_up_segments.clear()
        _active_follow_up_tasks.clear()
        _active_follow_up_tasks_by_stream.clear()
        _follow_up_idle_events_by_stream.clear()
        _planner_follow_up_entries_by_stream.clear()
        _stream_resend_guards.clear()
        _active_command_streams.clear()
        _active_command_stream_expiries.clear()
        _recent_command_stream_expiries.clear()
        if not _SDK_HOOK_HANDLER_AVAILABLE:
            logger.info("当前 maibot_sdk 未导出 HookHandler，智能分段已启用内置 hook_handler 声明兼容")

    async def on_unload(self) -> None:
        """处理插件卸载，短暂等待正在发送的尾段后再取消。"""
        pending_tasks = [task for task in _active_follow_up_tasks if not task.done()]
        try:
            if pending_tasks:
                await asyncio.wait(pending_tasks, timeout=_FOLLOW_UP_UNLOAD_GRACE_SECONDS)
        finally:
            remaining_tasks = [task for task in _active_follow_up_tasks if not task.done()]
            for task in remaining_tasks:
                task.cancel()
            try:
                if remaining_tasks:
                    await asyncio.gather(*remaining_tasks, return_exceptions=True)
            finally:
                _prepared_segment_registry.clear()
                _pending_follow_up_segments.clear()
                _active_follow_up_tasks.clear()
                _active_follow_up_tasks_by_stream.clear()
                _follow_up_idle_events_by_stream.clear()
                _planner_follow_up_entries_by_stream.clear()
                _stream_resend_guards.clear()
                _active_command_streams.clear()
                _active_command_stream_expiries.clear()
                _recent_command_stream_expiries.clear()

    def _load_local_config_fallback(self) -> dict[str, Any]:
        """回退读取插件目录下的 `config.toml`，按 mtime 缓存。"""
        global _local_plugin_config_cache, _local_plugin_config_cache_mtime

        config_path = os.path.join(os.path.dirname(__file__), "config.toml")
        if not os.path.isfile(config_path):
            return {}

        try:
            mtime = os.path.getmtime(config_path)
        except OSError as exc:
            logger.warning("读取插件 config.toml mtime 失败: %s", exc)
            return _local_plugin_config_cache

        if _local_plugin_config_cache_mtime == mtime and _local_plugin_config_cache:
            return _local_plugin_config_cache

        try:
            with open(config_path, "rb") as config_file:
                config_data = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.error("读取插件配置失败: %s", exc)
            return _local_plugin_config_cache

        if not isinstance(config_data, dict):
            return {}

        _local_plugin_config_cache = config_data
        _local_plugin_config_cache_mtime = mtime
        return _local_plugin_config_cache

    async def _get_plugin_config(self) -> dict[str, Any]:
        """获取插件完整配置。"""
        local_config = self._load_local_config_fallback()
        runtime_config = await self.ctx.config.get_all()
        if not isinstance(runtime_config, dict):
            return local_config
        return _merge_config_dicts(local_config, runtime_config)

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
        normalized_style = style if style in style_guides else "natural"

        return f"""按真人手机聊天节奏，把原文分成若干条消息。

{style_guides[normalized_style]}

要求：
- 只在自然发送点切分；没有自然切点就保持一条，相关的内容放在一条里，消息长短可以不均匀
- 原文没有标点时，存在明显的自然发送边界也要分条；原文的换行处通常就是现成的分条边界
- 不要改写原意，逐字保留原文字词；切点处的句号、逗号可以去掉，问号、感叹号、省略号、波浪号保留
- 不要在链接、数字或英文单词中间断开
- 括号内的动作、神态或旁白必须完整、独立成一条；整段只有括号内容时不切
- 最多分成 {max_segments} 条

示例：
原文："哈哈真的吗，那太好了！我还以为你不喜欢呢。下次我们一起去看电影吧，最近有个新片子挺有意思的。"
分条：["哈哈真的吗", "那太好了！我还以为你不喜欢呢", "下次我们一起去看电影吧，最近有个新片子挺有意思的"]

原文："你买早饭我到了要吃的"
分条：["你买早饭", "我到了要吃的"]

待分段原文：{text}

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
        # JSON 数组输出体积与原文同量级，长文时固定 max_tokens 会截断输出导致解析失败；
        # 按原文长度动态扩容，配置值只作为下限。
        effective_max_tokens = max(max_tokens, len(text) * 2 + 160)
        try:
            target_kind, target_name = await self._resolve_generation_model(model_name)
        except Exception as exc:
            logger.error("解析智能分段模型失败: %s", exc, exc_info=True)
            return None
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
                    max_tokens=effective_max_tokens,
                    request_type="plugin.smart_segmentation.segment",
                )
            else:
                raw_result = await self.ctx.llm.generate(
                    prompt=prompt,
                    model=target_name,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                )
        except Exception as exc:
            logger.error("智能分段 LLM 调用失败: %s", exc, exc_info=True)
            return None

        result = raw_result
        if isinstance(raw_result, dict) and isinstance(raw_result.get("result"), dict):
            result = raw_result.get("result") or {}
        if not isinstance(result, dict):
            logger.error("智能分段 LLM 返回结构异常: %r", result)
            return None

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
            json_text = _extract_json_array_text(_strip_thinking_content(response_text))
            segments = json.loads(json_text)
            normalized = _normalize_segments(segments, max_segments=0)
            # 先合并被模型误拆的括号对，确保每段括号成对；再按括号边界把动作描述拆成独立段。
            balanced = _merge_segments_balancing_brackets(normalized)
            final_segments = _split_segments_at_bracket_boundaries(balanced, max_segments=max_segments)
        except Exception as exc:
            logger.error("解析智能分段结果失败: %s, 原始返回: %r", exc, response_text)
            return None

        # 内容保真校验：只比对文字字词（忽略标点/空白/宽度/大小写）。
        # 模型改写、增删字词时丢弃本次结果，让上层重试或回退原文直发。
        if not _segments_preserve_original_content(text, final_segments):
            logger.warning(
                "智能分段结果改写了原文字词，丢弃本次分段: 原文=%r 分段=%r",
                text,
                final_segments,
            )
            return None
        return final_segments

    async def _segment_text_with_delayed_retry(
        self,
        text: str,
        *,
        style: str,
        model_name: str,
        max_segments: int,
        temperature: float,
        max_tokens: int,
    ) -> list[str] | None:
        """首个请求迟迟未返回时，并发重试同一个分段模型。"""

        async def _start_attempt() -> list[str] | None:
            return await self._segment_text(
                text,
                style=style,
                model_name=model_name,
                max_segments=max_segments,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        first_attempt = asyncio.create_task(_start_attempt())
        all_attempts = [first_attempt]
        active_attempts: set[asyncio.Task[list[str] | None]] = {first_attempt}
        try:
            done, _ = await asyncio.wait(
                active_attempts,
                timeout=_REPLYER_SEGMENT_RETRY_DELAY_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if done:
                first_result = first_attempt.result()
                if first_result is not None:
                    return first_result
                active_attempts.clear()
                logger.warning("首次智能分段请求未返回有效结果，立即重试同一模型")
            else:
                logger.warning(
                    "首次智能分段请求超过 %.2fs，开始并发重试同一模型",
                    _REPLYER_SEGMENT_RETRY_DELAY_SECONDS,
                )

            retry_attempt = asyncio.create_task(_start_attempt())
            all_attempts.append(retry_attempt)
            active_attempts.add(retry_attempt)

            while active_attempts:
                completed, active_attempts = await asyncio.wait(
                    active_attempts,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for attempt in completed:
                    result = attempt.result()
                    if result is not None:
                        logger.info(
                            "智能分段采用第 %s 次模型请求结果",
                            1 if attempt is first_attempt else 2,
                        )
                        return result
            return None
        finally:
            for attempt in all_attempts:
                if not attempt.done():
                    attempt.cancel()
            await asyncio.gather(*all_attempts, return_exceptions=True)

    async def _send_segments(
        self,
        stream_id: str,
        segments: list[str],
        *,
        typing_enabled: bool,
    ) -> bool:
        """逐条发送补充分段，段前等待完全复用 MaiBot 原生打字速度。"""
        all_segments_sent = True
        for index, segment in enumerate(segments):
            segment_sent = False
            for attempt in range(_FOLLOW_UP_SEND_MAX_ATTEMPTS):
                try:
                    send_ok = await self.ctx.send.text(
                        segment,
                        stream_id,
                        typing=typing_enabled and attempt == 0,
                        sync_to_maisaka_history=True,
                        maisaka_source_kind="guided_reply",
                        timeout_ms=_SEND_SEGMENT_RPC_TIMEOUT_MS,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    send_ok = False
                    logger.warning(
                        "发送分段消息异常，第 %s 段，第 %s/%s 次尝试: %s",
                        index + 1,
                        attempt + 1,
                        _FOLLOW_UP_SEND_MAX_ATTEMPTS,
                        exc,
                        exc_info=True,
                    )
                if send_ok:
                    segment_sent = True
                    break
                logger.warning(
                    "发送分段消息失败，第 %s 段，第 %s/%s 次尝试: %r",
                    index + 1,
                    attempt + 1,
                    _FOLLOW_UP_SEND_MAX_ATTEMPTS,
                    segment,
                )

            if not segment_sent:
                all_segments_sent = False
                logger.error("分段消息重试仍失败，继续发送后续正文，第 %s 段: %r", index + 1, segment)

        return all_segments_sent

    async def _get_segmentation_runtime_settings(self) -> dict[str, Any] | None:
        """读取并规范化运行时所需的分段配置。"""
        plugin_config = await self._get_plugin_config()
        plugin_enabled = bool(_get_nested_config_value(plugin_config, "plugin.enabled", True))
        segmentation_enabled = bool(_get_nested_config_value(plugin_config, "segmentation.enabled", True))
        if not plugin_enabled or not segmentation_enabled or not _runtime_enabled:
            return None

        min_length_raw = _get_nested_config_value(plugin_config, "segmentation.min_length", 15)
        max_segments_raw = _get_nested_config_value(
            plugin_config,
            "segmentation.max_segments",
            _DEFAULT_MAX_SEGMENTS,
        )
        temperature_raw = _get_nested_config_value(plugin_config, "segmentation.temperature", 0.3)
        max_tokens_raw = _get_nested_config_value(plugin_config, "segmentation.max_tokens", 600)
        style = str(_get_nested_config_value(plugin_config, "segmentation.style", "natural") or "natural")
        model_name = str(_get_nested_config_value(plugin_config, "segmentation.model", "") or "")
        typing_enabled = bool(_get_nested_config_value(plugin_config, "segmentation.typing_enabled", True))

        try:
            min_length = int(min_length_raw)
        except (TypeError, ValueError):
            min_length = 15
        try:
            max_segments = max(1, int(max_segments_raw))
        except (TypeError, ValueError):
            max_segments = _DEFAULT_MAX_SEGMENTS
        try:
            temperature = float(temperature_raw)
        except (TypeError, ValueError):
            temperature = 0.3
        try:
            max_tokens = int(max_tokens_raw)
        except (TypeError, ValueError):
            max_tokens = 600
        return {
            "min_length": min_length,
            "max_segments": max_segments,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "style": style,
            "model_name": model_name,
            "typing_enabled": typing_enabled,
        }

    @HookHandler(
        "maisaka.replyer.after_response",
        name="smart_segmentation_after_replyer_response",
        description="在 Maisaka replyer 拿到模型回复后立刻预分段，把结果登记到进程内缓存，发送链可零 LLM 调用直接消费",
        timeout_ms=_REPLYER_HOOK_TIMEOUT_MS,
    )
    async def handle_maisaka_replyer_after_response(
        self,
        response: str = "",
        session_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """早期路径预分段：发送链上不再做同步 LLM 调用。"""
        del kwargs

        if not _runtime_enabled:
            return {"action": "continue"}

        normalized_stream_id = str(session_id or "").strip()
        normalized_response = str(response or "").strip()
        if not normalized_stream_id or not normalized_response:
            return {"action": "continue"}

        # 命令期间产生的 LLM 回复极少见，但若发生就跟随发送链上同样的策略跳过。
        if _is_command_stream_active(normalized_stream_id):
            return {"action": "continue"}

        settings = await self._get_segmentation_runtime_settings()
        if settings is None:
            return {"action": "continue"}

        visible_text = _strip_thinking_content(normalized_response)
        if not visible_text or len(visible_text) < settings["min_length"]:
            return {"action": "continue"}

        # 整段就是括号包裹的动作/神态描述（如"（理理缓缓起身）"），不做任何分段处理
        if _is_action_only_text(visible_text):
            logger.debug(
                "智能分段跳过：整段消息为括号包裹的动作/神态描述，stream=%s",
                normalized_stream_id,
            )
            return {"action": "continue"}

        try:
            segments = await asyncio.wait_for(
                self._segment_text_with_delayed_retry(
                    visible_text,
                    style=settings["style"],
                    model_name=settings["model_name"],
                    max_segments=settings["max_segments"],
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                ),
                timeout=_REPLYER_PREPARED_SEGMENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "智能分段在 replyer.after_response 阶段超时（> %.2fs），将回退为原文直发",
                _REPLYER_PREPARED_SEGMENT_TIMEOUT_SECONDS,
            )
            return {"action": "continue"}

        if not segments or len(segments) <= 1:
            return {"action": "continue"}

        # 缓存按归一化后的原文 hash 索引；后续 after_build_message 用同样规则做查找。
        if _store_prepared_segments(normalized_stream_id, normalized_response, segments):
            logger.info(
                "智能分段已在 replyer.after_response 阶段预切分，共 %s 段，已登记到缓存 stream=%s",
                len(segments),
                normalized_stream_id,
            )
        return {"action": "continue"}

    @HookHandler(
        "maisaka.reply.before_post_process",
        name="smart_segmentation_preserve_prepared_response",
        description="预分段成功后跳过宿主文本改写，保证发送链能按原文哈希命中缓存",
    )
    async def handle_maisaka_reply_before_post_process(
        self,
        response: str = "",
        session_id: str = "",
        reply_message_id: str = "",
        reply_tool_args: dict[str, Any] | None = None,
        skip_post_process: bool = False,
        enable_splitter: bool = True,
        enable_chinese_typo: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """只在精确命中预分段缓存时关闭本次宿主文本后处理。"""
        if not _has_prepared_segments(session_id, response):
            return {"action": "continue"}
        modified_kwargs = dict(kwargs)
        modified_kwargs.update(
            {
                "response": response,
                "session_id": session_id,
                "reply_message_id": reply_message_id,
                "reply_tool_args": dict(reply_tool_args or {}),
                "skip_post_process": True,
                "enable_splitter": bool(enable_splitter),
                "enable_chinese_typo": bool(enable_chinese_typo),
            }
        )
        return {
            "action": "continue",
            "modified_kwargs": modified_kwargs,
        }

    @HookHandler(
        "maisaka.planner.before_request",
        name="smart_segmentation_wait_before_planner",
        description="等待同流分段全部发送并同步历史，再修补可能提前构建的 planner prompt",
        mode=HookMode.BLOCKING,
        timeout_ms=_FOLLOW_UP_WAIT_TIMEOUT_MS,
    )
    async def handle_maisaka_planner_before_request(
        self,
        messages: list[Any] | None = None,
        tool_definitions: list[Any] | None = None,
        selected_history_count: int = 0,
        built_message_count: int = 0,
        selection_reason: str = "",
        session_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """关闭补发与同一次 reply 内部 planner 续轮之间的竞态窗口。"""
        entries = _pop_planner_follow_up_entries(session_id)
        if not entries:
            return {"action": "continue"}

        await asyncio.gather(
            *(
                _wait_for_planner_follow_up_entry(entry, stream_id=session_id)
                for entry in entries
            )
        )

        modified_kwargs = dict(kwargs)
        modified_kwargs.update(
            {
                "messages": _repair_planner_messages(list(messages or []), entries),
                "tool_definitions": list(tool_definitions or []),
                "selected_history_count": selected_history_count,
                "built_message_count": built_message_count,
                "selection_reason": selection_reason,
                "session_id": session_id,
            }
        )
        return {"action": "continue", "modified_kwargs": modified_kwargs}

    @HookHandler(
        "chat.receive.before_process",
        name="smart_segmentation_pause_until_follow_ups_finish",
        description="同一聊天流仍在模拟打字补发时，延后处理下一条入站消息",
        mode=HookMode.BLOCKING,
        timeout_ms=_FOLLOW_UP_WAIT_TIMEOUT_MS,
    )
    async def handle_chat_receive_before_process(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """避免用户的新消息与尚未发完的旧回复交错。"""
        del kwargs
        if not isinstance(message, dict):
            return {"action": "continue"}

        stream_id = str(message.get("session_id", "") or "").strip()
        waited_task_count = await _wait_for_stream_follow_up_tasks(stream_id)
        if waited_task_count > 0:
            logger.info("智能分段已等待聊天流 %s 的 %s 个补发任务结束", stream_id, waited_task_count)
        return {"action": "continue"}

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
        description="只消费 replyer.after_response 阶段登记的预分段缓存，不再做任何发送前兜底 LLM 调用",
    )
    async def handle_smart_segmentation_after_build(
        self,
        message: dict[str, Any] | None = None,
        stream_id: str = "",
        processed_plain_text: str = "",
        display_message: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """仅消费 maisaka.replyer.after_response 阶段的预分段缓存。

        宿主在 ``send_service.after_build_message`` 实际传入的是 ``processed_plain_text``，
        旧测试里以 ``display_message`` 调用的链路同样保留兼容。
        缓存的唯一写入者是 ``maisaka.replyer.after_response``，
        这是判定"该消息是否来自回复模型"的唯一可靠信号；其他链路（命令回执、
        memory/expression/插件 ctx.send.text 等）永远不会进缓存，因此自动跳过。
        """
        updated_kwargs = dict(kwargs)
        updated_kwargs["message"] = message
        updated_kwargs["stream_id"] = stream_id
        updated_kwargs["processed_plain_text"] = processed_plain_text or display_message

        if not isinstance(message, dict):
            return {"action": "continue"}

        normalized_stream_id = str(stream_id or message.get("session_id", "") or "").strip()
        if not normalized_stream_id or _is_stream_guarded(normalized_stream_id):
            return {"action": "continue"}

        if _is_command_stream_active(normalized_stream_id):
            logger.debug("智能分段跳过：当前聊天流正在执行命令")
            return {"action": "continue"}
        grace_remaining = _get_command_stream_grace_remaining(normalized_stream_id)
        if grace_remaining is not None:
            logger.debug(
                "智能分段跳过：聊天流处于命令回执保护窗口，剩余 %.2fs",
                grace_remaining,
            )
            return {"action": "continue"}

        outbound_text_hint = processed_plain_text or display_message
        outbound_text = _strip_thinking_content(_extract_plain_text_outbound_message(message, outbound_text_hint))

        cached_segments = _pop_prepared_segments(normalized_stream_id, outbound_text)
        if not cached_segments:
            mention_body = _extract_leading_mentions_and_text_body(message)
            if mention_body is not None:
                mention_prefix, text_body = mention_body
                cached_segments = _pop_prepared_segments(normalized_stream_id, text_body)
                if cached_segments:
                    cached_segments[0] = f"{mention_prefix}{cached_segments[0]}"
        if not cached_segments or len(cached_segments) <= 1:
            return {"action": "continue"}

        settings = await self._get_segmentation_runtime_settings()
        if settings is None:
            return {"action": "continue"}

        segments = cached_segments
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
        updated_kwargs["processed_plain_text"] = first_segment

        _register_pending_follow_up_segments(
            lookup_keys=lookup_keys,
            pending_data={
                "stream_id": normalized_stream_id,
                "segments": follow_up_segments,
                "typing_enabled": settings["typing_enabled"],
                "planner_entry": _register_planner_follow_up_entry(
                    stream_id=normalized_stream_id,
                    segments=segments,
                ),
            },
        )
        logger.info(
            "智能分段命中 replyer 预分段缓存，首段直发，登记 %s 条补发消息 stream=%s",
            len(follow_up_segments),
            normalized_stream_id,
        )
        return {"action": "continue", "modified_kwargs": updated_kwargs}

    @HookHandler(
        "send_service.after_send",
        name="smart_segmentation_after_send",
        description="首段发送成功后启动后台补发，planner 请求会等待补发完成",
        mode=HookMode.OBSERVE,
    )
    async def handle_smart_segmentation_after_send(
        self,
        message: dict[str, Any] | None = None,
        sent: bool = False,
        **kwargs: Any,
    ) -> None:
        """快速登记后台补发；实际发送不占用宿主 after_send 的超时窗口。"""
        del kwargs
        if not isinstance(message, dict):
            return None
        message_id = str(message.get("message_id", "") or "").strip()
        normalized_stream_id = str(message.get("session_id", "") or "").strip()
        if normalized_stream_id and _is_stream_guarded(normalized_stream_id):
            return None
        tracking_text = str(
            message.get("display_message", "")
            or _extract_plain_text_outbound_message(message)
            or message.get("processed_plain_text", "")
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
        planner_entry = pending_data.get("planner_entry")
        _mark_planner_follow_up_started(planner_entry)
        if not sent:
            _complete_planner_follow_up_entry(planner_entry, send_ok=False)
            return None
        stream_id = str(pending_data.get("stream_id", "") or message.get("session_id", "") or "").strip()
        segments = pending_data.get("segments")
        typing_enabled = pending_data.get("typing_enabled")
        if not stream_id or not isinstance(segments, list) or not segments or not isinstance(typing_enabled, bool):
            _complete_planner_follow_up_entry(planner_entry, send_ok=False)
            return None
        follow_up_task = asyncio.create_task(
            self._run_follow_up_segments(
                stream_id=stream_id,
                segments=list(segments),
                message_id=message_id,
                typing_enabled=typing_enabled,
                planner_entry=planner_entry,
            )
        )
        _track_follow_up_task(follow_up_task, stream_id=stream_id)
        return None

    async def _run_follow_up_segments(
        self,
        *,
        stream_id: str,
        segments: list[str],
        message_id: str,
        typing_enabled: bool,
        planner_entry: Any = None,
    ) -> None:
        """在后台顺序补发剩余分段，并在任务边界记录异常。"""
        send_ok = False
        try:
            # 先让 after_send 响应有机会回到宿主；启用时由宿主模拟打字承担可见等待。
            await asyncio.sleep(0)
            with _guard_stream_resend(stream_id):
                send_ok = await self._send_segments(
                    stream_id,
                    segments,
                    typing_enabled=typing_enabled,
                )
        except asyncio.CancelledError:
            logger.warning(
                "智能分段补发被取消，stream_id=%s message_id=%s 剩余 %s 段未发完",
                stream_id, message_id, len(segments),
            )
            raise
        except Exception as exc:
            logger.error(
                "智能分段补发异常，stream_id=%s message_id=%s: %s",
                stream_id, message_id, exc, exc_info=True,
            )
            return
        finally:
            _complete_planner_follow_up_entry(planner_entry, send_ok=send_ok)
        if not send_ok:
            logger.error("智能分段补发失败，stream_id=%s message_id=%s", stream_id, message_id)

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
            plugin_config = await self._get_plugin_config()
            plugin_enabled = bool(_get_nested_config_value(plugin_config, "plugin.enabled", True))
            segmentation_enabled = bool(_get_nested_config_value(plugin_config, "segmentation.enabled", True))
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
