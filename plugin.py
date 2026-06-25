"""智能分段插件。"""

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import re
import time
import tomllib

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
_prepared_segment_queue_by_stream: dict[str, list[dict[str, Any]]] = {}
_PREPARED_SEGMENT_TTL_SECONDS = 60.0

# 首段直发后，剩余分段在 after_send 阶段补发
_pending_follow_up_segments: dict[str, dict[str, Any]] = {}
_PENDING_FOLLOW_UP_TTL_SECONDS = 60.0

# 后台补发协程的句柄集合。
# 宿主对 send_service.after_send 这个 hook 强制 5000ms 超时（见
# src/services/send_service.py 的 default_timeout_ms=5000 与
# src/plugin_runtime/host/hook_dispatcher.py 里的 asyncio.wait_for），
# 即便是 OBSERVE 观察型也会被 cancel；如果我们直接在 hook 体里串行补发
# N 段（每段还要 sleep），到点就会被宿主取消，后半截内容彻底丢失。
# 解法是让 hook 自己在毫秒级返回，把真正的补发循环丢进 asyncio.create_task
# 后台执行——子任务和 hook 协程是独立的 Task，父被 cancel 不会连带 cancel 子。
_active_follow_up_tasks: set["asyncio.Task[Any]"] = set()
_active_follow_up_tasks_by_stream: dict[str, set["asyncio.Task[Any]"]] = {}
_follow_up_idle_events_by_stream: dict[str, "asyncio.Event"] = {}

# 插件自身补发时关闭二次分段
_stream_resend_guards: dict[str, int] = {}

# 命令执行期间禁止把命令回执误判为主回复
_active_command_streams: dict[str, int] = {}
_recent_command_stream_expiries: dict[str, float] = {}
# 仅兜住命令 hook 与 send_service 之间的轻微异步抖动。值过大会把命令后紧跟的正常主回复一起误伤
# (旧 90s 窗口就出现过重启后首个 @ 回复不分段的回归)，1.0s 已经足够覆盖 IPC 微抖动，
# 同时把误伤窗口比之前的 2.0s 缩短一半。
_COMMAND_REPLY_GRACE_SECONDS = 1.0

# maisaka 早期路径自己的 LLM 超时；这是当前唯一会做分段 LLM 调用的入口。
_REPLYER_PREPARED_SEGMENT_TIMEOUT_SECONDS = 12.0

# chat.receive.before_process 取宿主默认超时（hook_blocking_timeout_sec，部署里仅 30s），
# 不够覆盖多分段补发，这里显式放宽到 120s。
_FOLLOW_UP_WAIT_TIMEOUT_MS = 120_000
_FOLLOW_UP_PROMPT_BATCH_TTL_SECONDS = 120.0

# planner.before_request 构建 prompt 早于 hook 调用；这里保留最近补发批次，
# 让 hook 能在等待补发完成后把缺失分段补进本轮已构建的 prompt。
_follow_up_prompt_batches_by_stream: dict[str, list[dict[str, Any]]] = {}


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
    llm_timeout_sec: float = Field(default=12.0, description="分段 LLM 调用超时（秒），需要小于 hook 宿主超时")
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

# 与 MaiBot 宿主 ``src.chat.utils.utils.process_llm_response`` 的括号清理正则保持一致；
# 不在插件侧收窄匹配范围，否则 replyer 原文与 after_build 出站文本的缓存键会再次漂移。
_HOST_REPLY_POSTPROCESS_BRACKET_PATTERN = re.compile(r"[(\[（](?=.*[一-鿿]).*?[)\]）]")
_NON_TEXT_REPLY_PLACEHOLDER_PATTERN = re.compile(r"[(\[（]\s*表情包(?:\s*[:：].*?)?\s*[)\]）]")


def _strip_host_reply_postprocessed_bracket_content(text: str) -> str:
    """按宿主 ``process_llm_response`` 的括号清理规则归一化缓存键。"""
    return _HOST_REPLY_POSTPROCESS_BRACKET_PATTERN.sub("", str(text or ""))


def _strip_non_text_reply_placeholders(text: str) -> str:
    """移除不应作为纯文本补发的回复占位符。"""
    return _NON_TEXT_REPLY_PLACEHOLDER_PATTERN.sub("", str(text or ""))


def _normalize_response_text_for_key(text: str) -> str:
    """归一化用于查找预分段缓存的文本：剥 thinking + 对齐宿主后处理 + 折叠空白。"""
    cleaned = _strip_thinking_content(str(text or ""))
    cleaned = _strip_host_reply_postprocessed_bracket_content(cleaned)
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
    now = time.monotonic()
    if _prepared_segment_registry:
        expired_keys = [k for k, v in _prepared_segment_registry.items() if v.get("expires_at", 0.0) <= now]
        for key in expired_keys:
            _prepared_segment_registry.pop(key, None)

    if not _prepared_segment_queue_by_stream:
        return
    expired_streams: list[str] = []
    for stream_id, entries in list(_prepared_segment_queue_by_stream.items()):
        active_entries = [entry for entry in entries if entry.get("expires_at", 0.0) > now]
        if active_entries:
            _prepared_segment_queue_by_stream[stream_id] = active_entries
        else:
            expired_streams.append(stream_id)
    for stream_id in expired_streams:
        _prepared_segment_queue_by_stream.pop(stream_id, None)


def _store_prepared_segments(stream_id: str, response_text: str, segments: list[str]) -> bool:
    """登记早期路径预分段；返回是否成功登记。"""
    normalized_stream_id = str(stream_id or "").strip()
    text_hash = _hash_normalized_text(response_text)
    visible_segments = _normalize_prepared_segments_for_host_visibility(segments)
    if not normalized_stream_id or not text_hash or not visible_segments:
        return False

    _prune_expired_prepared_segments()
    expires_at = time.monotonic() + _PREPARED_SEGMENT_TTL_SECONDS
    entry = {
        "stream_id": normalized_stream_id,
        "text_hash": text_hash,
        "segments": visible_segments,
        "expires_at": expires_at,
    }
    _prepared_segment_registry[(normalized_stream_id, text_hash)] = entry
    _prepared_segment_queue_by_stream.setdefault(normalized_stream_id, []).append(entry)
    return True


def _remove_prepared_queue_entry(stream_id: str, entry: dict[str, Any]) -> None:
    entries = _prepared_segment_queue_by_stream.get(stream_id)
    if not entries:
        return
    remaining = [candidate for candidate in entries if candidate is not entry]
    if remaining:
        _prepared_segment_queue_by_stream[stream_id] = remaining
    else:
        _prepared_segment_queue_by_stream.pop(stream_id, None)


def _pop_prepared_segments(stream_id: str, outbound_text: str) -> list[str] | None:
    """命中即返回缓存里的分段并移除条目；未命中返回 None。

    首选用归一化文本 hash 精确匹配；若宿主在 replyer 与 send_service 之间做了
    错别字注入/标点后处理，出站文本会发生不可逆变化（例如“撂挑子”→“料挑子”
    并额外补发“撂”），hash 会漂移。此时使用同 stream 的 replyer 预分段队列消费
    下一条回复：队列唯一写入者是 ``maisaka.replyer.after_response``，因此它仍是
    生命周期级的精准匹配，不是对任意长文本的兜底分段。
    """
    _prune_expired_prepared_segments()
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return None

    text_hash = _hash_normalized_text(outbound_text)
    entry = None
    if text_hash:
        entry = _prepared_segment_registry.pop((normalized_stream_id, text_hash), None)
        if entry is not None:
            _remove_prepared_queue_entry(normalized_stream_id, entry)

    if entry is None:
        stream_entries = _prepared_segment_queue_by_stream.get(normalized_stream_id)
        if not stream_entries:
            return None
        entry = stream_entries.pop(0)
        if stream_entries:
            _prepared_segment_queue_by_stream[normalized_stream_id] = stream_entries
        else:
            _prepared_segment_queue_by_stream.pop(normalized_stream_id, None)
        entry_hash = str(entry.get("text_hash", "") or "").strip()
        if entry_hash:
            _prepared_segment_registry.pop((normalized_stream_id, entry_hash), None)

    segments = entry.get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    return list(segments)


def _normalize_prepared_segments_for_host_visibility(segments: list[str]) -> list[str]:
    """把预分段清洗成插件补发时应发送的可见文本段。"""
    visible_segments: list[str] = []
    for segment in segments:
        visible_segment = _strip_thinking_content(str(segment or ""))
        visible_segment = _strip_non_text_reply_placeholders(visible_segment).strip()
        if visible_segment:
            visible_segments.append(visible_segment)
    return visible_segments


def _normalize_segments(segments: Any, *, max_segments: int) -> list[str]:
    """规范化模型返回的分段结果。"""
    if not isinstance(segments, list):
        raise ValueError("模型返回的分段结果不是列表")

    normalized_segments = [str(segment).strip() for segment in segments if str(segment).strip()]
    if not normalized_segments:
        raise ValueError("模型返回的分段结果为空")

    return normalized_segments[:max_segments]


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


# === 后台补发任务 ===

def _get_follow_up_idle_event(stream_id: Any) -> "asyncio.Event":
    """返回指定聊天流的补发空闲事件；空闲时为 set。"""
    normalized_stream_id = str(stream_id or "").strip()
    idle_event = _follow_up_idle_events_by_stream.get(normalized_stream_id)
    if idle_event is None:
        idle_event = asyncio.Event()
        idle_event.set()
        if normalized_stream_id:
            _follow_up_idle_events_by_stream[normalized_stream_id] = idle_event
    return idle_event


def _get_active_follow_up_task_count(stream_id: Any) -> int:
    """返回指定聊天流正在运行的补发任务数，并顺手清理已完成句柄。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return 0

    stream_tasks = _active_follow_up_tasks_by_stream.get(normalized_stream_id)
    if not stream_tasks:
        return 0

    pending_tasks = {task for task in stream_tasks if not task.done()}
    if pending_tasks:
        if len(pending_tasks) != len(stream_tasks):
            _active_follow_up_tasks_by_stream[normalized_stream_id] = pending_tasks
        return len(pending_tasks)

    _active_follow_up_tasks_by_stream.pop(normalized_stream_id, None)
    _get_follow_up_idle_event(normalized_stream_id).set()
    return 0


async def _wait_for_stream_follow_up_tasks(stream_id: Any) -> int:
    """等待指定聊天流的补发任务全部结束，返回等待前的任务数。"""
    pending_task_count = _get_active_follow_up_task_count(stream_id)
    if pending_task_count <= 0:
        return 0

    await _get_follow_up_idle_event(stream_id).wait()
    return pending_task_count


def _track_follow_up_task(task: "asyncio.Task[Any]", *, stream_id: Any) -> None:
    """登记后台补发任务并在结束时自动摘除。

    必须把句柄挂到模块级集合里：``asyncio.create_task`` 只持有弱引用，
    如果调用方不留引用，GC 一发生任务就可能被收掉、补发中断。
    """
    _active_follow_up_tasks.add(task)
    normalized_stream_id = str(stream_id or "").strip()
    if normalized_stream_id:
        _active_follow_up_tasks_by_stream.setdefault(normalized_stream_id, set()).add(task)
        _get_follow_up_idle_event(normalized_stream_id).clear()

    def _cleanup(completed_task: "asyncio.Task[Any]") -> None:
        _active_follow_up_tasks.discard(completed_task)
        if not normalized_stream_id:
            return

        stream_tasks = _active_follow_up_tasks_by_stream.get(normalized_stream_id)
        if stream_tasks is None:
            _get_follow_up_idle_event(normalized_stream_id).set()
            return

        stream_tasks.discard(completed_task)
        if stream_tasks:
            return

        _active_follow_up_tasks_by_stream.pop(normalized_stream_id, None)
        _get_follow_up_idle_event(normalized_stream_id).set()

    task.add_done_callback(_cleanup)


async def _drain_active_follow_up_tasks() -> None:
    """等待所有在跑的后台补发任务结束，仅用于卸载与测试同步。"""
    pending_tasks = [task for task in list(_active_follow_up_tasks) if not task.done()]
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)


def _prune_expired_follow_up_prompt_batches() -> None:
    """清理 planner prompt 注入批次，避免过期分段污染后续对话。"""
    if not _follow_up_prompt_batches_by_stream:
        return

    now = time.monotonic()
    expired_streams: list[str] = []
    for stream_id, batches in list(_follow_up_prompt_batches_by_stream.items()):
        active_batches = [batch for batch in batches if batch.get("expires_at", 0.0) > now]
        if active_batches:
            _follow_up_prompt_batches_by_stream[stream_id] = active_batches
        else:
            expired_streams.append(stream_id)
    for stream_id in expired_streams:
        _follow_up_prompt_batches_by_stream.pop(stream_id, None)


def _register_follow_up_prompt_batch(stream_id: Any, segments: list[str]) -> None:
    """登记一批补发段，供下一轮 planner hook 校验 prompt 是否缺段。"""
    normalized_stream_id = str(stream_id or "").strip()
    normalized_segments = [str(segment or "").strip() for segment in segments if str(segment or "").strip()]
    if not normalized_stream_id or not normalized_segments:
        return

    _prune_expired_follow_up_prompt_batches()
    _follow_up_prompt_batches_by_stream.setdefault(normalized_stream_id, []).append(
        {
            "segments": normalized_segments,
            "expires_at": time.monotonic() + _FOLLOW_UP_PROMPT_BATCH_TTL_SECONDS,
            "consumed": False,
        }
    )


def _get_unconsumed_follow_up_prompt_segments(stream_id: Any) -> list[str]:
    """返回同一聊天流尚未被 planner prompt 消费的补发段。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return []

    _prune_expired_follow_up_prompt_batches()
    batches = _follow_up_prompt_batches_by_stream.get(normalized_stream_id, [])
    segments: list[str] = []
    for batch in batches:
        if batch.get("consumed"):
            continue
        raw_segments = batch.get("segments")
        if not isinstance(raw_segments, list):
            continue
        for raw_segment in raw_segments:
            segment = str(raw_segment or "").strip()
            if segment and segment not in segments:
                segments.append(segment)
    return segments


def _mark_follow_up_prompt_batches_consumed(stream_id: Any) -> None:
    """标记同流补发段已经进入某一轮 planner prompt，不再重复注入。"""
    normalized_stream_id = str(stream_id or "").strip()
    if not normalized_stream_id:
        return

    for batch in _follow_up_prompt_batches_by_stream.get(normalized_stream_id, []):
        batch["consumed"] = True


def _prompt_content_to_text(content: Any) -> str:
    """把 Hook 序列化后的 prompt content 归一成可查找的文本。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    text_parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict):
            item_text = item.get("text") or item.get("content")
            if item_text:
                text_parts.append(str(item_text))
        elif isinstance(item, (list, tuple)) and item:
            # serialize_prompt_messages 对图片片段使用 tuple/list；这里只关心文本项。
            if isinstance(item[0], str) and len(item) == 1:
                text_parts.append(str(item[0]))
    return "".join(text_parts)


def _normalize_prompt_search_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _prompt_message_looks_like_self_message(message: dict[str, Any]) -> bool:
    role = str(message.get("role", "") or "").strip().lower()
    if role == "assistant":
        return True
    content_text = _prompt_content_to_text(message.get("content"))
    return 'is_self_message="true"' in content_text or "is_self_message='true'" in content_text


def _prompt_messages_contain_segments(messages: list[dict[str, Any]], segments: list[str]) -> bool:
    """判断当前 prompt 中是否已经包含所有补发段。"""
    normalized_segments = [_normalize_prompt_search_text(segment) for segment in segments if segment]
    if not normalized_segments:
        return True

    self_message_texts: list[str] = []
    all_message_texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized_text = _normalize_prompt_search_text(_prompt_content_to_text(message.get("content")))
        if not normalized_text:
            continue
        all_message_texts.append(normalized_text)
        if _prompt_message_looks_like_self_message(message):
            self_message_texts.append(normalized_text)

    search_text = "\n".join(self_message_texts or all_message_texts)
    return all(segment in search_text for segment in normalized_segments)


def _build_follow_up_prompt_message(segments: list[str]) -> dict[str, Any]:
    """构造与 Maisaka 自身消息标注一致的合成历史消息。"""
    joined_segments = "\n".join(str(segment or "").strip() for segment in segments if str(segment or "").strip())
    return {
        "role": "assistant",
        "content": f'<message user="MaiBot" is_self_message="true">\n{joined_segments}',
    }


def _inject_follow_up_segments_into_prompt_messages(
    messages: list[dict[str, Any]],
    *,
    selected_history_count: Any,
    segments: list[str],
) -> list[dict[str, Any]]:
    """把补发段插入到已选历史之后、当前轮注入消息之前。"""
    updated_messages = [dict(message) if isinstance(message, dict) else message for message in messages]
    try:
        history_count = max(0, int(selected_history_count))
    except (TypeError, ValueError):
        history_count = 0
    insert_index = min(len(updated_messages), max(1, history_count + 1))
    updated_messages.insert(insert_index, _build_follow_up_prompt_message(segments))
    return updated_messages


# === 命令保护窗口 ===

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


def _presync_follow_up_segments_to_maisaka_history(
    *,
    stream_id: str,
    first_message_dict: dict[str, Any],
    follow_up_segments: list[str],
) -> bool:
    """把所有剩余分段一次性预先同步进 maisaka `_chat_history`。

    背景：reply 工具不设置 `pause_execution`（见 reasoning_engine.py:2001-2005），
    工具调用返回后 reasoning 主循环立刻 `continue` 进入下一轮 planner。如果剩余
    分段还在后台 task 的 sleep 里没发出去，maisaka 历史只能看到首段——下一轮
    planner 会判定"还没说完"再调一次 reply 工具，造成对同一条用户消息重复回复。

    解法：拿到首段已发送的 SessionMessage 后，立刻克隆出 N 份只改写 raw_message /
    processed_plain_text / message_id 的伪 SessionMessage，**调宿主侧
    `send_service._sync_sent_message_to_maisaka_history`** 让它走跟首段同步完全
    一致的代码路径写入历史；然后才启动后台 task 真正把这些段发到用户客户端，且
    发送时 ``sync_to_maisaka_history=False`` 避免重复入库。

    为什么用宿主 send_service 内部函数而不是直接拿 runtime 自己 append：之前那条
    `heartflow_manager.heartflow_chat_list.get(stream_id)` 在生产里观察到返回
    None（详见 fix 历史），但 reply 工具走 send_service 链路同步首段是 work 的。
    我们让剩余段复用宿主同样的入口，从而 work 的概率最大。如果走宿主同步函数仍
    然失败，那 lookup 失败是宿主级问题，插件无法绕开——此时返回 False 让后台
    task 退化到原有的 send_service 异步同步路径，至少不丢段。
    """
    if not stream_id or not follow_up_segments or not isinstance(first_message_dict, dict):
        return False

    try:
        # 这些 import 都是宿主内部 API；插件本身已经在 import src.config / src.llm_models，
        # 这里维持同样的耦合层级，便于跟着宿主版本一起演进。
        from src.chat.heart_flow.heartflow_manager import heartflow_manager
        from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
        from src.plugin_runtime.hook_payloads import deserialize_session_message
        from src.services.send_service import _sync_sent_message_to_maisaka_history as _host_sync_sent
    except ImportError as exc:
        logger.warning("智能分段无法导入宿主历史同步依赖，将退化为旧的异步同步路径: %s", exc)
        return False

    runtime = heartflow_manager.heartflow_chat_list.get(stream_id)
    if runtime is None:
        return False

    try:
        first_session_message = deserialize_session_message(first_message_dict)
    except Exception as exc:
        logger.warning("智能分段反序列化首段 SessionMessage 失败，退化到异步同步路径: %s", exc)
        return False

    base_message_id = str(getattr(first_session_message, "message_id", "") or "").strip()
    synced_count = 0
    for index, segment_text in enumerate(follow_up_segments, start=1):
        normalized_segment = str(segment_text or "").strip()
        if not normalized_segment:
            continue
        try:
            # 浅拷贝足够：append_sent_message_to_chat_history 只读 message_info / timestamp /
            # message_id / is_notify / raw_message，我们只改自己关心的三个字段，其它字段
            # 跟首段共享引用不会被修改。
            cloned_message = copy.copy(first_session_message)
            cloned_message.raw_message = MessageSequence([TextComponent(normalized_segment)])
            cloned_message.processed_plain_text = normalized_segment
            # 给每段一个稳定但不冲突的 message_id：避免与首段同 ID 让历史合并丢段。
            cloned_message.message_id = (
                f"{base_message_id}_seg{index}" if base_message_id else f"smartseg_{stream_id}_{index}"
            )
            # 走宿主 send_service 内部的同步入口：跟首段完全相同的代码路径，
            # 既然首段在这条路上能成功 append，剩余段也必然能。该函数无返回值，
            # 异常会被它自己 catch 并打 warning；我们用 lookup 成功＋调用未抛异常
            # 作为成功标记。
            _host_sync_sent(cloned_message, source_kind="guided_reply")
            synced_count += 1
        except Exception as exc:
            logger.warning(
                "智能分段预同步第 %s 段进 maisaka 历史失败: %s",
                index,
                exc,
            )

    if synced_count > 0:
        logger.info(
            "智能分段已通过宿主 send_service 预先同步 %s/%s 段进 maisaka 历史 stream=%s",
            synced_count,
            len(follow_up_segments),
            stream_id,
        )
    return synced_count == len(follow_up_segments)


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
        _prepared_segment_queue_by_stream.clear()
        _pending_follow_up_segments.clear()
        _stream_resend_guards.clear()
        _active_command_streams.clear()
        _recent_command_stream_expiries.clear()
        # 重新加载时清空后台任务集合：reload 前的任务此时已经持不到新插件实例的 ctx，
        # 即便仍在跑也无法把消息发出去。直接放手让它们随旧实例的事件循环自然回收。
        _active_follow_up_tasks.clear()
        _active_follow_up_tasks_by_stream.clear()
        _follow_up_idle_events_by_stream.clear()
        _follow_up_prompt_batches_by_stream.clear()
        if not _SDK_HOOK_HANDLER_AVAILABLE:
            logger.info("当前 maibot_sdk 未导出 HookHandler，智能分段已启用内置 hook_handler 声明兼容")

    async def on_unload(self) -> None:
        """处理插件卸载。"""
        # 卸载前先取消所有在跑的后台补发任务，再等待它们结束。
        # 不取消的话，任务仍持有 self.ctx 的引用，宿主侧 send 通道可能已经销毁，
        # 任务会以异常方式失败但日志却落在卸载之后，难以排查。
        for task in list(_active_follow_up_tasks):
            if not task.done():
                task.cancel()
        await _drain_active_follow_up_tasks()
        _active_follow_up_tasks.clear()
        _active_follow_up_tasks_by_stream.clear()
        _follow_up_idle_events_by_stream.clear()
        _follow_up_prompt_batches_by_stream.clear()
        _prepared_segment_registry.clear()
        _prepared_segment_queue_by_stream.clear()
        _pending_follow_up_segments.clear()
        _stream_resend_guards.clear()
        _active_command_streams.clear()
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
- 括号（中文「（）」「【】」或英文「()」「[]」）内的内容（动作、神态、旁白等描述）必须作为独立的一条消息单独发送，不要和括号外的正文合在同一条
- 括号内的内容本身不能再拆开，需保持完整
- 如果整段内容就是被括号包裹的动作/神态描述，直接整段返回不再切分
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
            normalized = _normalize_segments(segments, max_segments=max_segments)
            # 先合并被模型误拆的括号对，确保每段括号成对；再按括号边界把动作描述拆成独立段。
            balanced = _merge_segments_balancing_brackets(normalized)
            return _split_segments_at_bracket_boundaries(balanced, max_segments=max_segments)
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
        sync_to_maisaka_history: bool = True,
    ) -> bool:
        """逐条发送分段结果。

        ``sync_to_maisaka_history`` 默认 True 是为了让旧的"send_service 同步写历史"
        路径继续生效；当上层（after_send）已经通过
        ``_presync_follow_up_segments_to_maisaka_history`` 把所有剩余段同步进
        历史后，应该把这个参数置为 False，避免同一段被记两次。
        """
        for index, segment in enumerate(segments):
            if index > 0 or delay_before_first:
                await asyncio.sleep(self._calculate_send_delay(segment, delay_base, delay_per_char, delay_max))

            # source_kind 与 maisaka 自带的 reply 工具保持一致，让
            # SessionBackedMessage 走 include_reply_components=False 的渲染。
            send_ok = await self.ctx.send.text(
                segment,
                stream_id,
                sync_to_maisaka_history=sync_to_maisaka_history,
                maisaka_source_kind="guided_reply",
            )
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

        llm_timeout_sec_raw = await self._get_config_value("segmentation.llm_timeout_sec", 12.0)
        try:
            llm_timeout_sec = float(llm_timeout_sec_raw)
        except (TypeError, ValueError):
            llm_timeout_sec = 12.0

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
            "llm_timeout_sec": llm_timeout_sec,
        }

    @HookHandler(
        "maisaka.replyer.after_response",
        name="smart_segmentation_after_replyer_response",
        description="在 Maisaka replyer 拿到模型回复后立刻预分段，把结果登记到进程内缓存，发送链可零 LLM 调用直接消费",
        timeout_ms=30_000,
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
                self._segment_text(
                    visible_text,
                    style=settings["style"],
                    model_name=settings["model_name"],
                    max_segments=settings["max_segments"],
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                ),
                timeout=settings["llm_timeout_sec"],
            )
        except asyncio.TimeoutError:
            logger.warning(
                "智能分段在 replyer.after_response 阶段超时（> %.2fs），本次回复将放行原文，发送链不会同步重试分段",
                settings["llm_timeout_sec"],
            )
            return {"action": "continue"}

        if not segments or len(segments) <= 1:
            return {"action": "continue"}

        visible_segments = _normalize_prepared_segments_for_host_visibility(segments)
        if not visible_segments or len(visible_segments) <= 1:
            return {"action": "continue"}

        # 缓存按归一化后的原文 hash 索引；后续 after_build_message 用同样规则做查找。
        if _store_prepared_segments(normalized_stream_id, normalized_response, visible_segments):
            logger.info(
                "智能分段已在 replyer.after_response 阶段预切分，共 %s 段，已登记到缓存 stream=%s",
                len(visible_segments),
                normalized_stream_id,
            )
        return {"action": "continue"}

    @HookHandler(
        "maisaka.planner.before_request",
        name="smart_segmentation_planner_follow_up_barrier",
        description="同一聊天流仍有智能分段补发未完成或当前 prompt 缺少补发段时，阻塞并把缺失的 bot 自身分段补入本轮 planner prompt",
        timeout_ms=_FOLLOW_UP_WAIT_TIMEOUT_MS,
        order="early",
    )
    async def handle_maisaka_planner_before_request(
        self,
        messages: list[dict[str, Any]] | None = None,
        session_id: str = "",
        selected_history_count: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """挡住 Maisaka 内部连续 planner 在分段补发期间抢跑。

        ``chat.receive.before_process`` 只能阻塞新入站消息；重复回复的现场发生在
        reply 工具返回后的同一轮内部 continue，不会经过入站 hook。这里恢复到
        Maisaka planner 生命周期：先等待同流补发任务结束，再检查本轮已经构建好的
        prompt 是否包含补发段。由于宿主是在调用 hook 前构建 prompt，单纯等待不会让
        prompt 自动重建；缺段时必须通过 modified_kwargs.messages 精准补入。
        """
        del kwargs

        normalized_stream_id = str(session_id or "").strip()
        if not normalized_stream_id:
            return {"action": "continue"}

        if _get_active_follow_up_task_count(normalized_stream_id) > 0:
            await _wait_for_stream_follow_up_tasks(normalized_stream_id)

        follow_up_segments = _get_unconsumed_follow_up_prompt_segments(normalized_stream_id)
        if not follow_up_segments:
            return {"action": "continue"}

        if not isinstance(messages, list):
            return {"action": "continue"}

        if _prompt_messages_contain_segments(messages, follow_up_segments):
            _mark_follow_up_prompt_batches_consumed(normalized_stream_id)
            return {"action": "continue"}

        updated_messages = _inject_follow_up_segments_into_prompt_messages(
            messages,
            selected_history_count=selected_history_count,
            segments=follow_up_segments,
        )
        _mark_follow_up_prompt_batches_consumed(normalized_stream_id)
        logger.info(
            "智能分段已在 planner.before_request 注入 %s 条补发段，避免内部 planner 抢跑 stream=%s",
            len(follow_up_segments),
            normalized_stream_id,
        )
        return {"action": "continue", "modified_kwargs": {"messages": updated_messages}}

    @HookHandler(
        "chat.receive.before_process",
        name="smart_segmentation_pause_until_follow_ups_finish",
        description="同一聊天流仍有智能分段补发未完成时，先阻塞入站消息处理，避免 planner 只看到首段就重复回复",
        timeout_ms=_FOLLOW_UP_WAIT_TIMEOUT_MS,
        order="early",
    )
    async def handle_chat_receive_before_process(
        self,
        message: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """在新入站消息进入处理链前，阻塞当前流直到智能分段补发任务结束。

        这只覆盖“用户又发来新消息”的路径；同一轮 Maisaka 内部 continue 会绕过入站
        hook，因此真正防止 planner 抢跑的是 ``maisaka.planner.before_request`` 屏障。
        两个 hook 分别保护外部入站与内部连续 planner，避免只看到首段就重复回复。
        """
        del kwargs

        if not isinstance(message, dict):
            return {"action": "continue"}

        normalized_stream_id = str(message.get("session_id", "") or "").strip()
        if not normalized_stream_id:
            return {"action": "continue"}

        if _get_active_follow_up_task_count(normalized_stream_id) > 0:
            await _wait_for_stream_follow_up_tasks(normalized_stream_id)

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

        # 提前在 BLOCKING handler 中登记补发段，避免 after_send 的 OBSERVE
        # 后台任务竞态导致 planner.before_request 屏障看不到已注册的段。
        _register_follow_up_prompt_batch(normalized_stream_id, follow_up_segments)

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
        logger.info(
            "智能分段命中 replyer 预分段缓存，首段直发，登记 %s 条补发消息 stream=%s",
            len(follow_up_segments),
            normalized_stream_id,
        )
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
        """在首段发送成功后异步补发其余分段。

        宿主对 ``send_service.after_send`` 强制 5000ms 超时，且观察型 hook
        同样会被 ``asyncio.wait_for`` cancel；过去把补发循环直接 ``await``
        在 hook 体里，发到一半就被宿主取消，多分段消息后半截内容会丢。
        这里只在 hook 协程里做查询与登记，真正的串行补发用 ``create_task``
        丢给后台事件循环——hook 体一返回，宿主就不再监管这个子协程的耗时。
        """
        del kwargs

        if not isinstance(message, dict):
            return None

        if not sent:
            return None

        message_id = str(message.get("message_id", "") or "").strip()
        normalized_stream_id = str(message.get("session_id", "") or "").strip()
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

        # 先把所有剩余段一次性预同步进 maisaka 历史，再启动后台真正发送任务。
        # 注意：_register_follow_up_prompt_batch 已在 after_build_message (BLOCKING) 中调用，
        # 此处不再重复注册。
        # 这是为了挡住下面这条会触发"对同一条用户消息重复回复"的回归路径：
        # reply 工具不设 pause_execution → reasoning 主循环 continue → 下一轮
        # planner 立即启动 → 此时后台 task 还在 sleep，maisaka 历史只看到首段
        # → 模型判定"还没说完"再调一次 reply 工具。
        history_synced = await self._presync_via_context_append(
            stream_id=stream_id,
            segments=list(segments),
            base_message_id=message_id,
        )

        follow_up_task = asyncio.create_task(
            self._run_follow_up_segments(
                stream_id=stream_id,
                segments=list(segments),
                delay_base=delay_base,
                delay_per_char=delay_per_char,
                delay_max=delay_max,
                message_id=message_id,
                # 预同步成功后，后台实际发送时就不要再让 send_service 重复入库——
                # 否则同一段会在 maisaka 历史里出现两次。
                sync_to_maisaka_history=not history_synced,
            )
        )
        _track_follow_up_task(follow_up_task, stream_id=stream_id)
        return None

    async def _presync_via_context_append(
        self,
        *,
        stream_id: str,
        segments: list[str],
        base_message_id: str = "",
    ) -> bool:
        """通过 ``ctx.maisaka.context.append`` 将剩余分段预先同步进 Maisaka 历史。

        与 ``_presync_follow_up_segments_to_maisaka_history`` 不同，此方法使用
        SDK 的 ``maisaka.context.append`` 能力，内部走 ``get_or_create_heartflow_chat``
        而非 ``heartflow_chat_list.get``，因此不会因 runtime 未缓存而静默失败。
        """
        if not stream_id or not segments:
            return False

        synced_count = 0
        for index, segment_text in enumerate(segments, start=1):
            normalized = str(segment_text or "").strip()
            if not normalized:
                continue
            # 给每段一个稳定但不冲突的 message_id，避免 wait 超时
            # 后新周期 planner 因缺少 msg_id 误判为"尚未发出的草稿"。
            synthetic_message_id = (
                f"{base_message_id}_seg{index}"
                if base_message_id
                else f"smartseg_{stream_id}_{index}"
            )
            try:
                result = await self.ctx.maisaka.context.append(
                    stream_id,
                    [{"type": "text", "data": normalized}],
                    source_kind="guided_reply",
                    message_id=synthetic_message_id,
                )
                if isinstance(result, dict) and result.get("success"):
                    synced_count += 1
            except Exception as exc:
                logger.warning(
                    "智能分段通过 ctx.maisaka.context.append 预同步失败: %s",
                    exc,
                )

        if synced_count > 0:
            logger.info(
                "智能分段已通过 ctx.maisaka.context.append 预先同步 %s/%s 段进 maisaka 历史 stream=%s",
                synced_count,
                len(segments),
                stream_id,
            )
        return synced_count == len(segments)

    async def _run_follow_up_segments(
        self,
        *,
        stream_id: str,
        segments: list[str],
        delay_base: float,
        delay_per_char: float,
        delay_max: float,
        message_id: str,
        sync_to_maisaka_history: bool = True,
    ) -> None:
        """在后台串行补发剩余分段，并兜住所有异常防止泄漏。"""
        try:
            with _guard_stream_resend(stream_id):
                send_ok = await self._send_segments(
                    stream_id,
                    segments,
                    delay_base=delay_base,
                    delay_per_char=delay_per_char,
                    delay_max=delay_max,
                    delay_before_first=True,
                    sync_to_maisaka_history=sync_to_maisaka_history,
                )
        except asyncio.CancelledError:
            logger.warning(
                "智能分段后台补发任务被取消，可能是插件卸载导致，stream_id=%s message_id=%s 剩余 %s 段未发完",
                stream_id,
                message_id,
                len(segments),
            )
            raise
        except Exception as exc:
            logger.error(
                "智能分段后台补发任务异常，stream_id=%s message_id=%s: %s",
                stream_id,
                message_id,
                exc,
                exc_info=True,
            )
            return

        if not send_ok:
            logger.error("智能分段补发失败，stream_id=%s message_id=%s", stream_id, message_id)
            return

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
