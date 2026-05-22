import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.smart_segmentation_plugin import plugin as plugin_module
from plugins.smart_segmentation_plugin.plugin import (
    _COMMAND_REPLY_GRACE_SECONDS,
    SmartSegmentationPlugin,
    _extract_plain_text_outbound_message,
    _get_command_stream_grace_remaining,
    _hash_normalized_text,
    _is_command_stream_active,
    _mark_command_stream_active,
    _mark_command_stream_inactive,
    _pop_prepared_segments,
    _replace_outbound_text,
    _store_prepared_segments,
)


def _reset_global_state() -> None:
    plugin_module._prepared_segment_registry.clear()
    plugin_module._pending_follow_up_segments.clear()
    plugin_module._stream_resend_guards.clear()
    plugin_module._active_command_streams.clear()
    plugin_module._recent_command_stream_expiries.clear()


def test_get_components_registers_hook_handlers_for_current_host() -> None:
    plugin = SmartSegmentationPlugin()

    components = plugin.get_components()
    hook_component_type = "HOOK_HANDLER" if plugin_module._SDK_HOOK_HANDLER_AVAILABLE else "hook_handler"
    hook_components = {
        component["name"]: component for component in components if component.get("type") == hook_component_type
    }

    assert "smart_segmentation_after_build" in hook_components
    assert hook_components["smart_segmentation_after_build"]["metadata"]["hook"] == "send_service.after_build_message"
    assert "smart_segmentation_after_send" in hook_components
    assert hook_components["smart_segmentation_after_send"]["metadata"]["hook"] == "send_service.after_send"
    assert "smart_segmentation_command_scope_enter" in hook_components
    assert hook_components["smart_segmentation_command_scope_enter"]["metadata"]["hook"] == "chat.command.before_execute"
    assert "smart_segmentation_command_scope_leave" in hook_components
    assert hook_components["smart_segmentation_command_scope_leave"]["metadata"]["hook"] == "chat.command.after_execute"
    # 早期路径是唯一的分段入口；没有这个 hook，发送链上的缓存查找永远 miss，插件等于失效。
    assert "smart_segmentation_after_replyer_response" in hook_components
    assert (
        hook_components["smart_segmentation_after_replyer_response"]["metadata"]["hook"]
        == "maisaka.replyer.after_response"
    )


def test_store_and_pop_prepared_segments_use_normalized_text_hash() -> None:
    _reset_global_state()
    try:
        ok = _store_prepared_segments(
            "stream-cache",
            "  你好啊\n这是一段被    奇怪空白污染的回复",
            ["你好啊", "这是一段被 奇怪空白污染的回复"],
        )
        assert ok

        # 文本里的多空白/换行差异不应影响命中
        segments = _pop_prepared_segments("stream-cache", "你好啊 这是一段被 奇怪空白污染的回复")
        assert segments == ["你好啊", "这是一段被 奇怪空白污染的回复"]

        # 命中后立刻移除，二次取应该 miss
        assert _pop_prepared_segments("stream-cache", "你好啊 这是一段被 奇怪空白污染的回复") is None
    finally:
        _reset_global_state()


def test_prepared_segments_isolated_across_streams() -> None:
    _reset_global_state()
    try:
        _store_prepared_segments("stream-a", "同一段文本", ["前", "后"])
        # 不同 stream 不应命中
        assert _pop_prepared_segments("stream-b", "同一段文本") is None
        assert _pop_prepared_segments("stream-a", "同一段文本") == ["前", "后"]
    finally:
        _reset_global_state()


def test_prepared_segments_expire_after_ttl() -> None:
    _reset_global_state()
    try:
        with patch.object(plugin_module.time, "monotonic", return_value=0.0):
            _store_prepared_segments("stream-ttl", "段一段二", ["段一", "段二"])

        # TTL 之外不再命中，且过期键会被自动清理掉
        with patch.object(
            plugin_module.time,
            "monotonic",
            return_value=plugin_module._PREPARED_SEGMENT_TTL_SECONDS + 1.0,
        ):
            assert _pop_prepared_segments("stream-ttl", "段一段二") is None
        assert not plugin_module._prepared_segment_registry
    finally:
        _reset_global_state()


def test_hash_normalized_text_is_whitespace_insensitive() -> None:
    assert _hash_normalized_text("你好啊  哈哈") == _hash_normalized_text(" 你好啊\n哈哈 ")
    assert _hash_normalized_text("") == ""


def test_extract_plain_text_allows_real_at_components_for_fallback_segmentation() -> None:
    message = {
        "raw_message": [
            {"type": "text", "data": "大家好我叫理理，小学六年级，喜欢画画和打游戏，数学不太行但美术课从来没输过！就这样啦够了吧"},
            {
                "type": "at",
                "data": {
                    "target_user_id": "10001",
                    "target_user_nickname": "久远",
                    "target_user_cardname": "久远",
                },
            },
        ],
    }

    assert (
        _extract_plain_text_outbound_message(message)
        == "大家好我叫理理，小学六年级，喜欢画画和打游戏，数学不太行但美术课从来没输过！就这样啦够了吧@久远"
    )


def test_extract_plain_text_still_rejects_media_components() -> None:
    message = {
        "raw_message": [
            {"type": "text", "data": "这段文字足够长，理论上可以分段"},
            {"type": "image", "data": "[图片]"},
        ],
    }

    assert _extract_plain_text_outbound_message(message) == ""


def test_replace_outbound_text_preserves_real_at_component_when_first_segment_keeps_it() -> None:
    message = {
        "raw_message": [
            {"type": "text", "data": "就这样啦够了吧"},
            {
                "type": "at",
                "data": {
                    "target_user_id": "10001",
                    "target_user_nickname": "久远",
                    "target_user_cardname": "久远",
                },
            },
        ],
        "message_info": {
            "additional_config": {
                "platform_io_target_group_id": "20001",
            },
        },
    }

    updated_message = _replace_outbound_text(message, "就这样啦够了吧@久远")

    assert updated_message["raw_message"] == [
        {"type": "text", "data": "就这样啦够了吧"},
        {
            "type": "at",
            "data": {
                "target_user_id": "10001",
                "target_user_nickname": "久远",
                "target_user_cardname": "久远",
            },
        },
    ]
    assert updated_message["processed_plain_text"] == "就这样啦够了吧@久远"
    assert updated_message["display_message"] == "就这样啦够了吧@久远"
    # additional_config 里只剩跟分段无关的字段
    assert updated_message["message_info"]["additional_config"] == {"platform_io_target_group_id": "20001"}


def test_after_send_uses_visible_first_segment_when_reply_component_rewrites_processed_text() -> None:
    plugin = SmartSegmentationPlugin()
    tracking_key = plugin_module._build_follow_up_tracking_key(
        stream_id="stream-reply",
        timestamp="1000.0",
        visible_text="第一段",
    )
    plugin_module._register_pending_follow_up_segments(
        lookup_keys=["send_api_1000", tracking_key],
        pending_data={
            "stream_id": "stream-reply",
            "segments": ["第二段", "第三段"],
            "delay_base": 0.1,
            "delay_per_char": 0.2,
            "delay_max": 0.3,
        },
    )

    send_segments_mock = AsyncMock(return_value=True)
    try:
        with patch.object(plugin, "_send_segments", send_segments_mock):
            asyncio.run(
                plugin.handle_smart_segmentation_after_send(
                    message={
                        "message_id": "platform-msg-1000",
                        "session_id": "stream-reply",
                        "timestamp": "1000.0",
                        "processed_plain_text": "原消息 第一段",
                        "raw_message": [
                            {
                                "type": "reply",
                                "data": {
                                    "target_message_id": "origin-1",
                                    "target_message_content": "原消息",
                                },
                            },
                            {"type": "text", "data": "第一段"},
                        ],
                    },
                    sent=True,
                )
            )

        send_segments_mock.assert_awaited_once_with(
            "stream-reply",
            ["第二段", "第三段"],
            delay_base=0.1,
            delay_per_char=0.2,
            delay_max=0.3,
            delay_before_first=True,
        )
        assert not plugin_module._pending_follow_up_segments
    finally:
        _reset_global_state()


def test_after_build_skips_when_no_prepared_cache_hit() -> None:
    """没有 maisaka.replyer.after_response 写入的预分段缓存时，发送链不应做任何 LLM 调用。

    这是修复 "插件对非回复模型产出的文本也做分段" 的关键回归点：之前会落到
    after_build 兜底分段路径，把 memory/expression/插件 ctx.send.text 等任何
    长文本都误判成主回复。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 8,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "",
        "delay_base": 0.35,
        "delay_per_char": 0.015,
        "delay_max": 1.2,
    }
    segment_text_mock = AsyncMock(return_value=["不应被调用"])

    with (
        patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
        patch.object(plugin, "_segment_text", segment_text_mock),
    ):
        result = asyncio.run(
            plugin.handle_smart_segmentation_after_build(
                message={
                    "message_id": "log-msg-1",
                    "session_id": "stream-log",
                    "timestamp": "2000.0",
                    "raw_message": [
                        {
                            "type": "text",
                            "data": "INFO 2026-05-07 20:26:24 plugin.smart_segmentation 智能分段已消费预分段标记",
                        }
                    ],
                    "message_info": {"additional_config": {}},
                },
                stream_id="stream-log",
                processed_plain_text="INFO 2026-05-07 20:26:24 plugin.smart_segmentation 智能分段已消费预分段标记",
            )
        )

    segment_text_mock.assert_not_awaited()
    assert result == {"action": "continue"}


def test_after_build_does_not_segment_reply_to_message_without_prepared_cache() -> None:
    """reply_to 单独存在不再触发分段——只有 maisaka.replyer.after_response 写入缓存的消息才会被分段。

    这是修复 "插件对非回复模型产出的文本也做分段" 的核心回归点。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 8,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "",
        "delay_base": 0.35,
        "delay_per_char": 0.015,
        "delay_max": 1.2,
    }
    segment_text_mock = AsyncMock(return_value=["不应被调用"])

    try:
        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            result = asyncio.run(
                plugin.handle_smart_segmentation_after_build(
                    message={
                        "message_id": "reply-msg-1",
                        "session_id": "stream-reply",
                        "timestamp": "3000.0",
                        "reply_to": "origin-1",
                        "raw_message": [
                            {
                                "type": "text",
                                "data": "这是一段足够长的插件 reply_to 文本，但不是来自回复模型，不应被分段。",
                            }
                        ],
                        "message_info": {"additional_config": {}},
                    },
                    stream_id="stream-reply",
                    processed_plain_text="这是一段足够长的插件 reply_to 文本，但不是来自回复模型，不应被分段。",
                )
            )

        segment_text_mock.assert_not_awaited()
        assert result == {"action": "continue"}
        assert not plugin_module._pending_follow_up_segments
    finally:
        _reset_global_state()


def test_after_build_does_not_segment_plain_bot_text_without_prepared_cache() -> None:
    """没有 reply_to 也没有 prepared_cache 的纯文本同样不应被分段。

    这是修复 "插件对任何非回复模型文本都做分段" 的回归点。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 8,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "",
        "delay_base": 0.35,
        "delay_per_char": 0.015,
        "delay_max": 1.2,
    }
    segment_text_mock = AsyncMock(return_value=["不应被调用"])

    try:
        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            result = asyncio.run(
                plugin.handle_smart_segmentation_after_build(
                    message={
                        "message_id": "plain-msg-1",
                        "session_id": "stream-plain",
                        "timestamp": "4000.0",
                        "raw_message": [
                            {
                                "type": "text",
                                "data": "好耶今天真的还挺开心的，晚点我再慢慢跟你讲。",
                            }
                        ],
                        "message_info": {"additional_config": {}},
                    },
                    stream_id="stream-plain",
                    processed_plain_text="好耶今天真的还挺开心的，晚点我再慢慢跟你讲。",
                )
            )

        segment_text_mock.assert_not_awaited()
        assert result == {"action": "continue"}
        assert not plugin_module._pending_follow_up_segments
    finally:
        _reset_global_state()


def test_after_build_consumes_prepared_cache_without_calling_segment_text() -> None:
    """命中早期路径预分段缓存时，发送链上一次 LLM 都不应调用。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 8,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "",
        "delay_base": 0.35,
        "delay_per_char": 0.015,
        "delay_max": 1.2,
    }
    response_text = "缓存命中走零 LLM 路径，把整段拆成两条发出去。"
    segment_text_mock = AsyncMock(return_value=["不应被调用"])

    try:
        _store_prepared_segments("stream-cache-hit", response_text, ["缓存命中走零 LLM 路径", "把整段拆成两条发出去"])

        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            result = asyncio.run(
                plugin.handle_smart_segmentation_after_build(
                    message={
                        "message_id": "cache-msg-1",
                        "session_id": "stream-cache-hit",
                        "timestamp": "6000.0",
                        "raw_message": [{"type": "text", "data": response_text}],
                        "message_info": {"additional_config": {}},
                    },
                    stream_id="stream-cache-hit",
                    processed_plain_text=response_text,
                )
            )

        segment_text_mock.assert_not_awaited()
        assert result["action"] == "continue"
        assert result["modified_kwargs"]["processed_plain_text"] == "缓存命中走零 LLM 路径"
        # 缓存命中后立即移除，避免下一条主回复误用旧缓存
        assert not plugin_module._prepared_segment_registry
        assert plugin_module._pending_follow_up_segments
    finally:
        _reset_global_state()


def test_maisaka_after_response_stores_prepared_cache() -> None:
    """maisaka.replyer.after_response 阶段应该把分段结果登记到缓存里。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 8,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "",
        "delay_base": 0.35,
        "delay_per_char": 0.015,
        "delay_max": 1.2,
    }
    response_text = "早期路径把这段足够长的回复预先切分好缓存起来。"
    segment_text_mock = AsyncMock(return_value=["早期路径把这段足够长的回复", "预先切分好缓存起来"])

    try:
        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            result = asyncio.run(
                plugin.handle_maisaka_replyer_after_response(
                    response=response_text,
                    session_id="stream-prepared",
                )
            )

        assert result == {"action": "continue"}
        segment_text_mock.assert_awaited_once()
        cached = _pop_prepared_segments("stream-prepared", response_text)
        assert cached == ["早期路径把这段足够长的回复", "预先切分好缓存起来"]
    finally:
        _reset_global_state()


def test_maisaka_after_response_skips_during_active_command() -> None:
    """命令期间不应做早期预分段，避免占用 LLM 配额做没意义的工作。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 8,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "",
        "delay_base": 0.35,
        "delay_per_char": 0.015,
        "delay_max": 1.2,
    }
    segment_text_mock = AsyncMock(return_value=["不应被调用"])

    try:
        _mark_command_stream_active("stream-cmd")

        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            asyncio.run(
                plugin.handle_maisaka_replyer_after_response(
                    response="命令期间的某段比较长的回复文本。",
                    session_id="stream-cmd",
                )
            )

        segment_text_mock.assert_not_awaited()
        assert not plugin_module._prepared_segment_registry
    finally:
        _reset_global_state()


def test_maisaka_after_response_skips_when_text_too_short() -> None:
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 50,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "",
        "delay_base": 0.35,
        "delay_per_char": 0.015,
        "delay_max": 1.2,
    }
    segment_text_mock = AsyncMock(return_value=["不应被调用"])

    try:
        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            asyncio.run(
                plugin.handle_maisaka_replyer_after_response(
                    response="太短了。",
                    session_id="stream-short",
                )
            )

        segment_text_mock.assert_not_awaited()
        assert not plugin_module._prepared_segment_registry
    finally:
        _reset_global_state()


def test_command_reply_grace_window_is_short_enough_to_avoid_masking_normal_reply() -> None:
    # 90s 的旧窗口会把命令结束后紧跟的主回复一起挡住，这里用一个不超过 10s 的上限来锁住回归。
    assert _COMMAND_REPLY_GRACE_SECONDS <= 10.0
    # 老版本的 2.0s 太宽，调到 1.0s 已经够覆盖 IPC 微抖动；再往上反弹时应该重新评估。
    assert _COMMAND_REPLY_GRACE_SECONDS <= 1.5


def test_command_after_execute_does_not_extend_grace_window() -> None:
    _reset_global_state()
    stream_id = "stream-normal"

    try:
        with patch.object(plugin_module.time, "monotonic", return_value=1000.0):
            _mark_command_stream_active(stream_id)
        active_expiry = plugin_module._recent_command_stream_expiries[stream_id]

        # 命令结束时不允许续期，否则会把结束后的业务主回复一起挡住。
        with patch.object(plugin_module.time, "monotonic", return_value=1000.5):
            _mark_command_stream_inactive(stream_id)

        assert not _is_command_stream_active(stream_id)
        assert plugin_module._recent_command_stream_expiries[stream_id] == active_expiry
    finally:
        _reset_global_state()


def test_grace_window_remaining_expires_without_renewal() -> None:
    _reset_global_state()
    stream_id = "stream-expire"

    try:
        with patch.object(plugin_module.time, "monotonic", return_value=0.0):
            _mark_command_stream_active(stream_id)
            _mark_command_stream_inactive(stream_id)

        # 窗口内能取到剩余时间。
        within_window = _COMMAND_REPLY_GRACE_SECONDS / 2
        with patch.object(plugin_module.time, "monotonic", return_value=within_window):
            remaining = _get_command_stream_grace_remaining(stream_id)
        assert remaining is not None
        assert 0 < remaining <= _COMMAND_REPLY_GRACE_SECONDS

        # 超过窗口立即返回 None，并清理掉过期键，避免脏状态残留。
        with patch.object(plugin_module.time, "monotonic", return_value=_COMMAND_REPLY_GRACE_SECONDS + 1.0):
            assert _get_command_stream_grace_remaining(stream_id) is None
        assert stream_id not in plugin_module._recent_command_stream_expiries
    finally:
        _reset_global_state()
