import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.smart_segmentation_plugin import plugin as plugin_module
from plugins.smart_segmentation_plugin.plugin import (
    _COMMAND_REPLY_GRACE_SECONDS,
    _PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY,
    SmartSegmentationPlugin,
    _extract_plain_text_outbound_message,
    _extract_prepared_segments_from_outbound_message,
    _get_command_stream_grace_remaining,
    _inject_prepared_segments_into_message,
    _is_command_stream_active,
    _mark_command_stream_active,
    _mark_command_stream_inactive,
    _replace_outbound_text,
)


def test_get_components_registers_hook_handlers_for_current_host() -> None:
    plugin = SmartSegmentationPlugin()

    components = plugin.get_components()
    hook_component_type = "HOOK_HANDLER" if plugin_module._SDK_HOOK_HANDLER_AVAILABLE else "hook_handler"
    hook_components = {component["name"]: component for component in components if component.get("type") == hook_component_type}

    assert "smart_segmentation_after_build" in hook_components
    assert hook_components["smart_segmentation_after_build"]["metadata"]["hook"] == "send_service.after_build_message"
    assert "smart_segmentation_after_send" in hook_components
    assert hook_components["smart_segmentation_after_send"]["metadata"]["hook"] == "send_service.after_send"
    assert "smart_segmentation_command_scope_enter" in hook_components
    assert hook_components["smart_segmentation_command_scope_enter"]["metadata"]["hook"] == "chat.command.before_execute"
    assert "smart_segmentation_command_scope_leave" in hook_components
    assert hook_components["smart_segmentation_command_scope_leave"]["metadata"]["hook"] == "chat.command.after_execute"


def test_inject_and_extract_prepared_segments_use_same_storage_channel() -> None:
    message = {
        "message_info": {
            "additional_config": {},
        }
    }

    _inject_prepared_segments_into_message(message, ["第一段", "第二段"])

    assert message["message_info"]["additional_config"][_PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY] == "第一段|||SPLIT|||第二段"

    outbound_message = {
        "message_info": {
            "additional_config": dict(message["message_info"]["additional_config"]),
        }
    }

    segments, source = _extract_prepared_segments_from_outbound_message(outbound_message)

    assert segments == ["第一段", "第二段"]
    assert source == "additional_config"


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
                _PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY: "旧标记",
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
    assert _PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY not in updated_message["message_info"]["additional_config"]
    assert updated_message["message_info"]["additional_config"]["platform_io_target_group_id"] == "20001"


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
        plugin_module._pending_follow_up_segments.clear()
        plugin_module._stream_resend_guards.clear()


def test_after_build_skips_generic_plain_text_without_reply_context() -> None:
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
    segment_text_mock = AsyncMock(return_value=["第一段", "第二段"])

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
                display_message="INFO 2026-05-07 20:26:24 plugin.smart_segmentation 智能分段已消费预分段标记",
            )
        )

    segment_text_mock.assert_not_awaited()
    assert result == {"action": "continue"}


def test_after_build_keeps_fallback_for_explicit_reply_context() -> None:
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
    segment_text_mock = AsyncMock(return_value=["第一段", "第二段"])

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
                                "data": "这是一段足够长的 bot 回复文本，需要在发送前兜底分段，避免整条一起发出去。",
                            }
                        ],
                        "message_info": {"additional_config": {}},
                    },
                    stream_id="stream-reply",
                    display_message="这是一段足够长的 bot 回复文本，需要在发送前兜底分段，避免整条一起发出去。",
                )
            )

        segment_text_mock.assert_awaited_once_with(
            "这是一段足够长的 bot 回复文本，需要在发送前兜底分段，避免整条一起发出去。",
            style="natural",
            model_name="",
            max_segments=8,
            temperature=0.3,
            max_tokens=600,
        )
        assert result["action"] == "continue"
        assert result["modified_kwargs"]["display_message"] == "第一段"
        assert result["modified_kwargs"]["message"]["display_message"] == "第一段"
        assert plugin_module._pending_follow_up_segments
    finally:
        plugin_module._pending_follow_up_segments.clear()
        plugin_module._stream_resend_guards.clear()


def test_after_build_allows_plain_bot_reply_without_reply_context() -> None:
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
    segment_text_mock = AsyncMock(return_value=["第一段", "第二段"])

    try:
        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            result = asyncio.run(
                plugin.handle_smart_segmentation_after_build(
                    message={
                        "message_id": "plain-reply-msg-1",
                        "session_id": "stream-plain-reply",
                        "timestamp": "4000.0",
                        "raw_message": [
                            {
                                "type": "text",
                                "data": "好耶今天真的还挺开心的，晚点我再慢慢跟你讲。",
                            }
                        ],
                        "message_info": {"additional_config": {}},
                    },
                    stream_id="stream-plain-reply",
                    display_message="好耶今天真的还挺开心的，晚点我再慢慢跟你讲。",
                )
            )

        segment_text_mock.assert_awaited_once_with(
            "好耶今天真的还挺开心的，晚点我再慢慢跟你讲。",
            style="natural",
            model_name="",
            max_segments=8,
            temperature=0.3,
            max_tokens=600,
        )
        assert result["action"] == "continue"
        assert result["modified_kwargs"]["display_message"] == "第一段"
        assert result["modified_kwargs"]["message"]["display_message"] == "第一段"
        assert plugin_module._pending_follow_up_segments
    finally:
        plugin_module._pending_follow_up_segments.clear()
        plugin_module._stream_resend_guards.clear()


def _reset_command_stream_state() -> None:
    plugin_module._active_command_streams.clear()
    plugin_module._recent_command_stream_expiries.clear()


def test_command_reply_grace_window_is_short_enough_to_avoid_masking_normal_reply() -> None:
    # 90s 的旧窗口会把命令结束后紧跟的主回复一起挡住，这里用一个不超过 10s 的上限来锁住回归。
    assert _COMMAND_REPLY_GRACE_SECONDS <= 10.0


def test_command_after_execute_does_not_extend_grace_window() -> None:
    _reset_command_stream_state()
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
        _reset_command_stream_state()


def test_grace_window_remaining_expires_without_renewal() -> None:
    _reset_command_stream_state()
    stream_id = "stream-expire"

    try:
        with patch.object(plugin_module.time, "monotonic", return_value=0.0):
            _mark_command_stream_active(stream_id)
            _mark_command_stream_inactive(stream_id)

        # 窗口内能取到剩余时间。
        with patch.object(plugin_module.time, "monotonic", return_value=0.5):
            remaining = _get_command_stream_grace_remaining(stream_id)
        assert remaining is not None
        assert 0 < remaining <= _COMMAND_REPLY_GRACE_SECONDS

        # 超过窗口立即返回 None，并清理掉过期键，避免脏状态残留。
        with patch.object(plugin_module.time, "monotonic", return_value=_COMMAND_REPLY_GRACE_SECONDS + 1.0):
            assert _get_command_stream_grace_remaining(stream_id) is None
        assert stream_id not in plugin_module._recent_command_stream_expiries
    finally:
        _reset_command_stream_state()
