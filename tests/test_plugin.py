import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.smart_segmentation_plugin import plugin as plugin_module
from plugins.smart_segmentation_plugin.plugin import (
    _COMMAND_REPLY_GRACE_SECONDS,
    _PREPARED_SEGMENTS_ADDITIONAL_CONFIG_KEY,
    SmartSegmentationPlugin,
    _extract_prepared_segments_from_outbound_message,
    _get_command_stream_grace_remaining,
    _inject_prepared_segments_into_message,
    _is_command_stream_active,
    _mark_command_stream_active,
    _mark_command_stream_inactive,
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
