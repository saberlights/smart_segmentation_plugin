import asyncio
import json
import random
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.smart_segmentation_plugin import plugin as plugin_module
from plugins.smart_segmentation_plugin.plugin import (
    _COMMAND_REPLY_GRACE_SECONDS,
    PluginSectionConfig,
    SegmentationSectionConfig,
    SmartSegmentationPlugin,
    _extract_json_array_text,
    _extract_plain_text_outbound_message,
    _get_command_stream_grace_remaining,
    _has_unbalanced_brackets,
    _hash_normalized_text,
    _is_action_only_text,
    _is_command_stream_active,
    _mark_command_stream_active,
    _mark_command_stream_inactive,
    _merge_segments_balancing_brackets,
    _pop_prepared_segments,
    _replace_outbound_text,
    _segments_preserve_original_content,
    _split_segments_at_bracket_boundaries,
    _split_text_at_brackets,
    _store_prepared_segments,
    _strip_thinking_content,
)


def _reset_global_state() -> None:
    plugin_module._prepared_segment_registry.clear()
    plugin_module._pending_follow_up_segments.clear()
    plugin_module._stream_resend_guards.clear()
    plugin_module._active_command_streams.clear()
    plugin_module._active_command_stream_expiries.clear()
    plugin_module._recent_command_stream_expiries.clear()
    plugin_module._active_follow_up_tasks.clear()
    plugin_module._active_follow_up_tasks_by_stream.clear()
    plugin_module._follow_up_idle_events_by_stream.clear()
    plugin_module._planner_follow_up_entries_by_stream.clear()


def test_config_schema_uses_user_facing_webui_copy() -> None:
    assert PluginSectionConfig.__ui_label__ == "基础设置"
    assert SegmentationSectionConfig.__ui_label__ == "智能分段"

    plugin_fields = PluginSectionConfig.model_fields
    segmentation_fields = SegmentationSectionConfig.model_fields
    assert plugin_fields["config_version"].default == "1.2.0"
    assert "保持默认值" in plugin_fields["name"].description
    assert "true 开启，false 关闭" in plugin_fields["enabled"].description
    assert "留空使用默认模型" in segmentation_fields["model"].description
    assert "填写正整数" in segmentation_fields["min_length"].description
    assert "填写正整数" in segmentation_fields["max_segments"].description
    assert "true 开启，false 关闭" in segmentation_fields["typing_enabled"].description


def test_webui_hints_match_config_toml_comments() -> None:
    config_path = Path(plugin_module.__file__).with_name("config.toml")
    comments: dict[tuple[str, str], str] = {}
    section = ""
    pending_comment = ""

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            pending_comment = line[2:]
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            pending_comment = ""
            continue
        if section and pending_comment and "=" in line:
            field_name = line.split("=", 1)[0].strip()
            comments[(section, field_name)] = pending_comment
        if line:
            pending_comment = ""

    schema = SmartSegmentationPlugin().get_webui_config_schema(
        plugin_id="saberlights.smart-segmentation-plugin"
    )
    schema_fields = {
        (section_name, field_name): field
        for section_name, section_schema in schema["sections"].items()
        for field_name, field in section_schema["fields"].items()
    }

    assert set(schema_fields) == set(comments)
    for field_path, comment in comments.items():
        assert schema_fields[field_path]["hint"] == comment


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
    assert "smart_segmentation_wait_before_planner" in hook_components
    assert (
        hook_components["smart_segmentation_wait_before_planner"]["metadata"]["hook"]
        == "maisaka.planner.before_request"
    )
    assert "smart_segmentation_pause_until_follow_ups_finish" in hook_components
    assert (
        hook_components["smart_segmentation_pause_until_follow_ups_finish"]["metadata"]["hook"]
        == "chat.receive.before_process"
    )
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
    replyer_hook_timeout_ms = hook_components["smart_segmentation_after_replyer_response"]["metadata"]["timeout_ms"]
    assert replyer_hook_timeout_ms == 25_000
    assert replyer_hook_timeout_ms > plugin_module._REPLYER_PREPARED_SEGMENT_TIMEOUT_SECONDS * 1000
    assert "smart_segmentation_preserve_prepared_response" in hook_components
    assert (
        hook_components["smart_segmentation_preserve_prepared_response"]["metadata"]["hook"]
        == "maisaka.reply.before_post_process"
    )


def test_runtime_settings_use_one_config_snapshot_and_respect_configured_segment_count() -> None:
    plugin = SmartSegmentationPlugin()
    mock_ctx = MagicMock()
    mock_ctx.config.get_all = AsyncMock(
        return_value={
            "plugin": {"enabled": True},
            "segmentation": {
                "enabled": True,
                "model": "utils",
                "style": "active",
                "min_length": "3",
                "max_segments": 99,
                "temperature": "0.4",
                "max_tokens": "700",
                "typing_enabled": False,
                "delay_base": 99,
            },
        }
    )
    plugin._ctx = mock_ctx

    with patch.object(plugin, "_load_local_config_fallback", return_value={}):
        settings = asyncio.run(plugin._get_segmentation_runtime_settings())

    assert settings == {
        "min_length": 3,
        "max_segments": 99,
        "temperature": 0.4,
        "max_tokens": 700,
        "style": "active",
        "model_name": "utils",
        "typing_enabled": False,
    }
    assert mock_ctx.config.get_all.await_count == 1


def test_runtime_settings_enable_typing_by_default() -> None:
    plugin = SmartSegmentationPlugin()
    mock_ctx = MagicMock()
    mock_ctx.config.get_all = AsyncMock(
        return_value={
            "plugin": {"enabled": True},
            "segmentation": {"enabled": True},
        }
    )
    plugin._ctx = mock_ctx

    with patch.object(plugin, "_load_local_config_fallback", return_value={}):
        settings = asyncio.run(plugin._get_segmentation_runtime_settings())

    assert settings is not None
    assert settings["typing_enabled"] is True


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


def test_before_post_process_preserves_response_with_prepared_segments() -> None:
    _reset_global_state()
    plugin = SmartSegmentationPlugin()
    response_text = "第一段第二段"

    try:
        _store_prepared_segments("stream-post-process", response_text, ["第一段", "第二段"])

        result = asyncio.run(
            plugin.handle_maisaka_reply_before_post_process(
                response=response_text,
                session_id="stream-post-process",
                reply_message_id="source-message",
                reply_tool_args={"set_quote": False},
                skip_post_process=False,
                enable_splitter=True,
                enable_chinese_typo=False,
            )
        )

        assert result == {
            "action": "continue",
            "modified_kwargs": {
                "response": response_text,
                "session_id": "stream-post-process",
                "reply_message_id": "source-message",
                "reply_tool_args": {"set_quote": False},
                "skip_post_process": True,
                "enable_splitter": True,
                "enable_chinese_typo": False,
            },
        }
        assert _pop_prepared_segments("stream-post-process", response_text) == ["第一段", "第二段"]
    finally:
        _reset_global_state()


def test_replyer_segmentation_preserves_overflow_text_within_segment_limit() -> None:
    """模型忽略段数上限时，插件仍必须保留完整回复且不超过配置段数。"""
    _reset_global_state()
    plugin = SmartSegmentationPlugin()
    response_text = "第一段第二段第三段"
    mock_ctx = MagicMock()
    mock_ctx.config.get_all = AsyncMock(
        return_value={
            "plugin": {"enabled": True},
            "segmentation": {
                "enabled": True,
                "model": "",
                "min_length": 1,
                "max_segments": 2,
            },
        }
    )
    mock_ctx.llm.generate = AsyncMock(
        return_value={
            "success": True,
            "response": '["第一段", "第二段", "第三段"]',
            "model_name": "test-model",
        }
    )
    plugin._ctx = mock_ctx

    try:
        asyncio.run(
            plugin.handle_maisaka_replyer_after_response(
                response=response_text,
                session_id="stream-overflow",
            )
        )

        assert _pop_prepared_segments("stream-overflow", response_text) == ["第一段", "第二段第三段"]
    finally:
        _reset_global_state()


def test_replyer_segmentation_rejects_model_rewritten_text() -> None:
    """字词被模型改写的分段结果必须被丢弃，回退为原文直发。"""
    _reset_global_state()
    plugin = SmartSegmentationPlugin()
    response_text = "原文绝不能被模型修改"
    mock_ctx = MagicMock()
    mock_ctx.config.get_all = AsyncMock(
        return_value={
            "plugin": {"enabled": True},
            "segmentation": {
                "enabled": True,
                "model": "",
                "min_length": 1,
                "max_segments": 8,
            },
        }
    )
    mock_ctx.llm.generate = AsyncMock(
        return_value={
            "success": True,
            "response": '["原文已经", "被模型修改"]',
            "model_name": "test-model",
        }
    )
    plugin._ctx = mock_ctx

    try:
        asyncio.run(
            plugin.handle_maisaka_replyer_after_response(
                response=response_text,
                session_id="stream-rewritten",
            )
        )

        assert _pop_prepared_segments("stream-rewritten", response_text) is None
    finally:
        _reset_global_state()


def test_replyer_segmentation_accepts_model_punctuation_changes() -> None:
    """标点变化不应导致整个分段结果被丢弃。"""
    _reset_global_state()
    plugin = SmartSegmentationPlugin()
    response_text = "原文没有句号"
    mock_ctx = MagicMock()
    mock_ctx.config.get_all = AsyncMock(
        return_value={
            "plugin": {"enabled": True},
            "segmentation": {
                "enabled": True,
                "model": "",
                "min_length": 1,
                "max_segments": 8,
            },
        }
    )
    mock_ctx.llm.generate = AsyncMock(
        return_value={
            "success": True,
            "response": '["原文。", "没有句号"]',
            "model_name": "test-model",
        }
    )
    plugin._ctx = mock_ctx

    try:
        asyncio.run(
            plugin.handle_maisaka_replyer_after_response(
                response=response_text,
                session_id="stream-inserted-period",
            )
        )

        assert _pop_prepared_segments("stream-inserted-period", response_text) == ["原文。", "没有句号"]
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
            "typing_enabled": True,
        },
    )

    send_segments_mock = AsyncMock(return_value=True)

    async def _invoke_and_wait() -> None:
        # hook 体本身已经被改成 create_task + return，调用方拿不到补发任务句柄；
        # 用 _drain_active_follow_up_tasks 等到后台真正发完，再去 assert mock 调用。
        await plugin.handle_smart_segmentation_after_send(
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
        await plugin_module._drain_active_follow_up_tasks()

    try:
        with patch.object(plugin, "_send_segments", send_segments_mock):
            asyncio.run(_invoke_and_wait())

        send_segments_mock.assert_awaited_once_with(
            "stream-reply",
            ["第二段", "第三段"],
            typing_enabled=True,
        )
        assert not plugin_module._pending_follow_up_segments
    finally:
        _reset_global_state()


def test_after_send_returns_immediately_when_follow_up_is_slower_than_host_timeout() -> None:
    """宿主对 send_service.after_send 强制 5000ms 超时（OBSERVE 也会被 cancel），
    所以补发循环必须从 hook 体里剥离到 asyncio.create_task；
    一旦 hook 自己 await 串行补发，超长消息就会丢掉后半段（实战中表现为多分段
    回复"……一条黑色针织连衣裙……" 后面 3 段没发出去）。本用例模拟一个比宿主
    超时还慢的补发协程，验证：
    1) hook 本身在远低于宿主 5000ms 阈值内返回；
    2) 剩余分段最终仍然在后台跑完。"""
    plugin = SmartSegmentationPlugin()
    tracking_key = plugin_module._build_follow_up_tracking_key(
        stream_id="stream-late",
        timestamp="1234.0",
        visible_text="首段",
    )
    plugin_module._register_pending_follow_up_segments(
        lookup_keys=["msg-late", tracking_key],
        pending_data={
            "stream_id": "stream-late",
            "segments": ["段二", "段三", "段四"],
            "typing_enabled": True,
        },
    )

    async def _slow_send_segments(*_args, **_kwargs):
        # 在测试事件循环里也明显比 hook 调用本身慢；如果 hook 体真的 await 它，
        # 后面的 wait_for(timeout=0.1) 一定会抛 TimeoutError。
        await asyncio.sleep(0.4)
        return True

    send_segments_mock = AsyncMock(side_effect=_slow_send_segments)

    async def _run() -> None:
        with patch.object(plugin, "_send_segments", send_segments_mock):
            await asyncio.wait_for(
                plugin.handle_smart_segmentation_after_send(
                    message={
                        "message_id": "msg-late",
                        "session_id": "stream-late",
                        "timestamp": "1234.0",
                        "processed_plain_text": "首段",
                        "raw_message": [{"type": "text", "data": "首段"}],
                    },
                    sent=True,
                ),
                timeout=0.1,
            )
            # hook 已经立即返回，但后台补发还在跑；等它跑完再 assert。
            await plugin_module._drain_active_follow_up_tasks()

    try:
        asyncio.run(_run())
        send_segments_mock.assert_awaited_once_with(
            "stream-late",
            ["段二", "段三", "段四"],
            typing_enabled=True,
        )
        assert not plugin_module._pending_follow_up_segments
        assert "stream-late" not in plugin_module._follow_up_idle_events_by_stream
    finally:
        _reset_global_state()


def test_on_unload_allows_in_flight_follow_up_to_finish_within_grace_period() -> None:
    """正常卸载应先短暂等待已接收的尾段，而不是立刻取消并丢消息。"""
    plugin = SmartSegmentationPlugin()

    async def _run() -> tuple[list[str], bool]:
        completed_segments: list[str] = []

        async def _finish_follow_up() -> None:
            await asyncio.sleep(0.01)
            completed_segments.append("尾段")

        task = asyncio.create_task(_finish_follow_up())
        plugin_module._track_follow_up_task(task, stream_id="stream-unload")
        await asyncio.sleep(0)
        await plugin.on_unload()
        return completed_segments, task.cancelled()

    _reset_global_state()
    try:
        completed_segments, task_cancelled = asyncio.run(_run())

        assert completed_segments == ["尾段"]
        assert task_cancelled is False
        assert not plugin_module._active_follow_up_tasks
        assert not plugin_module._active_follow_up_tasks_by_stream
        assert not plugin_module._follow_up_idle_events_by_stream
    finally:
        _reset_global_state()


def test_chat_receive_before_process_waits_until_same_stream_follow_ups_finish() -> None:
    """入站消息处理前的轻量阻塞 hook 必须等到同 stream 的补发真正结束。

    用户在机器人仍模拟打字时追发新消息，不能让新一轮入站处理与旧回复交错。
    这里选择只携带轻量消息体的 chat.receive.before_process，避免传输 planner 大提示词。
    """
    _reset_global_state()
    plugin = SmartSegmentationPlugin()

    tracking_key = plugin_module._build_follow_up_tracking_key(
        stream_id="stream-planner-guard",
        timestamp="5678.0",
        visible_text="首段",
    )
    plugin_module._register_pending_follow_up_segments(
        lookup_keys=["msg-planner-guard", tracking_key],
        pending_data={
            "stream_id": "stream-planner-guard",
            "segments": ["段二", "段三"],
            "typing_enabled": True,
        },
    )

    release_send = asyncio.Event()
    send_started = asyncio.Event()

    async def _blocked_send_segments(*_args, **_kwargs):
        send_started.set()
        await release_send.wait()
        return True

    send_segments_mock = AsyncMock(side_effect=_blocked_send_segments)

    async def _run() -> None:
        with patch.object(plugin, "_send_segments", send_segments_mock):
            await plugin.handle_smart_segmentation_after_send(
                message={
                    "message_id": "msg-planner-guard",
                    "session_id": "stream-planner-guard",
                    "timestamp": "5678.0",
                    "processed_plain_text": "首段",
                    "raw_message": [{"type": "text", "data": "首段"}],
                },
                sent=True,
            )
            await asyncio.wait_for(send_started.wait(), timeout=0.1)

            planner_gate_task = asyncio.create_task(
                plugin.handle_chat_receive_before_process(
                    message={"session_id": "stream-planner-guard"},
                )
            )

            blocked = False
            try:
                await asyncio.wait_for(asyncio.shield(planner_gate_task), timeout=0.05)
            except asyncio.TimeoutError:
                blocked = True
            assert blocked is True

            release_send.set()
            result = await asyncio.wait_for(planner_gate_task, timeout=0.1)
            assert result["action"] == "continue"
            # 轻量阻塞 hook 只等待、不再回填 prompt，因此不得返回 modified_kwargs
            assert "modified_kwargs" not in result
            await plugin_module._drain_active_follow_up_tasks()

    try:
        asyncio.run(_run())
        send_segments_mock.assert_awaited_once()
    finally:
        _reset_global_state()


def test_planner_before_request_waits_for_follow_ups_and_repairs_built_prompt() -> None:
    """内部 planner 可能先构建 prompt、后进入 Hook；闸门必须等待补发完成，
    并把已经构建好的首段消息修补为本次完整回复。"""
    _reset_global_state()
    plugin = SmartSegmentationPlugin()
    response_text = "行啊汇报完了，是不是该给点奖励我明天没档期"
    segments = ["行啊", "汇报完了，是不是该给点奖励", "我明天没档期"]
    release_send = asyncio.Event()
    send_started = asyncio.Event()

    async def _blocked_send_segments(*_args, **_kwargs):
        send_started.set()
        await release_send.wait()
        return True

    async def _run() -> None:
        _store_prepared_segments("stream-planner", response_text, segments)
        with (
            patch.object(
                plugin,
                "_get_segmentation_runtime_settings",
                AsyncMock(
                    return_value={
                        "min_length": 1,
                        "max_segments": 8,
                        "temperature": 0.3,
                        "max_tokens": 600,
                        "style": "natural",
                        "model_name": "",
                        "typing_enabled": True,
                    }
                ),
            ),
            patch.object(plugin, "_send_segments", AsyncMock(side_effect=_blocked_send_segments)),
        ):
            build_result = await plugin.handle_smart_segmentation_after_build(
                message={
                    "message_id": "reply-msg",
                    "session_id": "stream-planner",
                    "timestamp": "1000.0",
                    "raw_message": [{"type": "text", "data": response_text}],
                    "message_info": {"additional_config": {}},
                },
                stream_id="stream-planner",
                processed_plain_text=response_text,
            )
            first_message = build_result["modified_kwargs"]["message"]
            planner_messages = [
                {"role": "tool", "content": "工具结果声称三段均已发送"},
                {
                    "role": "user",
                    "content": '<message msg_id="first" user="bot" is_self_message="true">\n行啊',
                },
                {"role": "user", "content": "时间：2026-07-26 23:45:52"},
            ]

            planner_task = asyncio.create_task(
                plugin.handle_maisaka_planner_before_request(
                    messages=planner_messages,
                    tool_definitions=[{"type": "function"}],
                    selected_history_count=2,
                    built_message_count=3,
                    selection_reason="test",
                    session_id="stream-planner",
                )
            )
            await asyncio.sleep(0)
            assert planner_task.done() is False

            await plugin.handle_smart_segmentation_after_send(message=first_message, sent=True)
            await asyncio.wait_for(send_started.wait(), timeout=0.1)
            assert planner_task.done() is False

            release_send.set()
            planner_result = await asyncio.wait_for(planner_task, timeout=0.2)

        modified = planner_result["modified_kwargs"]
        assert modified["messages"][1]["content"].endswith("\n" + "\n".join(segments))
        assert modified["tool_definitions"] == [{"type": "function"}]
        assert modified["selected_history_count"] == 2
        assert modified["built_message_count"] == 3
        assert modified["selection_reason"] == "test"
        assert modified["session_id"] == "stream-planner"

    try:
        asyncio.run(_run())
    finally:
        _reset_global_state()


def test_planner_releases_quickly_when_first_send_never_reaches_after_send() -> None:
    """首段在宿主 before_send/路由/Platform IO 阶段失败时不会触发 after_send。

    Planner 只能为异步 OBSERVE Hook 留一个很短的到达窗口，不能误等完整补发超时。
    """
    _reset_global_state()
    plugin = SmartSegmentationPlugin()
    response_text = "第一段第二段第三段"
    segments = ["第一段", "第二段", "第三段"]

    async def _run() -> None:
        _store_prepared_segments("stream-first-send-failed", response_text, segments)
        with patch.object(
            plugin,
            "_get_segmentation_runtime_settings",
            AsyncMock(
                return_value={
                    "min_length": 1,
                    "max_segments": 8,
                    "temperature": 0.3,
                    "max_tokens": 600,
                    "style": "natural",
                    "model_name": "",
                    "typing_enabled": True,
                }
            ),
        ):
            await plugin.handle_smart_segmentation_after_build(
                message={
                    "message_id": "first-send-failed",
                    "session_id": "stream-first-send-failed",
                    "timestamp": "1000.0",
                    "raw_message": [{"type": "text", "data": response_text}],
                    "message_info": {"additional_config": {}},
                },
                stream_id="stream-first-send-failed",
                processed_plain_text=response_text,
            )

        with patch.object(
            plugin_module,
            "_FIRST_SEND_OBSERVE_GRACE_SECONDS",
            0.02,
            create=True,
        ):
            result = await asyncio.wait_for(
                plugin.handle_maisaka_planner_before_request(
                    messages=[],
                    session_id="stream-first-send-failed",
                ),
                timeout=0.1,
            )

        assert result["action"] == "continue"
        assert not plugin_module._pending_follow_up_segments

    try:
        asyncio.run(_run())
    finally:
        _reset_global_state()


def test_register_planner_entry_prunes_expired_entries_from_other_streams() -> None:
    """新回复到来时也必须回收其他流的过期 Planner 状态。"""
    _reset_global_state()
    try:
        expired_entry = plugin_module._register_planner_follow_up_entry(
            stream_id="stream-expired",
            segments=["首段", "尾段"],
        )
        expired_entry["expires_at"] = 0.0

        current_entry = plugin_module._register_planner_follow_up_entry(
            stream_id="stream-current",
            segments=["新首段", "新尾段"],
        )

        assert "stream-expired" not in plugin_module._planner_follow_up_entries_by_stream
        assert plugin_module._planner_follow_up_entries_by_stream == {
            "stream-current": [current_entry],
        }
    finally:
        _reset_global_state()


def test_send_segments_marks_follow_ups_for_maisaka_history_sync() -> None:
    """补发的分段必须显式声明 sync_to_maisaka_history，否则 maisaka 历史
    只会留下首段，下一轮规划器会把剩余内容彻底丢掉（实战中表现为：分段后的
    回复"等下让我想想，感觉后面那个是聞かせて？……" 只剩下"等下让我想想"
    被记入历史）。"""
    plugin = SmartSegmentationPlugin()
    send_text_mock = AsyncMock(return_value=True)
    mock_ctx = MagicMock()
    mock_ctx.send = MagicMock()
    mock_ctx.send.text = send_text_mock
    # ctx 是只读 property，运行时由 Runner 注入；测试里直接打到下层 _ctx
    plugin._ctx = mock_ctx

    ok = asyncio.run(
        plugin._send_segments(
            "stream-history",
            ["第二段", "第三段"],
            typing_enabled=True,
        )
    )

    assert ok is True
    assert send_text_mock.await_count == 2
    assert [call.args[0] for call in send_text_mock.await_args_list] == ["第二段", "第三段"]
    for call in send_text_mock.await_args_list:
        # ctx.send.text(segment, stream_id, sync_to_maisaka_history=True, maisaka_source_kind="guided_reply")
        assert call.kwargs.get("typing") is True
        assert call.kwargs.get("sync_to_maisaka_history") is True
        assert call.kwargs.get("maisaka_source_kind") == "guided_reply"
        assert call.kwargs.get("timeout_ms") == plugin_module._SEND_SEGMENT_RPC_TIMEOUT_MS


def test_send_segments_retries_failed_segment_and_continues_with_later_text() -> None:
    plugin = SmartSegmentationPlugin()
    send_text_mock = AsyncMock(side_effect=[True, False, False, True])
    mock_ctx = MagicMock()
    mock_ctx.send = MagicMock()
    mock_ctx.send.text = send_text_mock
    plugin._ctx = mock_ctx

    ok = asyncio.run(
        plugin._send_segments(
            "stream-failure",
            ["第二段", "第三段", "第四段"],
            typing_enabled=True,
        )
    )

    assert ok is False
    assert [call.args[0] for call in send_text_mock.await_args_list] == [
        "第二段",
        "第三段",
        "第三段",
        "第四段",
    ]
    assert send_text_mock.await_args_list[2].kwargs["typing"] is False


def test_send_segments_retries_rpc_exception_without_losing_later_text() -> None:
    plugin = SmartSegmentationPlugin()
    send_text_mock = AsyncMock(side_effect=[RuntimeError("rpc disconnected"), True, True])
    mock_ctx = MagicMock()
    mock_ctx.send = MagicMock()
    mock_ctx.send.text = send_text_mock
    plugin._ctx = mock_ctx

    ok = asyncio.run(
        plugin._send_segments(
            "stream-rpc-error",
            ["第二段", "第三段"],
            typing_enabled=True,
        )
    )

    assert ok is True
    assert [call.args[0] for call in send_text_mock.await_args_list] == ["第二段", "第二段", "第三段"]
    assert send_text_mock.await_args_list[1].kwargs["typing"] is False


def test_send_segments_disables_typing_for_all_attempts_when_configured_off() -> None:
    plugin = SmartSegmentationPlugin()
    send_text_mock = AsyncMock(side_effect=[False, True, True])
    mock_ctx = MagicMock()
    mock_ctx.send = MagicMock()
    mock_ctx.send.text = send_text_mock
    plugin._ctx = mock_ctx

    ok = asyncio.run(
        plugin._send_segments(
            "stream-no-typing",
            ["第二段", "第三段"],
            typing_enabled=False,
        )
    )

    assert ok is True
    assert [call.kwargs["typing"] for call in send_text_mock.await_args_list] == [False, False, False]


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
        "typing_enabled": True,
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
        "typing_enabled": True,
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
        "typing_enabled": True,
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
        "typing_enabled": True,
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


def test_after_build_propagates_disabled_typing_to_background_send() -> None:
    _reset_global_state()
    plugin = SmartSegmentationPlugin()
    response_text = "首段尾段"
    send_segments_mock = AsyncMock(return_value=True)

    async def _run() -> None:
        _store_prepared_segments("stream-no-typing", response_text, ["首段", "尾段"])
        with (
            patch.object(
                plugin,
                "_get_segmentation_runtime_settings",
                AsyncMock(
                    return_value={
                        "min_length": 1,
                        "max_segments": 8,
                        "temperature": 0.3,
                        "max_tokens": 600,
                        "style": "natural",
                        "model_name": "",
                        "typing_enabled": False,
                    }
                ),
            ),
            patch.object(plugin, "_send_segments", send_segments_mock),
        ):
            build_result = await plugin.handle_smart_segmentation_after_build(
                message={
                    "message_id": "no-typing-message",
                    "session_id": "stream-no-typing",
                    "timestamp": "6200.0",
                    "raw_message": [{"type": "text", "data": response_text}],
                    "message_info": {"additional_config": {}},
                },
                stream_id="stream-no-typing",
                processed_plain_text=response_text,
            )
            await plugin.handle_smart_segmentation_after_send(
                message=build_result["modified_kwargs"]["message"],
                sent=True,
            )
            await plugin_module._drain_active_follow_up_tasks()

    try:
        asyncio.run(_run())
        send_segments_mock.assert_awaited_once_with(
            "stream-no-typing",
            ["尾段"],
            typing_enabled=False,
        )
    finally:
        _reset_global_state()


def test_after_build_matches_reply_body_when_host_prepends_attach_at_component() -> None:
    """attach_at 在 replyer 后处理阶段才加入，不能让已完成的正文预分段缓存失配。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 8,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "",
        "typing_enabled": True,
    }
    response_text = "先确认一下这件事我马上回来告诉你结果"
    at_component = {
        "type": "at",
        "data": {
            "target_user_id": "42",
            "target_user_nickname": "久远",
            "target_user_cardname": "久远",
        },
    }

    try:
        _store_prepared_segments(
            "stream-attach-at",
            response_text,
            ["先确认一下这件事", "我马上回来告诉你结果"],
        )
        with patch.object(
            plugin,
            "_get_segmentation_runtime_settings",
            AsyncMock(return_value=runtime_settings),
        ):
            result = asyncio.run(
                plugin.handle_smart_segmentation_after_build(
                    message={
                        "message_id": "attach-at-msg",
                        "session_id": "stream-attach-at",
                        "timestamp": "6100.0",
                        "raw_message": [
                            at_component,
                            {"type": "text", "data": response_text},
                        ],
                        "message_info": {"additional_config": {}},
                    },
                    stream_id="stream-attach-at",
                    processed_plain_text=f"@久远{response_text}",
                )
            )

        first_message = result["modified_kwargs"]["message"]
        assert first_message["processed_plain_text"] == "@久远先确认一下这件事"
        assert first_message["raw_message"] == [
            at_component,
            {"type": "text", "data": "先确认一下这件事"},
        ]
        assert plugin_module._pending_follow_up_segments
        assert not plugin_module._prepared_segment_registry
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
        "typing_enabled": True,
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


def test_maisaka_after_response_retries_stalled_segmentation_model() -> None:
    """首个分段请求卡住时，第二次同配置模型请求成功后仍应完成预分段。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 1,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "doubao1.6",
    }
    response_text = "你自己打出来的字问我，装什么，这会儿倒开始纯了，不告诉你，自己百度"

    async def _run() -> tuple[dict[str, object], list[str], bool]:
        attempt_count = 0
        attempted_models: list[str] = []
        first_attempt_cancelled = asyncio.Event()

        async def _segment_text(*_args, **_kwargs):
            nonlocal attempt_count
            attempt_count += 1
            attempted_models.append(_kwargs["model_name"])
            if attempt_count == 1:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    first_attempt_cancelled.set()
                    raise
            return ["你自己打出来的字问我", "装什么", "这会儿倒开始纯了", "不告诉你，自己百度"]

        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", AsyncMock(side_effect=_segment_text)),
            patch.object(plugin_module, "_REPLYER_SEGMENT_RETRY_DELAY_SECONDS", 0.01),
            patch.object(plugin_module, "_REPLYER_PREPARED_SEGMENT_TIMEOUT_SECONDS", 0.2),
        ):
            result = await plugin.handle_maisaka_replyer_after_response(
                response=response_text,
                session_id="stream-stalled-segmentation",
            )
        return result, attempted_models, first_attempt_cancelled.is_set()

    _reset_global_state()
    try:
        result, attempted_models, first_attempt_cancelled = asyncio.run(_run())

        assert result == {"action": "continue"}
        assert attempted_models == ["doubao1.6", "doubao1.6"]
        assert first_attempt_cancelled is True
        assert _pop_prepared_segments("stream-stalled-segmentation", response_text) == [
            "你自己打出来的字问我",
            "装什么",
            "这会儿倒开始纯了",
            "不告诉你，自己百度",
        ]
    finally:
        _reset_global_state()


def test_maisaka_after_response_cancels_both_attempts_after_total_timeout() -> None:
    """两次分段请求都卡住时必须取消请求并保持原文回退。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 1,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "doubao1.6",
    }
    response_text = "两个分段模型请求都卡住时，这条完整原文仍然必须安全发出去"

    async def _run() -> tuple[dict[str, object], int, int]:
        attempt_count = 0
        cancelled_count = 0

        async def _segment_text(*_args, **_kwargs):
            nonlocal attempt_count, cancelled_count
            attempt_count += 1
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled_count += 1
                raise

        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", AsyncMock(side_effect=_segment_text)),
            patch.object(plugin_module, "_REPLYER_SEGMENT_RETRY_DELAY_SECONDS", 0.01),
            patch.object(plugin_module, "_REPLYER_PREPARED_SEGMENT_TIMEOUT_SECONDS", 0.04),
        ):
            result = await plugin.handle_maisaka_replyer_after_response(
                response=response_text,
                session_id="stream-double-timeout",
            )
            await asyncio.sleep(0)
        return result, attempt_count, cancelled_count

    _reset_global_state()
    try:
        result, attempt_count, cancelled_count = asyncio.run(_run())

        assert result == {"action": "continue"}
        assert attempt_count == 2
        assert cancelled_count == 2
        assert _pop_prepared_segments("stream-double-timeout", response_text) is None
    finally:
        _reset_global_state()


def test_maisaka_after_response_delayed_retry_handles_randomized_texts() -> None:
    """不同长度正文触发尾延迟重试时，都必须采用同模型返回的精确分段。"""
    random_source = random.Random(20260727)
    response_cases: list[tuple[str, list[str]]] = []
    for _ in range(8):
        left = "".join(random_source.choice("甲乙丙丁戊己庚辛壬癸") for _ in range(random_source.randint(1, 24)))
        right = "".join(random_source.choice("春夏秋冬东西南北") for _ in range(random_source.randint(1, 24)))
        response_cases.append((left + right, [left, right]))

    async def _run() -> None:
        plugin = SmartSegmentationPlugin()
        runtime_settings = {
            "min_length": 1,
            "max_segments": 8,
            "temperature": 0.3,
            "max_tokens": 600,
            "style": "natural",
            "model_name": "doubao1.6",
        }

        for case_index, (response_text, expected_segments) in enumerate(response_cases):
            attempt_count = 0

            async def _segment_text(*_args, **_kwargs):
                nonlocal attempt_count
                attempt_count += 1
                assert _kwargs["model_name"] == "doubao1.6"
                if attempt_count == 1:
                    await asyncio.Future()
                return expected_segments

            with (
                patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
                patch.object(plugin, "_segment_text", AsyncMock(side_effect=_segment_text)),
                patch.object(plugin_module, "_REPLYER_SEGMENT_RETRY_DELAY_SECONDS", 0.001),
                patch.object(plugin_module, "_REPLYER_PREPARED_SEGMENT_TIMEOUT_SECONDS", 0.1),
            ):
                result = await plugin.handle_maisaka_replyer_after_response(
                    response=response_text,
                    session_id=f"stream-random-retry-{case_index}",
                )

            assert result == {"action": "continue"}
            assert attempt_count == 2
            assert _pop_prepared_segments(f"stream-random-retry-{case_index}", response_text) == expected_segments

    _reset_global_state()
    try:
        asyncio.run(_run())
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
        "typing_enabled": True,
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


def test_active_command_marker_expires_if_after_execute_hook_never_arrives() -> None:
    """宿主漏掉 after_execute 时，命令标记不能永久禁用该流的智能分段。"""
    _reset_global_state()
    stream_id = "stream-lost-command-hook"
    try:
        with patch.object(plugin_module.time, "monotonic", return_value=0.0):
            _mark_command_stream_active(stream_id)
            assert _is_command_stream_active(stream_id)

        with patch.object(plugin_module.time, "monotonic", return_value=24 * 60 * 60.0):
            assert not _is_command_stream_active(stream_id)

        assert stream_id not in plugin_module._active_command_streams
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
        "typing_enabled": True,
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


def test_new_command_prunes_expired_grace_windows_from_other_streams() -> None:
    """不同聊天流持续执行命令时，过期保护窗口不能无界累积。"""
    _reset_global_state()
    try:
        with patch.object(plugin_module.time, "monotonic", return_value=0.0):
            _mark_command_stream_active("stream-expired-command")
            _mark_command_stream_inactive("stream-expired-command")

        with patch.object(
            plugin_module.time,
            "monotonic",
            return_value=_COMMAND_REPLY_GRACE_SECONDS + 1.0,
        ):
            _mark_command_stream_active("stream-current-command")

        assert "stream-expired-command" not in plugin_module._recent_command_stream_expiries
        assert "stream-current-command" in plugin_module._recent_command_stream_expiries
    finally:
        _reset_global_state()


def test_is_action_only_text_detects_full_wrap_brackets() -> None:
    # 全角动作描述：常见的角色扮演旁白
    assert _is_action_only_text("（理理缓缓起身）")
    assert _is_action_only_text("  （理理缓缓起身）  ")
    # 半角括号同样视为动作描述
    assert _is_action_only_text("(walks slowly toward the window)")
    # 中文方括号也算
    assert _is_action_only_text("【动作：理理朝你挥了挥手】")
    # 嵌套括号仍然算"整段"
    assert _is_action_only_text("（理理（轻声）说道）")


def test_is_action_only_text_rejects_mixed_content() -> None:
    # 括号外还有正文，不算纯括号
    assert not _is_action_only_text("你好啊（理理挥手）")
    assert not _is_action_only_text("（理理挥手）你好啊")
    # 中间断开（先闭再开）不算整段一对包裹
    assert not _is_action_only_text("（前半）（后半）")
    # 单边括号
    assert not _is_action_only_text("（理理缓缓起身")
    assert not _is_action_only_text("理理缓缓起身）")
    # 空文本
    assert not _is_action_only_text("")
    assert not _is_action_only_text("   ")


def test_has_unbalanced_brackets_detects_missing_pair() -> None:
    assert _has_unbalanced_brackets("（理理")
    assert _has_unbalanced_brackets("理理）")
    assert _has_unbalanced_brackets("a (b")
    assert not _has_unbalanced_brackets("（理理缓缓起身）")
    assert not _has_unbalanced_brackets("a (b) c")
    assert not _has_unbalanced_brackets("")


def test_merge_segments_balancing_brackets_joins_split_action() -> None:
    # 模型把"（理理缓缓起身）"切成了两段，应该合回单段
    assert _merge_segments_balancing_brackets(["你好啊（理理", "缓缓起身）今天怎么样"]) == [
        "你好啊（理理缓缓起身）今天怎么样",
    ]
    # 跨多段也要合并到括号闭合为止
    assert _merge_segments_balancing_brackets(["a（b", "c", "d）e"]) == ["a（bcd）e"]
    # 已经平衡的分段保持不变
    assert _merge_segments_balancing_brackets(["你好啊", "今天怎么样"]) == ["你好啊", "今天怎么样"]
    # 含完整括号的段不影响后续分段
    assert _merge_segments_balancing_brackets(["你好啊（理理挥手）", "今天怎么样"]) == [
        "你好啊（理理挥手）",
        "今天怎么样",
    ]
    # 原文里就缺一半括号时，剩余 buffer 仍要原样吐出，不能吞段
    assert _merge_segments_balancing_brackets(["a（b", "c"]) == ["a（bc"]


def test_segment_text_splits_brackets_into_independent_segments() -> None:
    """_segment_text 应该把括号包裹的动作描述拆成独立段，而不是与正文混在一段。"""
    plugin = SmartSegmentationPlugin()
    # 模型返回单段未拆分；插件后处理负责按括号边界拆出动作描述
    fake_llm_result = {
        "success": True,
        "response": '["你好啊（理理缓缓起身）今天怎么样"]',
        "model_name": "test-model",
    }

    async def fake_resolve(_configured_name: str) -> tuple[str, str]:
        return ("task", "")

    # SDK 里 ctx 是只读 property，没法直接 patch.object；这里走内部 _ctx 注入。
    ctx_mock = MagicMock()
    ctx_mock.llm.generate = AsyncMock(return_value=fake_llm_result)
    plugin._ctx = ctx_mock

    with patch.object(plugin, "_resolve_generation_model", side_effect=fake_resolve):
        segments = asyncio.run(
            plugin._segment_text(
                "你好啊（理理缓缓起身）今天怎么样",
                style="natural",
                model_name="",
                max_segments=8,
                temperature=0.3,
                max_tokens=600,
            )
        )

    assert segments == ["你好啊", "（理理缓缓起身）", "今天怎么样"]


def test_segment_text_recovers_when_model_splits_brackets() -> None:
    """模型把括号切错（断在括号中间）时，先合并再按括号边界拆出独立段。"""
    plugin = SmartSegmentationPlugin()
    fake_llm_result = {
        "success": True,
        "response": '["你好啊（理理", "缓缓起身）今天怎么样"]',
        "model_name": "test-model",
    }

    async def fake_resolve(_configured_name: str) -> tuple[str, str]:
        return ("task", "")

    ctx_mock = MagicMock()
    ctx_mock.llm.generate = AsyncMock(return_value=fake_llm_result)
    plugin._ctx = ctx_mock

    with patch.object(plugin, "_resolve_generation_model", side_effect=fake_resolve):
        segments = asyncio.run(
            plugin._segment_text(
                "你好啊（理理缓缓起身）今天怎么样",
                style="natural",
                model_name="",
                max_segments=8,
                temperature=0.3,
                max_tokens=600,
            )
        )

    assert segments == ["你好啊", "（理理缓缓起身）", "今天怎么样"]


def test_segment_text_keeps_model_bracket_width_through_content_check() -> None:
    """保真校验忽略括号宽度等标点差异，模型的半角括号结果原样保留。"""
    plugin = SmartSegmentationPlugin()
    original_text = (
        "谁害羞了！\n\n"
        "（拍他手，没拍开）\n\n"
        "你别捏了  还疼着呢\n\n"
        "昨晚被你揉了一整晚还不够吗\n\n"
        "（声音闷在枕头里）\n\n"
        "久远哥你够了\n\n"
        "我要起床了\n\n"
        "……你再捏我真踹你下床"
    )
    ctx_mock = MagicMock()
    ctx_mock.llm.generate = AsyncMock(
        return_value={
            "success": True,
            "response": (
                '["谁害羞了！", "(拍他手，没拍开)", "你别捏了", "还疼着呢", '
                '"昨晚被你揉了一整晚还不够吗", "(声音闷在枕头里)", '
                '"久远哥你够了", "我要起床了", "……你再捏我真踹你下床"]'
            ),
            "model_name": "test-model",
        }
    )
    plugin._ctx = ctx_mock

    with patch.object(
        plugin,
        "_resolve_generation_model",
        AsyncMock(return_value=("task", "")),
    ):
        segments = asyncio.run(
            plugin._segment_text(
                original_text,
                style="natural",
                model_name="",
                max_segments=8,
                temperature=0.3,
                max_tokens=600,
            )
        )

    assert segments == [
        "谁害羞了！",
        "(拍他手，没拍开)",
        "你别捏了",
        "还疼着呢",
        "昨晚被你揉了一整晚还不够吗",
        "(声音闷在枕头里)",
        "久远哥你够了",
        "我要起床了……你再捏我真踹你下床",
    ]


def test_segment_text_accepts_standalone_pause_omitted_by_model() -> None:
    plugin = SmartSegmentationPlugin()
    original_text = (
        "……\n\n"
        "你能不能别问这种明知故问的问题\n\n"
        "腿都抖了一整天了你说呢\n\n"
        "但是寸止那个真的过分\n\n"
        "下次再那样我咬你"
    )
    ctx_mock = MagicMock()
    ctx_mock.llm.generate = AsyncMock(
        return_value={
            "success": True,
            "response": (
                '["你能不能别问这种明知故问的问题", "腿都抖了一整天了你说呢", '
                '"但是寸止那个真的过分", "下次再那样我咬你"]'
            ),
            "model_name": "test-model",
        }
    )
    plugin._ctx = ctx_mock

    with patch.object(
        plugin,
        "_resolve_generation_model",
        AsyncMock(return_value=("task", "")),
    ):
        segments = asyncio.run(
            plugin._segment_text(
                original_text,
                style="natural",
                model_name="",
                max_segments=16,
                temperature=0.3,
                max_tokens=600,
            )
        )

    assert segments == [
        "你能不能别问这种明知故问的问题",
        "腿都抖了一整天了你说呢",
        "但是寸止那个真的过分",
        "下次再那样我咬你",
    ]


def test_segment_text_merges_model_overflow_without_restoring_original_punctuation() -> None:
    plugin = SmartSegmentationPlugin()
    ctx_mock = MagicMock()
    ctx_mock.llm.generate = AsyncMock(
        return_value={
            "success": True,
            "response": '["甲", "乙", "丙"]',
            "model_name": "test-model",
        }
    )
    plugin._ctx = ctx_mock

    with patch.object(
        plugin,
        "_resolve_generation_model",
        AsyncMock(return_value=("task", "")),
    ):
        segments = asyncio.run(
            plugin._segment_text(
                "甲，乙，丙",
                style="natural",
                model_name="",
                max_segments=2,
                temperature=0.3,
                max_tokens=600,
            )
        )

    assert segments == ["甲", "乙丙"]


def test_segment_text_safely_rejects_unexpected_llm_result_shape() -> None:
    plugin = SmartSegmentationPlugin()
    ctx_mock = MagicMock()
    ctx_mock.llm.generate = AsyncMock(return_value=["不是宿主约定的结果对象"])
    plugin._ctx = ctx_mock

    with patch.object(
        plugin,
        "_resolve_generation_model",
        AsyncMock(return_value=("task", "")),
    ):
        segments = asyncio.run(
            plugin._segment_text(
                "这是一段等待分段的完整正文",
                style="natural",
                model_name="",
                max_segments=8,
                temperature=0.3,
                max_tokens=600,
            )
        )

    assert segments is None


def test_segment_text_safely_handles_model_resolution_failure() -> None:
    plugin = SmartSegmentationPlugin()
    plugin._ctx = MagicMock()

    with patch.object(
        plugin,
        "_resolve_generation_model",
        AsyncMock(side_effect=RuntimeError("model config unavailable")),
    ):
        segments = asyncio.run(
            plugin._segment_text(
                "这是一段等待分段的完整正文",
                style="natural",
                model_name="",
                max_segments=8,
                temperature=0.3,
                max_tokens=600,
            )
        )

    assert segments is None


def test_split_text_at_brackets_extracts_bracketed_blocks() -> None:
    assert _split_text_at_brackets("你好啊（理理缓缓起身）今天怎么样") == [
        "你好啊",
        "（理理缓缓起身）",
        "今天怎么样",
    ]
    # 起始就是括号
    assert _split_text_at_brackets("（动作）然后呢") == ["（动作）", "然后呢"]
    # 末尾就是括号
    assert _split_text_at_brackets("先讲一下（动作）") == ["先讲一下", "（动作）"]
    # 多个括号块
    assert _split_text_at_brackets("a（b）c（d）e") == ["a", "（b）", "c", "（d）", "e"]
    # 嵌套括号视为一个整块
    assert _split_text_at_brackets("（外（内）外）") == ["（外（内）外）"]
    # 没有括号
    assert _split_text_at_brackets("纯文本没有括号") == ["纯文本没有括号"]
    # 空文本
    assert _split_text_at_brackets("") == []
    # 未闭合的括号不拆，保持原样落入同一段
    assert _split_text_at_brackets("你好啊（理理") == ["你好啊（理理"]


def test_split_segments_at_bracket_boundaries_respects_max_segments() -> None:
    # 多段都按括号边界拆开，结果是独立段
    assert _split_segments_at_bracket_boundaries(
        ["你好啊（动作1）今天好吗", "再见（动作2）"],
        max_segments=8,
    ) == ["你好啊", "（动作1）", "今天好吗", "再见", "（动作2）"]

    # max_segments 限制：超出的尾部段合并回最后一段，避免吃掉内容
    assert _split_segments_at_bracket_boundaries(
        ["a（b）c（d）e"],
        max_segments=3,
    ) == ["a", "（b）", "c（d）e"]

    # 空输入透传
    assert _split_segments_at_bracket_boundaries([], max_segments=8) == []

    # 没有括号时与输入一致（清掉空白）
    assert _split_segments_at_bracket_boundaries(
        ["你好啊", "今天怎么样"],
        max_segments=8,
    ) == ["你好啊", "今天怎么样"]


def test_maisaka_after_response_skips_action_only_text() -> None:
    """整段消息就是括号包裹的动作描述时，早期路径应直接跳过，不调用 LLM 分段。"""
    plugin = SmartSegmentationPlugin()
    runtime_settings = {
        "min_length": 8,
        "max_segments": 8,
        "temperature": 0.3,
        "max_tokens": 600,
        "style": "natural",
        "model_name": "",
        "typing_enabled": True,
    }
    segment_text_mock = AsyncMock(return_value=["不应被调用"])

    try:
        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            result = asyncio.run(
                plugin.handle_maisaka_replyer_after_response(
                    response="（理理缓缓起身，目光柔和地看向窗外的远方）",
                    session_id="stream-action-only",
                )
            )

        assert result == {"action": "continue"}
        segment_text_mock.assert_not_awaited()
        assert not plugin_module._prepared_segment_registry
    finally:
        _reset_global_state()


def test_build_segmentation_prompt_mentions_bracket_rule() -> None:
    # prompt 必须显式声明括号要独立成段，避免提示词漂移时悄无声息地回归
    prompt = SmartSegmentationPlugin._build_segmentation_prompt(
        "你好啊（理理缓缓起身）今天怎么样",
        style="natural",
        max_segments=8,
    )
    assert "括号" in prompt
    assert "独立" in prompt


def test_segmentation_prompt_stays_compact_without_losing_natural_chat_calibration() -> None:
    text = "不是你先别急，我刚刚看了一下，应该只是配置没有生效，重启一下插件再试试。"
    prompt = SmartSegmentationPlugin._build_segmentation_prompt(
        text,
        style="natural",
        max_segments=16,
    )

    assert "像和朋友微信聊天一样自然地分条发送。有的消息短有的长，节奏随意。" in prompt
    assert "相关的内容放在一条里" in prompt
    assert "长短" in prompt and "不均匀" in prompt
    assert '["哈哈真的吗", "那太好了！我还以为你不喜欢呢", "下次我们一起去看电影吧，最近有个新片子挺有意思的"]' in prompt
    assert "不要改写原意" in prompt
    assert "最多分成 16 条" in prompt
    assert "通常建议分成" not in prompt
    assert "即时反应、转折、补充说明、问题、建议和收尾" not in prompt
    assert len(prompt) - len(text) <= 520


def test_segmentation_prompt_calibrates_clear_boundaries_without_punctuation() -> None:
    prompt = SmartSegmentationPlugin._build_segmentation_prompt(
        "你买早饭我到了要吃的",
        style="natural",
        max_segments=16,
    )

    assert "原文没有标点" in prompt
    assert "明显的自然发送边界" in prompt
    assert '原文："你买早饭我到了要吃的"' in prompt
    assert '分条：["你买早饭", "我到了要吃的"]' in prompt


def test_strip_thinking_content_covers_think_tag_variants() -> None:
    assert _strip_thinking_content("<think>推理过程</think>正文") == "正文"
    assert _strip_thinking_content("<thinking>推理过程</thinking>正文") == "正文"
    assert _strip_thinking_content("<THINK>大写推理</THINK>正文") == "正文"
    # 未闭合时只剥离标签本身，保留可见内容
    assert _strip_thinking_content("正文<think>残留标签") == "正文残留标签"


def test_extract_json_array_text_handles_fence_variants() -> None:
    assert _extract_json_array_text('```json\n["甲", "乙"]\n```') == '["甲", "乙"]'
    # 大写围栏与围栏内前置说明文字都不应破坏提取
    assert _extract_json_array_text('```JSON\n分条结果：["甲", "乙"]\n```') == '["甲", "乙"]'
    assert _extract_json_array_text('好的，分条如下：["甲", "乙"] 希望符合要求') == '["甲", "乙"]'
    assert _extract_json_array_text('```\n["甲"]\n```') == '["甲"]'


def test_content_check_ignores_punctuation_width_and_case() -> None:
    # 全半角、大小写、标点、空白差异均放行
    assert _segments_preserve_original_content("Ｈｅｌｌｏ，世界！！", ["hello", "世界"])
    assert _segments_preserve_original_content("你好啊。（笑）", ["你好啊", "(笑)"])
    assert _segments_preserve_original_content("……\n\n先睡了", ["先睡了"])
    # 字词级改写与内容缺失必须被拦截
    assert not _segments_preserve_original_content("我明天再去", ["我今天再去"])
    assert not _segments_preserve_original_content("先吃饭然后去看电影", ["先吃饭"])


def test_segment_text_rejects_dropped_sentence_and_falls_back() -> None:
    """模型丢掉半句话时必须丢弃整个分段结果，不能带着缺失内容发出去。"""
    plugin = SmartSegmentationPlugin()
    ctx_mock = MagicMock()
    ctx_mock.llm.generate = AsyncMock(
        return_value={
            "success": True,
            "response": '["今天先聊到这"]',
            "model_name": "test-model",
        }
    )
    plugin._ctx = ctx_mock

    with patch.object(
        plugin,
        "_resolve_generation_model",
        AsyncMock(return_value=("task", "")),
    ):
        segments = asyncio.run(
            plugin._segment_text(
                "今天先聊到这，明天记得叫我起床",
                style="natural",
                model_name="",
                max_segments=8,
                temperature=0.3,
                max_tokens=600,
            )
        )

    assert segments is None


def test_delayed_retry_recovers_when_first_result_rewrites_text() -> None:
    """首次结果改写字词被保真校验拒绝后，重试拿到忠实结果应被采用。"""
    plugin = SmartSegmentationPlugin()
    original_text = "先去吃饭，回来再打游戏"
    ctx_mock = MagicMock()
    ctx_mock.llm.generate = AsyncMock(
        side_effect=[
            {"success": True, "response": '["先去恰饭", "回来再打游戏"]', "model_name": "test-model"},
            {"success": True, "response": '["先去吃饭", "回来再打游戏"]', "model_name": "test-model"},
        ]
    )
    plugin._ctx = ctx_mock

    with patch.object(
        plugin,
        "_resolve_generation_model",
        AsyncMock(return_value=("task", "")),
    ):
        segments = asyncio.run(
            plugin._segment_text_with_delayed_retry(
                original_text,
                style="natural",
                model_name="",
                max_segments=8,
                temperature=0.3,
                max_tokens=600,
            )
        )

    assert segments == ["先去吃饭", "回来再打游戏"]
    assert ctx_mock.llm.generate.await_count == 2


def test_segment_text_scales_max_tokens_with_input_length() -> None:
    """长文按原文长度扩容 max_tokens，短文保持配置下限，避免 JSON 输出被截断。"""
    plugin = SmartSegmentationPlugin()
    long_text = "今天路上看到一只小猫在晒太阳" * 40
    ctx_mock = MagicMock()
    ctx_mock.llm.generate = AsyncMock(
        return_value={
            "success": True,
            "response": json.dumps([long_text], ensure_ascii=False),
            "model_name": "test-model",
        }
    )
    plugin._ctx = ctx_mock

    with patch.object(
        plugin,
        "_resolve_generation_model",
        AsyncMock(return_value=("task", "")),
    ):
        segments = asyncio.run(
            plugin._segment_text(
                long_text,
                style="natural",
                model_name="",
                max_segments=8,
                temperature=0.3,
                max_tokens=600,
            )
        )

        assert segments == [long_text]
        assert ctx_mock.llm.generate.await_args.kwargs["max_tokens"] == len(long_text) * 2 + 160

        short_text = "哈哈好呀"
        ctx_mock.llm.generate = AsyncMock(
            return_value={
                "success": True,
                "response": '["哈哈", "好呀"]',
                "model_name": "test-model",
            }
        )
        asyncio.run(
            plugin._segment_text(
                short_text,
                style="natural",
                model_name="",
                max_segments=8,
                temperature=0.3,
                max_tokens=600,
            )
        )
        assert ctx_mock.llm.generate.await_args.kwargs["max_tokens"] == 600
