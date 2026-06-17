import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.smart_segmentation_plugin import plugin as plugin_module
from plugins.smart_segmentation_plugin.plugin import (
    _COMMAND_REPLY_GRACE_SECONDS,
    SmartSegmentationPlugin,
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
    _split_segments_at_bracket_boundaries,
    _split_text_at_brackets,
    _store_prepared_segments,
)


def _reset_global_state() -> None:
    plugin_module._prepared_segment_registry.clear()
    plugin_module._pending_follow_up_segments.clear()
    plugin_module._stream_resend_guards.clear()
    plugin_module._active_command_streams.clear()
    plugin_module._recent_command_stream_expiries.clear()
    plugin_module._active_follow_up_tasks.clear()
    follow_up_tasks_by_stream = getattr(plugin_module, "_active_follow_up_tasks_by_stream", None)
    if isinstance(follow_up_tasks_by_stream, dict):
        follow_up_tasks_by_stream.clear()
    follow_up_idle_events = getattr(plugin_module, "_follow_up_idle_events_by_stream", None)
    if isinstance(follow_up_idle_events, dict):
        follow_up_idle_events.clear()


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


def test_prepared_segments_match_host_postprocessed_bracket_placeholder() -> None:
    """宿主删除 `[表情包]` 后，发送阶段仍应命中 replyer 预分段缓存。"""
    _reset_global_state()
    raw_response = "哼\n\n我就当你是喜欢我这嘴毒的性格了\n\n[表情包]"
    outbound_text = "哼\n\n我就当你是喜欢我这嘴毒的性格了"

    try:
        ok = _store_prepared_segments(
            "stream-placeholder",
            raw_response,
            ["哼", "我就当你是喜欢我这嘴毒的性格了", "[表情包]"],
        )
        assert ok
        assert _hash_normalized_text(raw_response) == _hash_normalized_text(outbound_text)

        segments = _pop_prepared_segments("stream-placeholder", outbound_text)
        assert segments == ["哼", "我就当你是喜欢我这嘴毒的性格了"]
        assert not plugin_module._prepared_segment_registry
    finally:
        _reset_global_state()


def test_cache_key_mirrors_host_bracket_postprocess_pattern() -> None:
    """缓存键必须镜像宿主的括号清理规则，而不是插件侧自行收窄。"""
    raw_response = "英文 [emoji] 中文"
    outbound_text = "英文 中文"

    assert _hash_normalized_text(raw_response) == _hash_normalized_text(outbound_text)


def test_prepared_segments_keep_normal_action_bracket_segments() -> None:
    """只移除非文本占位符，不能误删正常动作括号段。"""
    _reset_global_state()
    raw_response = "你好啊（挥手）今天怎么样"
    outbound_text = "你好啊今天怎么样"

    try:
        ok = _store_prepared_segments(
            "stream-action-bracket",
            raw_response,
            ["你好啊", "（挥手）", "今天怎么样"],
        )
        assert ok

        segments = _pop_prepared_segments("stream-action-bracket", outbound_text)
        assert segments == ["你好啊", "（挥手）", "今天怎么样"]
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
            delay_base=0.1,
            delay_per_char=0.2,
            delay_max=0.3,
            delay_before_first=True,
            sync_to_maisaka_history=True,
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
            "delay_base": 0.0,
            "delay_per_char": 0.0,
            "delay_max": 0.0,
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
            delay_base=0.0,
            delay_per_char=0.0,
            delay_max=0.0,
            delay_before_first=True,
            sync_to_maisaka_history=True,
        )
        assert not plugin_module._pending_follow_up_segments
    finally:
        _reset_global_state()


def test_chat_receive_before_process_waits_until_same_stream_follow_ups_finish() -> None:
    """入站消息处理前的轻量阻塞 hook 必须等到同 stream 的补发真正结束。

    这是线上严重回归的硬防线：插件不能让下一轮 planner 在"只看到首段、后续分段仍在
    后台 sleep/send"的窗口里抢跑。阻塞从吞 20MB prompt 的 maisaka.planner.before_request
    迁到只带入站消息的 chat.receive.before_process 后，这条防线必须照样生效。
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
            "delay_base": 0.0,
            "delay_per_char": 0.0,
            "delay_max": 0.0,
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
        with (
            patch.object(plugin_module, "_presync_follow_up_segments_to_maisaka_history", return_value=False),
            patch.object(plugin, "_send_segments", send_segments_mock),
        ):
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

    with patch.object(plugin_module.asyncio, "sleep", AsyncMock()):
        ok = asyncio.run(
            plugin._send_segments(
                "stream-history",
                ["第二段", "第三段"],
                delay_base=0.1,
                delay_per_char=0.2,
                delay_max=0.3,
                delay_before_first=True,
            )
        )

    assert ok is True
    assert send_text_mock.await_count == 2
    for call in send_text_mock.await_args_list:
        # ctx.send.text(segment, stream_id, sync_to_maisaka_history=True, maisaka_source_kind="guided_reply")
        assert call.kwargs.get("sync_to_maisaka_history") is True
        assert call.kwargs.get("maisaka_source_kind") == "guided_reply"


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


def test_after_build_consumes_prepared_cache_after_host_removes_placeholder() -> None:
    """replyer 原文含 `[表情包]` 但出站文本已被宿主清理时，仍应命中预分段缓存。"""
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
    raw_response = "哼\n\n我就当你是喜欢我这嘴毒的性格了\n\n[表情包]"
    outbound_text = "哼\n\n我就当你是喜欢我这嘴毒的性格了"
    segment_text_mock = AsyncMock(return_value=["哼", "我就当你是喜欢我这嘴毒的性格了", "[表情包]"])

    try:
        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            early_result = asyncio.run(
                plugin.handle_maisaka_replyer_after_response(
                    response=raw_response,
                    session_id="stream-placeholder-hit",
                )
            )
            result = asyncio.run(
                plugin.handle_smart_segmentation_after_build(
                    message={
                        "message_id": "placeholder-msg-1",
                        "session_id": "stream-placeholder-hit",
                        "timestamp": "7000.0",
                        "raw_message": [{"type": "text", "data": outbound_text}],
                        "message_info": {"additional_config": {}},
                    },
                    stream_id="stream-placeholder-hit",
                    processed_plain_text=outbound_text,
                )
            )

        assert early_result == {"action": "continue"}
        segment_text_mock.assert_awaited_once()
        assert result["action"] == "continue"
        assert result["modified_kwargs"]["processed_plain_text"] == "哼"
        assert not plugin_module._prepared_segment_registry
        pending = next(iter(plugin_module._pending_follow_up_segments.values()))
        assert pending["segments"] == ["我就当你是喜欢我这嘴毒的性格了"]
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


def test_maisaka_after_response_skips_when_segment_text_returns_none() -> None:
    """LLM 分段失败返回 None 时，replyer hook 应直接放行且不登记空缓存。"""
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
    segment_text_mock = AsyncMock(return_value=None)

    try:
        with (
            patch.object(plugin, "_get_segmentation_runtime_settings", AsyncMock(return_value=runtime_settings)),
            patch.object(plugin, "_segment_text", segment_text_mock),
        ):
            result = asyncio.run(
                plugin.handle_maisaka_replyer_after_response(
                    response="这是一段足够长的回复文本，但这次模型分段失败返回 None。",
                    session_id="stream-none",
                )
            )

        assert result == {"action": "continue"}
        segment_text_mock.assert_awaited_once()
        assert not plugin_module._prepared_segment_registry
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


def test_presync_follow_up_segments_appends_clones_to_maisaka_history() -> None:
    """预先同步剩余段进 maisaka._chat_history：每段克隆首段 SessionMessage，
    改写 raw_message / processed_plain_text / message_id 后调
    ``runtime.append_sent_message_to_chat_history``。
    """
    _reset_global_state()

    # 构造一个最小的可被 deserialize_session_message 接受的 message dict。
    # 字段参考 PluginMessageUtils._session_message_to_dict 的产物。
    first_message_dict = {
        "message_id": "platform-msg-9001",
        "session_id": "stream-presync",
        "timestamp": "2026-05-29T08:00:00",
        "platform": "qq",
        "is_at": False,
        "is_emoji": False,
        "is_picture": False,
        "is_command": False,
        "is_mentioned": False,
        "is_notify": False,
        "processed_plain_text": "首段",
        "raw_message": [{"type": "text", "data": "首段"}],
        "message_info": {
            "user_info": {
                "user_id": "10001",
                "user_nickname": "理理",
                "user_cardname": "理理",
            },
            "group_info": {
                "group_id": "20001",
                "group_name": "测试群",
            },
            "additional_config": {},
        },
    }

    appended: list[tuple[Any, str]] = []

    class _FakeRuntime:
        def append_sent_message_to_chat_history(self, message, *, source_kind="guided_reply"):
            appended.append((message, source_kind))
            return True

    fake_runtime = _FakeRuntime()

    class _FakeHeartflowManager:
        heartflow_chat_list = {"stream-presync": fake_runtime}

    fake_manager = _FakeHeartflowManager()

    # _presync_follow_up_segments_to_maisaka_history 内部用 from src... import 拿
    # heartflow_manager，patch 必须落到目标模块属性。
    import importlib

    heartflow_module = importlib.import_module("src.chat.heart_flow.heartflow_manager")

    try:
        with patch.object(heartflow_module, "heartflow_manager", fake_manager):
            ok = plugin_module._presync_follow_up_segments_to_maisaka_history(
                stream_id="stream-presync",
                first_message_dict=first_message_dict,
                follow_up_segments=["段二", "段三"],
            )

        assert ok is True
        assert len(appended) == 2

        first_clone, first_source = appended[0]
        # 必须改写文本，避免后续段拿到首段同样的内容
        assert first_clone.processed_plain_text == "段二"
        assert [c.text for c in first_clone.raw_message.components] == ["段二"]
        # message_id 必须与首段不同，且与下一段也不同，避免合并/去重
        assert first_clone.message_id != "platform-msg-9001"
        # source_kind 必须是 guided_reply，沿用 maisaka 自己 reply 工具的语义
        assert first_source == "guided_reply"

        second_clone, _ = appended[1]
        assert second_clone.processed_plain_text == "段三"
        assert second_clone.message_id != first_clone.message_id
        # message_info 应该跟首段保持一致（同一用户/群）
        assert second_clone.message_info.user_info.user_id == "10001"
        assert second_clone.message_info.group_info.group_id == "20001"
    finally:
        _reset_global_state()


def test_presync_returns_false_when_runtime_missing_so_send_falls_back_to_sync_history() -> None:
    """maisaka runtime 不存在时（例如插件 reload 中途）必须返回 False，
    让 _run_follow_up_segments 退回到老的"send_service 同步入库"路径，
    避免后续段在 maisaka 历史中彻底丢失。"""
    _reset_global_state()

    class _EmptyHeartflowManager:
        heartflow_chat_list: dict[str, Any] = {}

    import importlib

    heartflow_module = importlib.import_module("src.chat.heart_flow.heartflow_manager")

    first_message_dict = {
        "message_id": "platform-msg-9100",
        "session_id": "stream-missing",
        "timestamp": "2026-05-29T08:01:00",
        "platform": "qq",
        "is_at": False,
        "is_emoji": False,
        "is_picture": False,
        "is_command": False,
        "is_mentioned": False,
        "is_notify": False,
        "processed_plain_text": "首段",
        "raw_message": [{"type": "text", "data": "首段"}],
        "message_info": {
            "user_info": {"user_id": "10001", "user_nickname": "理理", "user_cardname": "理理"},
            "group_info": {"group_id": "20001", "group_name": "群"},
            "additional_config": {},
        },
    }

    with patch.object(heartflow_module, "heartflow_manager", _EmptyHeartflowManager()):
        ok = plugin_module._presync_follow_up_segments_to_maisaka_history(
            stream_id="stream-missing",
            first_message_dict=first_message_dict,
            follow_up_segments=["段二"],
        )
    assert ok is False


def test_after_send_disables_send_service_history_sync_when_presync_succeeded() -> None:
    """端到端：after_send 命中 pending 后预同步成功 → 后台发送禁用
    send_service 的 sync_to_maisaka_history，避免每段被记两次。
    """
    _reset_global_state()
    plugin = SmartSegmentationPlugin()

    tracking_key = plugin_module._build_follow_up_tracking_key(
        stream_id="stream-e2e-presync",
        timestamp="2026-05-29T09:00:00",
        visible_text="首段",
    )
    plugin_module._register_pending_follow_up_segments(
        lookup_keys=["msg-e2e", tracking_key],
        pending_data={
            "stream_id": "stream-e2e-presync",
            "segments": ["段二", "段三"],
            "delay_base": 0.0,
            "delay_per_char": 0.0,
            "delay_max": 0.0,
        },
    )

    appended_to_history: list[Any] = []

    class _FakeRuntime:
        def append_sent_message_to_chat_history(self, message, *, source_kind="guided_reply"):
            appended_to_history.append(message)
            return True

    class _FakeHeartflowManager:
        heartflow_chat_list = {"stream-e2e-presync": _FakeRuntime()}

    import importlib

    heartflow_module = importlib.import_module("src.chat.heart_flow.heartflow_manager")

    send_segments_mock = AsyncMock(return_value=True)

    async def _run() -> None:
        with patch.object(heartflow_module, "heartflow_manager", _FakeHeartflowManager()):
            with patch.object(plugin, "_send_segments", send_segments_mock):
                await plugin.handle_smart_segmentation_after_send(
                    message={
                        "message_id": "msg-e2e",
                        "session_id": "stream-e2e-presync",
                        "timestamp": "2026-05-29T09:00:00",
                        "platform": "qq",
                        "is_at": False,
                        "is_emoji": False,
                        "is_picture": False,
                        "is_command": False,
                        "is_mentioned": False,
                        "is_notify": False,
                        "processed_plain_text": "首段",
                        "raw_message": [{"type": "text", "data": "首段"}],
                        "message_info": {
                            "user_info": {
                                "user_id": "10001",
                                "user_nickname": "理理",
                                "user_cardname": "理理",
                            },
                            "group_info": {"group_id": "20001", "group_name": "群"},
                            "additional_config": {},
                        },
                    },
                    sent=True,
                )
                await plugin_module._drain_active_follow_up_tasks()

    try:
        asyncio.run(_run())

        # 1) 预同步把全部剩余段写进了 maisaka 历史（共 2 段）
        assert len(appended_to_history) == 2
        assert [m.processed_plain_text for m in appended_to_history] == ["段二", "段三"]

        # 2) 后台 _send_segments 被 sync_to_maisaka_history=False 调用——避免重复入库
        send_segments_mock.assert_awaited_once()
        call_kwargs = send_segments_mock.await_args.kwargs
        assert call_kwargs["sync_to_maisaka_history"] is False
        # 但 segments 内容与 delays 没被改
        assert send_segments_mock.await_args.args == ("stream-e2e-presync", ["段二", "段三"])
        assert call_kwargs["delay_before_first"] is True
    finally:
        _reset_global_state()


def test_send_segments_honors_sync_to_maisaka_history_false() -> None:
    """关闭 sync_to_maisaka_history 时，发送链调用必须把这个参数透传下去，
    否则即使上层走了预同步路径，下层依然会走 send_service 入库，造成重复。"""
    plugin = SmartSegmentationPlugin()
    send_text_mock = AsyncMock(return_value=True)
    mock_ctx = MagicMock()
    mock_ctx.send = MagicMock()
    mock_ctx.send.text = send_text_mock
    plugin._ctx = mock_ctx

    with patch.object(plugin_module.asyncio, "sleep", AsyncMock()):
        ok = asyncio.run(
            plugin._send_segments(
                "stream-no-sync",
                ["段一", "段二"],
                delay_base=0.0,
                delay_per_char=0.0,
                delay_max=0.0,
                sync_to_maisaka_history=False,
            )
        )

    assert ok is True
    assert send_text_mock.await_count == 2
    for call in send_text_mock.await_args_list:
        assert call.kwargs.get("sync_to_maisaka_history") is False
        assert call.kwargs.get("maisaka_source_kind") == "guided_reply"
