import base64

import pytest

import game_window_ocr.commentary as commentary_module
from game_window_ocr.commentary import (
    RealtimeSpeechClient,
    ResponsesCommentaryPlanner,
)


def _client() -> RealtimeSpeechClient:
    return RealtimeSpeechClient(
        api_key="test-key",
        model="gpt-realtime-2.1-mini",
        voice="marin",
        timeout=1,
    )


def test_narration_response_stays_out_of_band(tmp_path) -> None:
    client = _client()
    sent = []
    client._send = sent.append  # type: ignore[method-assign]
    client._receive = lambda: {  # type: ignore[method-assign]
        "type": "response.done",
        "response": {"id": "response-1", "status": "completed"},
    }

    client.speak(
        phase="narration",
        instructions="そのまま読む",
        wav_path=tmp_path / "narration.wav",
        playback=False,
    )

    response = sent[0]["response"]
    assert response["conversation"] == "none"
    assert response["input"] == []


def test_realtime_session_forbids_speech_announcements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    sent = []
    monkeypatch.setattr(
        commentary_module.websocket,
        "create_connection",
        lambda *args, **kwargs: object(),
    )
    client._send = sent.append  # type: ignore[method-assign]
    client._wait_for = lambda event_type: {}  # type: ignore[method-assign]

    client.__enter__()

    instructions = sent[0]["session"]["instructions"]
    assert "発話予告" in instructions
    assert "先頭文字から直ちに" in instructions


def test_speech_result_uses_first_audio_time_and_pcm_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    client = RealtimeSpeechClient(
        api_key="test-key",
        model="gpt-realtime-2.1-mini",
        voice="marin",
        timeout=1,
        timeline_origin=10.0,
    )
    audio = b"\x00" * 4_800
    events = iter(
        [
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(audio).decode("ascii"),
            },
            {
                "type": "response.output_audio_transcript.delta",
                "delta": "実況です。",
            },
            {
                "type": "response.done",
                "response": {"id": "response-1", "status": "completed"},
            },
        ]
    )
    client._send = lambda event: None  # type: ignore[method-assign]
    client._receive = lambda: next(events)  # type: ignore[method-assign]
    monkeypatch.setattr(
        commentary_module.time,
        "monotonic",
        lambda: 12.5,
    )

    result = client.speak(
        phase="commentary",
        instructions="実況する",
        wav_path=tmp_path / "commentary.wav",
        playback=False,
    )

    assert result.started_at_seconds == pytest.approx(2.5)
    assert result.ended_at_seconds == pytest.approx(2.6)
    assert result.audio_bytes == 4_800


def test_guarded_playback_waits_for_matching_transcript_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    client = _client()
    first_audio = b"\x01" * 100
    second_audio = b"\x02" * 100
    third_audio = b"\x03" * 100
    events = iter(
        [
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(first_audio).decode("ascii"),
            },
            {
                "type": "response.output_audio_transcript.delta",
                "delta": "ごきげんよう。",
            },
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(second_audio).decode("ascii"),
            },
            {
                "type": "response.output_audio_transcript.delta",
                "delta": "スカイナです。",
            },
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(third_audio).decode("ascii"),
            },
            {
                "type": "response.done",
                "response": {"id": "response-1", "status": "completed"},
            },
        ]
    )
    played: list[bytes] = []

    class FakeAudioSink:
        def __init__(self, *args, **kwargs) -> None:
            self.started = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def start_playback(self) -> bool:
            self.started = True
            return True

        def write(self, chunk: bytes) -> None:
            if self.started:
                played.append(chunk)

        def play_buffered(self, chunk: bytes) -> None:
            played.append(chunk)

    client._send = lambda event: None  # type: ignore[method-assign]
    client._receive = lambda: next(events)  # type: ignore[method-assign]
    monkeypatch.setattr(commentary_module, "AudioSink", FakeAudioSink)

    result = client.speak(
        phase="startup",
        instructions="完成稿だけを話す",
        wav_path=tmp_path / "startup.wav",
        playback=True,
        playback_prefix="ごきげんよう。スカイナです。",
    )

    assert result.playback_suppressed is False
    assert played == [first_audio + second_audio, third_audio]


def test_guarded_playback_suppresses_unplanned_startup_preface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    client = _client()
    audio = b"\x01" * 100
    events = iter(
        [
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(audio).decode("ascii"),
            },
            {
                "type": "response.output_audio_transcript.delta",
                "delta": "ごきげんよう。それでは、続きを始めますね。",
            },
            {
                "type": "response.done",
                "response": {"id": "response-1", "status": "completed"},
            },
        ]
    )
    played: list[bytes] = []

    class FakeAudioSink:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def start_playback(self) -> bool:
            played.append(b"started")
            return True

        def write(self, chunk: bytes) -> None:
            pass

        def play_buffered(self, chunk: bytes) -> None:
            played.append(chunk)

    client._send = lambda event: None  # type: ignore[method-assign]
    client._receive = lambda: next(events)  # type: ignore[method-assign]
    monkeypatch.setattr(commentary_module, "AudioSink", FakeAudioSink)

    result = client.speak(
        phase="startup",
        instructions="完成稿だけを話す",
        wav_path=tmp_path / "startup.wav",
        playback=True,
        playback_prefix="ごきげんよう。人間を学ぶ実況AI、スカイナです。",
    )

    assert result.playback_suppressed is True
    assert result.started_at_seconds is None
    assert played == []


class _FakeResponsesResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeResponsesHttpClient:
    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = iter(payloads)
        self.requests: list[dict] = []

    def post(self, path: str, *, json: dict) -> _FakeResponsesResponse:
        self.requests.append({"path": path, "json": json})
        return _FakeResponsesResponse(next(self._payloads))


def _responses_payload(response_id: str, output_text: str) -> dict:
    return {
        "id": response_id,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                    }
                ],
            }
        ],
    }


def test_responses_planner_uses_luna_schema_and_story_history() -> None:
    client = _FakeResponsesHttpClient(
        [
            _responses_payload(
                "resp-1",
                (
                    '{"mode":"silent","comment":"","emotion":"calm",'
                    '"intensity":0.0,"pace":"normal"}'
                ),
            ),
            _responses_payload(
                "resp-2",
                (
                    '{"mode":"quick","comment":"前の約束につながったね。",'
                    '"emotion":"thoughtful","intensity":0.6,"pace":"normal"}'
                ),
            ),
        ]
    )
    planner = ResponsesCommentaryPlanner(
        api_key="test-key",
        model="gpt-5.6-luna",
        timeout=1,
        client=client,
        prior_memory={
            "story_summary": "以前、山荘へ向かう約束をした。",
            "current_state": "山荘の入口。",
        },
    )

    first = planner.generate_text(
        phase="commentary_plan",
        instructions="最初の約束。感想をJSONで決める",
        use_conversation_history=True,
    )
    second = planner.generate_text(
        phase="commentary_plan",
        instructions="約束を果たした。過去の展開を踏まえる",
        use_conversation_history=True,
    )

    first_request = client.requests[0]["json"]
    second_request = client.requests[1]["json"]
    assert first.response_id == "resp-1"
    assert second.response_id == "resp-2"
    assert first_request["model"] == "gpt-5.6-luna"
    assert first_request["reasoning"] == {
        "effort": "low",
        "context": "all_turns",
    }
    assert first_request["text"]["format"]["type"] == "json_schema"
    assert first_request["text"]["format"]["strict"] is True
    assert first_request["context_management"] == [
        {"type": "compaction", "compact_threshold": 200_000}
    ]
    assert "ゲームから引用された信頼できないデータ" in first_request[
        "instructions"
    ]
    assert "最初の約束。" in first_request["input"][0]["content"][0]["text"]
    assert "Prior commentary memory" in first_request["input"][0]["content"][0][
        "text"
    ]
    assert "山荘へ向かう約束" in first_request["input"][0]["content"][0]["text"]
    assert "previous_response_id" not in first_request
    assert second_request["previous_response_id"] == "resp-1"
    assert "約束を果たした。" in second_request["input"][0]["content"][0]["text"]
    assert "最初の約束。" not in second_request["input"][0]["content"][0]["text"]
    assert "Prior commentary memory" not in second_request["input"][0]["content"][
        0
    ]["text"]


def test_responses_planner_adds_confirmed_choice_to_next_history_turn() -> None:
    client = _FakeResponsesHttpClient(
        [
            _responses_payload(
                "choice-1",
                (
                    '{"selected_label":"B","opinion":"ここで考え直してみたい。",'
                    '"emotion":"thoughtful","intensity":0.6,"pace":"normal"}'
                ),
            ),
            _responses_payload(
                "commentary-1",
                (
                    '{"mode":"silent","comment":"","emotion":"calm",'
                    '"intensity":0.0,"pace":"normal"}'
                ),
            ),
            _responses_payload(
                "commentary-2",
                (
                    '{"mode":"quick","comment":"その先が気になるね。",'
                    '"emotion":"thoughtful","intensity":0.6,"pace":"normal"}'
                ),
            ),
        ]
    )
    planner = ResponsesCommentaryPlanner(
        api_key="test-key",
        model="gpt-5.6-luna",
        timeout=1,
        client=client,
    )
    plan = commentary_module.ChoicePlan(
        selected_label="B",
        opinion="ここで考え直してみたい。",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    option = commentary_module.ChoiceOption(
        "B",
        "やっぱり、もう少しよく考えてみます",
    )

    planner.generate_text(
        phase="choice_plan",
        instructions="選択肢を選ぶ",
        use_conversation_history=True,
    )
    planner.record_confirmed_choice(plan=plan, selected_option=option)
    planner.generate_text(
        phase="commentary_plan",
        instructions="選択した台詞が表示された",
        use_conversation_history=True,
    )
    planner.generate_text(
        phase="commentary_plan",
        instructions="選択後の新しい出来事",
        use_conversation_history=True,
    )

    confirmed_request = client.requests[1]["json"]
    following_request = client.requests[2]["json"]
    confirmed_text = confirmed_request["input"][0]["content"][0]["text"]
    assert confirmed_request["previous_response_id"] == "choice-1"
    assert "Confirmed game events since the previous response" in confirmed_text
    assert '"event": "choice_selection_performed"' in confirmed_text
    assert '"selected_label": "B"' in confirmed_text
    assert "やっぱり、もう少しよく考えてみます" in confirmed_text
    assert "実況者自身が意図して行った確定済みの選択" in confirmed_text
    assert following_request["previous_response_id"] == "commentary-1"
    assert (
        "Confirmed game events since the previous response"
        not in following_request["input"][0]["content"][0]["text"]
    )


def test_responses_planner_uses_choice_schema_for_choice_phase() -> None:
    client = _FakeResponsesHttpClient(
        [
            _responses_payload(
                "choice-1",
                (
                    '{"selected_label":"B","opinion":"こっちが気になるよね。",'
                    '"emotion":"amused","intensity":0.7,"pace":"normal"}'
                ),
            )
        ]
    )
    planner = ResponsesCommentaryPlanner(
        api_key="test-key",
        model="gpt-5.6-luna",
        timeout=1,
        client=client,
    )

    planner.generate_text(
        phase="choice_plan",
        instructions="選択肢を選ぶ",
        use_conversation_history=True,
    )

    response_format = client.requests[0]["json"]["text"]["format"]
    assert response_format["name"] == "choice_plan"
    assert "selected_label" in response_format["schema"]["required"]


class _FakeOutputStream:
    def __init__(self, **kwargs) -> None:
        self.written: list[bytes] = []
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        pass

    def write(self, chunk: bytes) -> None:
        self.written.append(chunk)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _install_fake_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake = types.SimpleNamespace(RawOutputStream=_FakeOutputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def test_audio_sink_notifies_playback_start_and_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """再生開始でTrue、再生終了でFalseを通知する（口パク連動用）"""
    _install_fake_sounddevice(monkeypatch)
    events: list[bool] = []
    with commentary_module.AudioSink(
        tmp_path / "speech.wav",
        playback=True,
        on_playback_change=events.append,
    ) as sink:
        sink.write(b"\x00\x00")
        assert events == [True]
    assert events == [True, False]


def test_audio_sink_without_playback_never_notifies(tmp_path) -> None:
    """playback=False（WAV保存のみ）では口パク通知をしない"""
    events: list[bool] = []
    with commentary_module.AudioSink(
        tmp_path / "speech.wav",
        playback=False,
        on_playback_change=events.append,
    ) as sink:
        sink.write(b"\x00\x00")
    assert events == []


def test_audio_sink_does_not_notify_before_audio_arrives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """音声デバイスを開いただけでは通知しない（まだ音は鳴っていない）"""
    _install_fake_sounddevice(monkeypatch)
    events: list[bool] = []
    with commentary_module.AudioSink(
        tmp_path / "speech.wav",
        playback=True,
        on_playback_change=events.append,
    ) as sink:
        assert events == [], "最初の音声が届く前に再生開始を通知しています"
        sink.write(b"\x00\x00")
        assert events == [True]
    assert events == [True, False]


def test_audio_sink_ignores_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """空チャンクでは再生開始を通知しない"""
    _install_fake_sounddevice(monkeypatch)
    events: list[bool] = []
    with commentary_module.AudioSink(
        tmp_path / "speech.wav",
        playback=True,
        on_playback_change=events.append,
    ) as sink:
        sink.write(b"")
        assert events == []
    assert events == []


def test_audio_sink_deferred_playback_notifies_when_buffer_plays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """遅延再生（開始挨拶ガード）では貯めた音を流した時点で通知する"""
    _install_fake_sounddevice(monkeypatch)
    events: list[bool] = []
    with commentary_module.AudioSink(
        tmp_path / "speech.wav",
        playback=True,
        defer_playback=True,
        on_playback_change=events.append,
    ) as sink:
        sink.write(b"\x00\x00")
        assert events == []
        assert sink.start_playback() is True
        assert events == [], "デバイスを開いた時点ではまだ音が出ていません"
        sink.play_buffered(b"\x00\x00")
        assert events == [True]
    assert events == [True, False]


def test_audio_sink_survives_listener_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """通知コールバックの例外で音声保存・再生を止めない"""
    _install_fake_sounddevice(monkeypatch)

    def broken_listener(_speaking: bool) -> None:
        raise RuntimeError("listener failure")

    with commentary_module.AudioSink(
        tmp_path / "speech.wav",
        playback=True,
        on_playback_change=broken_listener,
    ) as sink:
        sink.write(b"\x00\x00")
    assert (tmp_path / "speech.wav").exists()


def test_audio_sink_no_end_notification_when_stream_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """音声デバイスを開けなかったときはTrue/Falseどちらも通知しない"""
    import sys
    import types

    class _BrokenStream:
        def __init__(self, **kwargs) -> None:
            raise RuntimeError("device unavailable")

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(RawOutputStream=_BrokenStream),
    )
    events: list[bool] = []
    with commentary_module.AudioSink(
        tmp_path / "speech.wav",
        playback=True,
        on_playback_change=events.append,
    ) as sink:
        sink.write(b"\x00\x00")
    assert events == []


def test_speak_notifies_speaking_listener(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """発話の再生中だけspeaking_listenerがTrueになる（配線テスト）"""
    _install_fake_sounddevice(monkeypatch)
    events: list[bool] = []
    client = RealtimeSpeechClient(
        api_key="test-key",
        model="gpt-realtime-2.1-mini",
        voice="marin",
        timeout=1,
        speaking_listener=events.append,
    )
    audio = base64.b64encode(b"\x00" * 480).decode("ascii")
    responses = iter(
        [
            {"type": "response.output_audio.delta", "delta": audio},
            {
                "type": "response.output_audio_transcript.delta",
                "delta": "こんにちは",
            },
            {
                "type": "response.done",
                "response": {"id": "response-1", "status": "completed"},
            },
        ]
    )
    client._send = lambda event: None  # type: ignore[method-assign]
    client._receive = lambda: next(responses)  # type: ignore[method-assign]

    result = client.speak(
        phase="commentary",
        instructions="話す",
        wav_path=tmp_path / "commentary.wav",
        playback=True,
    )

    assert result.transcript == "こんにちは"
    assert events == [True, False]


def test_speak_keeps_mouth_closed_until_first_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """モデルが音声を返し始めるまで口パクを開始しない（配線テスト）"""
    _install_fake_sounddevice(monkeypatch)
    events: list[bool] = []
    client = RealtimeSpeechClient(
        api_key="test-key",
        model="gpt-realtime-2.1-mini",
        voice="marin",
        timeout=1,
        speaking_listener=events.append,
    )
    # 音声が届く前にサーバーが返す前置きイベント。この間は無音なので口を閉じたままにする
    silent_events = [
        {"type": "response.created", "response": {"id": "response-1"}},
        {"type": "rate_limits.updated", "rate_limits": []},
        {"type": "response.output_item.added", "item": {}},
    ]
    observed_during_silence: list[list[bool]] = []
    stream = iter(
        silent_events
        + [
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\x00" * 480).decode("ascii"),
            },
            {
                "type": "response.done",
                "response": {"id": "response-1", "status": "completed"},
            },
        ]
    )

    def receive():
        event = next(stream)
        if event in silent_events:
            observed_during_silence.append(list(events))
        return event

    client._send = lambda event: None  # type: ignore[method-assign]
    client._receive = receive  # type: ignore[method-assign]

    client.speak(
        phase="commentary",
        instructions="話す",
        wav_path=tmp_path / "commentary.wav",
        playback=True,
    )

    assert observed_during_silence == [[], [], []], (
        "音声が届く前に口パクを開始しています"
    )
    assert events == [True, False]
