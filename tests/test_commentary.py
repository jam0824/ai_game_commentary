import json

from game_window_ocr.commentary import RealtimeSpeechClient


def _client() -> RealtimeSpeechClient:
    return RealtimeSpeechClient(
        api_key="test-key",
        model="gpt-realtime-2.1-mini",
        voice="marin",
        timeout=1,
    )


def test_record_game_text_adds_turn_to_default_conversation() -> None:
    client = _client()
    sent = []
    client._send = sent.append  # type: ignore[method-assign]

    client.record_game_text(turn_number=2, text="前半で\n後半。")

    event = sent[0]
    assert event["type"] == "conversation.item.create"
    payload = json.loads(event["item"]["content"][0]["text"])
    assert payload == {
        "event": "game_screen_ocr",
        "turn": 2,
        "game_text": "前半で後半。",
    }


def test_commentary_plan_uses_default_conversation() -> None:
    client = _client()
    sent = []
    client._send = sent.append  # type: ignore[method-assign]
    client._receive = lambda: {  # type: ignore[method-assign]
        "type": "response.done",
        "response": {"id": "response-1", "status": "completed"},
    }

    client.generate_text(
        phase="commentary_plan",
        instructions="短く感想を話す",
        use_conversation_history=True,
    )

    response = sent[0]["response"]
    assert response["output_modalities"] == ["text"]
    assert "conversation" not in response
    assert "input" not in response


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
        use_conversation_history=False,
    )

    response = sent[0]["response"]
    assert response["conversation"] == "none"
    assert response["input"] == []


def test_generate_text_uses_output_text_done_when_delta_is_missing() -> None:
    client = _client()
    sent = []
    events = iter(
        [
            {
                "type": "response.output_text.done",
                "text": '{"comment":"怪しい","emotion":"tense"}',
            },
            {
                "type": "response.done",
                "response": {"id": "response-1", "status": "completed"},
            },
        ]
    )
    client._send = sent.append  # type: ignore[method-assign]
    client._receive = lambda: next(events)  # type: ignore[method-assign]

    result = client.generate_text(
        phase="commentary_plan",
        instructions="JSONを出す",
        use_conversation_history=True,
    )

    assert result.text == '{"comment":"怪しい","emotion":"tense"}'
