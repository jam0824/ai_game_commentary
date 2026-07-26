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
