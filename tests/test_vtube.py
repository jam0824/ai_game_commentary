import threading
import time

from game_window_ocr import vtube
from game_window_ocr.vtube import (
    VTubeStudioController,
    resolve_emotion_mood,
)


class TestEmotionToMood:
    def test_thoughtful_maps_to_tense(self) -> None:
        """thoughtfulはtenseと同じ扱い（真剣）にする"""
        assert resolve_emotion_mood("thoughtful") == "tense"

    def test_calm_and_amused_stay_default_faced(self) -> None:
        """calmとamusedはデフォルト（表情なし）の状態を使う"""
        assert resolve_emotion_mood("calm") == "calm"
        assert resolve_emotion_mood("amused") == "amused"
        assert "calm" not in vtube.MOOD_EXPRESSIONS
        assert "amused" not in vtube.MOOD_EXPRESSIONS

    def test_unknown_emotion_falls_back_to_calm(self) -> None:
        """未知の感情名が来ても落ちずにcalmへフォールバックする"""
        assert resolve_emotion_mood("angry") == "calm"
        assert resolve_emotion_mood("") == "calm"

    def test_emotion_name_is_normalized(self) -> None:
        """大文字や前後空白があっても正しく解決する"""
        assert resolve_emotion_mood(" SURPRISED ") == "surprised"

    def test_all_commentary_emotions_resolve_to_defined_moods(self) -> None:
        """実況プランの全感情がムード定義済みの値へ解決される"""
        emotions = (
            "calm",
            "amused",
            "excited",
            "surprised",
            "tense",
            "sad",
            "thoughtful",
        )
        for emotion in emotions:
            assert resolve_emotion_mood(emotion) in vtube.MOODS

    def test_thoughtful_uses_serious_expression(self) -> None:
        """thoughtfulはtenseと同じ「真剣」表情になる"""
        mood = resolve_emotion_mood("thoughtful")
        assert vtube.MOOD_EXPRESSIONS[mood] == "真剣"


class TestControllerOffline:
    def test_set_emotion_and_speaking_without_connection(self) -> None:
        """未接続でもset_emotion/set_speakingは例外を出さず状態だけ更新する"""
        controller = VTubeStudioController()
        controller.set_emotion("excited")
        controller.set_speaking(True)
        assert controller.engine.mood == "excited"
        assert controller.state["is_speaking"] is True
        controller.set_speaking(False)
        assert controller.state["is_speaking"] is False
        controller.stop()

    def test_start_returns_false_when_connect_fails(self, capsys) -> None:
        """接続失敗時はFalseを返し、警告だけ出して実況を止めない"""

        async def failing_connect():
            raise RuntimeError("接続できません")

        controller = VTubeStudioController(connect=failing_connect)
        try:
            assert controller.start(wait=5.0) is False
        finally:
            controller.stop()
        captured = capsys.readouterr()
        assert "警告" in captured.err


class _FakeRequestFactory:
    """pyvts.vts_request相当の最小リクエストビルダー"""

    def requestSetMultiParameterValue(
        self,
        parameters,
        values,
        face_found=False,
    ):
        return {
            "messageType": "InjectParameterDataRequest",
            "data": {
                "faceFound": face_found,
                "parameterValues": [
                    {"id": name, "value": value}
                    for name, value in zip(parameters, values)
                ],
            },
        }

    def requestTrackingParameterList(self):
        return {"messageType": "InputParameterListRequest", "data": {}}

    def requestCustomParameter(
        self,
        name,
        min=0.0,
        max=1.0,
        default_value=0.0,
        info="",
    ):
        return {
            "messageType": "ParameterCreationRequest",
            "data": {"parameterName": name},
        }

    def BaseRequest(self, message_type, data=None):
        return {"messageType": message_type, "data": data or {}}


_FAKE_EXPRESSIONS = [
    {"name": "悲しみ", "file": "sad.exp3.json", "active": False},
    {"name": "真剣", "file": "serious.exp3.json", "active": False},
    {"name": "興奮", "file": "excited.exp3.json", "active": False},
    {"name": "驚き", "file": "surprised.exp3.json", "active": False},
]


class _FakeVts:
    def __init__(self) -> None:
        self.vts_request = _FakeRequestFactory()
        self.requests: list[dict] = []
        self.active_files: set[str] = set()
        self.closed = False

    async def request(self, payload):
        self.requests.append(payload)
        message_type = payload.get("messageType")
        if message_type == "InputParameterListRequest":
            return {
                "messageType": "InputParameterListResponse",
                "data": {
                    "defaultParameters": [
                        {"name": name} for name in vtube.FACE_PARAMS
                    ],
                    "customParameters": [],
                },
            }
        if message_type == "ParameterCreationRequest":
            return {"messageType": "ParameterCreationResponse", "data": {}}
        if message_type == "ExpressionStateRequest":
            return {
                "messageType": "ExpressionStateResponse",
                "data": {
                    "expressions": [
                        {
                            **expression,
                            "active": expression["file"] in self.active_files,
                        }
                        for expression in _FAKE_EXPRESSIONS
                    ]
                },
            }
        if message_type == "ExpressionActivationRequest":
            file = payload["data"]["expressionFile"]
            if payload["data"]["active"]:
                self.active_files.add(file)
            else:
                self.active_files.discard(file)
            return {"messageType": "ExpressionActivationResponse", "data": {}}
        return {"messageType": f"{message_type}Response", "data": {}}

    async def close(self):
        self.closed = True


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestControllerIntegration:
    """モジュール間の配線テスト（VTS接続はフェイク、内部接続は本物）"""

    def _start_controller(self) -> tuple[VTubeStudioController, _FakeVts]:
        fake = _FakeVts()

        async def connect():
            return fake

        controller = VTubeStudioController(connect=connect)
        assert controller.start(wait=5.0) is True
        return controller, fake

    def test_emotion_flows_to_expression_and_mouth_moves_while_speaking(
        self,
    ) -> None:
        """感情指定が表情リクエストへ流れ、発話中はMouthOpenが動く"""
        controller, fake = self._start_controller()
        try:
            assert _wait_until(
                lambda: any(
                    request.get("messageType") == "InjectParameterDataRequest"
                    for request in list(fake.requests)
                )
            ), "待機モーションのパラメータ送信が始まりません"

            controller.set_emotion("thoughtful")
            assert _wait_until(
                lambda: any(
                    request.get("messageType") == "ExpressionActivationRequest"
                    and request["data"].get("expressionFile")
                    == "serious.exp3.json"
                    and request["data"].get("active") is True
                    for request in list(fake.requests)
                )
            ), "thoughtfulで「真剣」表情がONになりません"

            marker = len(fake.requests)
            controller.set_speaking(True)

            def mouth_moved() -> bool:
                for request in list(fake.requests)[marker:]:
                    if (
                        request.get("messageType")
                        != "InjectParameterDataRequest"
                    ):
                        continue
                    for parameter in request["data"]["parameterValues"]:
                        if (
                            parameter["id"] == "MouthOpen"
                            and parameter["value"] > 0.3
                        ):
                            return True
                return False

            assert _wait_until(mouth_moved), "発話中に口が開きません"
            controller.set_speaking(False)
        finally:
            controller.stop()
        assert fake.closed, "停止時にVTS接続がクローズされません"

    def test_mouth_closes_immediately_after_speech_ends(self) -> None:
        """発話終了後に口が開いたまま取り残されない"""
        controller, fake = self._start_controller()
        try:
            controller.set_speaking(True)
            assert _wait_until(
                lambda: controller.state["mouth_open"] > 0.3
            ), "発話中に口が開きません"

            controller.set_speaking(False)
            assert _wait_until(
                lambda: controller.state["mouth_open"] == 0.0,
                timeout=0.5,
            ), "発話終了後も口が開いたままです"

            # 口パクコルーチンが再び開けにこないことを確かめる
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                assert controller.state["mouth_open"] == 0.0
                time.sleep(0.02)
        finally:
            controller.stop()

    def test_stop_deactivates_active_expression(self) -> None:
        """停止時にONだった表情をOFFへ戻す"""
        controller, fake = self._start_controller()
        controller.set_emotion("surprised")
        assert _wait_until(
            lambda: any(
                request.get("messageType") == "ExpressionActivationRequest"
                and request["data"].get("expressionFile")
                == "surprised.exp3.json"
                and request["data"].get("active") is True
                for request in list(fake.requests)
            )
        )
        controller.stop()
        assert any(
            request.get("messageType") == "ExpressionActivationRequest"
            and request["data"].get("expressionFile") == "surprised.exp3.json"
            and request["data"].get("active") is False
            for request in list(fake.requests)
        ), "停止時に表情がリセットされません"
        assert fake.closed

    def test_threads_are_daemon(self) -> None:
        """連携スレッドはデーモンで、実況本体の終了を妨げない"""
        controller, _fake = self._start_controller()
        try:
            thread = controller._thread
            assert thread is not None
            assert thread.daemon is True
        finally:
            controller.stop()
