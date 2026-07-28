import asyncio
import random
import threading
import time

import pytest

from game_window_ocr import vtube
from game_window_ocr.vtube import (
    VTubeStudioController,
    resolve_emotion_mood,
    rest_look_target,
    pick_look_target,
    turn_tilt,
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


def _glance_eye_trace(
    aim_x: float,
    seconds: float = 3.0,
    dt: float = 1.0 / vtube.FPS,
    speed: float = 1.0,
) -> list[tuple[float, float, float]]:
    """正面からaim_xへ振り向く間の (時刻, 首X, 目玉X) を再現する"""
    import math

    trace = []
    cur = 0.0
    t = 0.0
    while t <= seconds:
        trace.append((t, cur, vtube.eye_look_offset(cur, aim_x - cur)))
        follow = 1.0 - math.exp(-dt * vtube.LOOK_FOLLOW_RATE * speed)
        cur += (aim_x - cur) * follow
        t += dt
    return trace


class TestEyeMotion:
    def test_eyes_lead_the_head(self) -> None:
        """振り向き始めは首より先に目が動く"""
        trace = _glance_eye_trace(vtube.SCREEN_LOOK_X)
        _t, head, eye = trace[0]
        assert head == 0.0
        assert eye <= -0.9, f"目の先行が弱すぎます: {eye}"

    def test_eyes_do_not_stay_pinned_at_the_limit(self) -> None:
        """目が振り切れたまま固まらない（白目がちに見えないように）"""
        trace = _glance_eye_trace(vtube.SCREEN_LOOK_X)
        pinned = [t for t, _head, eye in trace if abs(eye) >= 1.0]
        assert pinned, "先行の振り切れ自体は残す"
        assert max(pinned) <= 0.2, f"振り切れが長すぎます: {max(pinned):.2f}秒"

    def test_eyes_recenter_after_the_head_arrives(self) -> None:
        """首が到着したら目は中央側へ戻る（首も目も横を向いたままにしない）"""
        trace = _glance_eye_trace(vtube.SCREEN_LOOK_X)
        start_eye = trace[0][2]
        settled_eye = trace[-1][2]
        assert abs(settled_eye) < abs(start_eye)
        assert abs(settled_eye) <= 0.55, f"定常のズレが強すぎます: {settled_eye}"

    def test_eye_offset_stays_in_range(self) -> None:
        """目線はVTSの定格(-1〜1)を超えない"""
        for aim in (vtube.SCREEN_LOOK_X, -vtube.SCREEN_LOOK_X, 0.0, -120.0):
            for _t, _head, eye in _glance_eye_trace(aim, seconds=1.0):
                assert -1.0 <= eye <= 1.0

    def test_eye_and_head_look_the_same_way_vertically(self) -> None:
        """上下の向きは首と目で一致させる（LOOK_Y_SIGNを反転しても矛盾しない）"""
        head_y, eye_y = vtube.vertical_gaze(head=10.0, residual=0.0)
        assert head_y * eye_y > 0.0, "首と目が上下逆を向いています"

    def test_eyes_hold_their_point_against_the_sway(self) -> None:
        """首が揺れても視線は同じ点に留まる（揺れと逆方向へ補正する）"""
        still = vtube.eye_look_offset(head=0.0, residual=0.0, sway=0.0)
        swayed = vtube.eye_look_offset(head=0.0, residual=0.0, sway=4.0)
        assert swayed < still, "首の揺れに目線が逆補正されていません"

    def test_sway_compensation_stays_subtle(self) -> None:
        """揺れの補正は控えめ（揺れだけで目が泳がない）"""
        widest = max(p.sway_x + p.sub_x for p in vtube.MOODS.values())
        drift = abs(vtube.eye_look_offset(head=0.0, residual=0.0, sway=widest))
        assert drift <= 0.1, f"揺れによる目線のブレが大きすぎます: {drift}"


class TestEyeOpen:
    def test_every_mood_can_widen_the_eyes_when_surprised(self) -> None:
        """どのムードから驚いても目の見開きが見た目に出る"""
        for name, params in vtube.MOODS.items():
            calm_eyes = vtube.eye_open_value(1.0, params, 0.0)
            wide_eyes = vtube.eye_open_value(
                1.0, params, vtube.surprise_eye_bonus(params, 1.0)
            )
            assert wide_eyes - calm_eyes >= 0.10, (
                f"{name}で驚いても目が見開きません: "
                f"{calm_eyes:.2f} -> {wide_eyes:.2f}"
            )

    def test_normal_eyes_leave_room_to_widen(self) -> None:
        """平常時の目は満開手前にして、見開く余地を残す"""
        for name, params in vtube.MOODS.items():
            assert params.eye_open_base < 1.0, f"{name}に見開く余地がありません"

    def test_surprise_opens_the_eyes_fully(self) -> None:
        """驚きのピークでは目一杯まで見開く"""
        for params in vtube.MOODS.values():
            peak = vtube.eye_open_value(
                1.0, params, vtube.surprise_eye_bonus(params, 1.0)
            )
            assert peak == pytest.approx(1.0)

    def test_blink_closes_the_eyes_even_while_surprised(self) -> None:
        """驚いている最中のまばたきでも目は閉じきる"""
        params = vtube.MOODS["calm"]
        bonus = vtube.surprise_eye_bonus(params, 1.0)
        assert vtube.eye_open_value(0.0, params, bonus) == 0.0

    def test_eyes_return_to_normal_after_the_burst(self) -> None:
        """驚きが収まれば元の開きへ戻る"""
        params = vtube.MOODS["calm"]
        assert vtube.surprise_eye_bonus(params, 0.0) == 0.0
        assert vtube.eye_open_value(1.0, params, 0.0) == params.eye_open_base


class TestMoodReaction:
    """感情が変わったとき、次の待機を待たずに動きへ反映されるか"""

    def test_wait_is_interrupted_by_a_mood_change(self) -> None:
        """待機中に感情が変わったら、待たずに次の動きへ移る"""
        controller = VTubeStudioController()

        async def scenario() -> tuple[bool, float]:
            started = time.perf_counter()
            task = asyncio.create_task(controller.wait_for_mood_change(30.0))
            await asyncio.sleep(0.05)
            controller.engine.set_mood("surprised")
            changed = await asyncio.wait_for(task, timeout=2.0)
            return changed, time.perf_counter() - started

        changed, elapsed = asyncio.run(scenario())
        assert changed is True
        assert elapsed < 1.0, f"感情の変化に{elapsed:.2f}秒かかっています"

    def test_wait_uses_the_full_time_when_the_mood_stays(self) -> None:
        """感情が変わらなければ指定どおり待つ"""
        controller = VTubeStudioController()

        async def scenario() -> tuple[bool, float]:
            started = time.perf_counter()
            changed = await controller.wait_for_mood_change(0.5)
            return changed, time.perf_counter() - started

        changed, elapsed = asyncio.run(scenario())
        assert changed is False
        assert elapsed >= 0.5


class TestBlinkTiming:
    def test_every_phase_lasts_at_least_two_frames(self) -> None:
        """30fpsで送るので、まばたきの各段階は最低2フレーム持たせる

        速いムードで1フレームを切ると、まばたきが抜けたりチラついたりする。
        """
        floor = 2.0 / vtube.FPS
        for name, params in vtube.MOODS.items():
            for seconds in (
                vtube.BLINK_CLOSE,
                vtube.BLINK_HALF,
                vtube.BLINK_OPEN,
            ):
                phase = vtube.blink_phase(seconds, params.speed)
                assert phase >= floor - 1e-9, (
                    f"{name}のまばたきが速すぎます: {phase * vtube.FPS:.1f}フレーム"
                )

    def test_slow_moods_still_blink_slower(self) -> None:
        """下限を入れても、重いムードのゆっくりしたまばたきは残る"""
        slow = vtube.blink_phase(vtube.BLINK_OPEN, vtube.MOODS["sad"].speed)
        quick = vtube.blink_phase(vtube.BLINK_OPEN, vtube.MOODS["excited"].speed)
        assert slow > quick


class _AlwaysGlanceRng:
    """必ずゲーム画面のチラ見を選ぶrng（揺らぎは中央値に固定）"""

    def random(self) -> float:
        return 0.0

    def uniform(self, low: float, high: float) -> float:
        return (low + high) / 2.0


class _NeverGlanceRng(_AlwaysGlanceRng):
    """絶対にチラ見を選ばないrng（きょろきょろ側の検証用）"""

    def random(self) -> float:
        return 1.0

    def uniform(self, low: float, high: float) -> float:
        return high


class TestLookMotion:
    def test_screen_glance_turns_head_far_enough_to_read(self) -> None:
        """ゲーム画面のチラ見は右下の小さい表示でも分かる大きさで首を振る"""
        rng = _AlwaysGlanceRng()
        for mood, params in vtube.MOODS.items():
            x, y, _hold = pick_look_target(params, rng)
            assert x <= -20.0, f"{mood}のチラ見が小さすぎます: x={x}"
            assert y >= 6.0, f"{mood}のチラ見が上を向いていません: y={y}"

    def test_screen_glance_stays_inside_vts_range(self) -> None:
        """振り幅を大きくしてもVTSのFaceAngle定格(±30)を超えない"""
        rng = _AlwaysGlanceRng()
        params = vtube.MOODS["calm"]
        for _ in range(200):
            x, y, _hold = pick_look_target(params, random)
            assert -30.0 < x < 30.0
            assert -30.0 < y < 30.0
        assert abs(pick_look_target(params, rng)[0]) < 30.0

    def test_head_tilts_toward_the_direction_of_the_turn(self) -> None:
        """振り向いた側へ首も傾く（首の角度だけの平行移動にしない）"""
        assert turn_tilt(0.0) == 0.0
        left = turn_tilt(vtube.SCREEN_LOOK_X)
        assert abs(left) >= 4.0, f"振り向きの傾きが小さすぎます: {left}"
        assert left < 0.0, "左を向いたら左側へ傾く"
        assert turn_tilt(-vtube.SCREEN_LOOK_X) == -left

    def test_head_tilt_is_capped(self) -> None:
        """狙いが極端でも首が倒れすぎない"""
        assert abs(turn_tilt(-120.0)) <= vtube.LOOK_TILT_MAX

    def test_idle_gaze_does_not_stay_turned_away(self) -> None:
        """チラ見が大きくなっても、待機位置まで横を向いたままにはしない"""
        rng = _AlwaysGlanceRng()
        for mood, params in vtube.MOODS.items():
            x, _y = rest_look_target(params, rng)
            assert abs(x) <= 8.0, f"{mood}の待機位置が横を向きすぎです: x={x}"

    def test_random_glance_respects_mood_range(self) -> None:
        """チラ見以外のきょろきょろはムードの振れ幅に収まる"""
        rng = _NeverGlanceRng()
        for params in vtube.MOODS.values():
            x, y, _hold = pick_look_target(params, rng)
            assert abs(x) <= params.look_range_x
            assert abs(y) <= params.look_range_y


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

    def test_screen_glance_reaches_face_angle_and_tilt(self) -> None:
        """視線の狙いがFaceAngleXまで届き、同時に首の傾き(Z)も連動する"""
        controller, fake = self._start_controller()
        try:
            controller.state["look_x"] = vtube.SCREEN_LOOK_X
            controller.state["look_y"] = vtube.SCREEN_LOOK_Y

            def sent(name: str):
                for request in reversed(list(fake.requests)):
                    if request.get("messageType") != "InjectParameterDataRequest":
                        continue
                    for parameter in request["data"]["parameterValues"]:
                        if parameter["id"] == name:
                            return parameter["value"]
                return None

            assert _wait_until(
                lambda: (sent("FaceAngleX") or 0.0) <= -15.0
            ), "チラ見の狙いがFaceAngleXまで届いていません"
            # calmの揺れ(sway_z+sub_z)は最大3.1程度。それを超える傾きは振り向き由来
            assert _wait_until(
                lambda: (sent("FaceAngleZ") or 0.0) <= -3.5
            ), "振り向きに首の傾きが連動していません"
        finally:
            controller.stop()

    def test_look_override_pins_and_releases_the_gaze(self) -> None:
        """視線を固定でき、解除すれば自動のきょろきょろへ戻る"""
        controller, fake = self._start_controller()
        try:
            controller.set_look_override((vtube.SCREEN_LOOK_X, vtube.SCREEN_LOOK_Y))

            def face_angle_x():
                for request in reversed(list(fake.requests)):
                    if request.get("messageType") != "InjectParameterDataRequest":
                        continue
                    for parameter in request["data"]["parameterValues"]:
                        if parameter["id"] == "FaceAngleX":
                            return parameter["value"]
                return None

            assert _wait_until(
                lambda: (face_angle_x() or 0.0) <= -15.0
            ), "視線の固定がFaceAngleXへ反映されません"

            # 固定中はきょろきょろの指示で上書きされない
            controller.state["look_x"] = 30.0
            time.sleep(0.3)
            assert (face_angle_x() or 0.0) <= -15.0, "固定中の視線が上書きされました"

            controller.set_look_override(None)
            assert _wait_until(
                lambda: (face_angle_x() or 0.0) > 0.0
            ), "固定を解除しても視線が戻りません"
        finally:
            controller.stop()

    def test_gaze_reacts_to_a_new_emotion_without_waiting(self) -> None:
        """感情が変わったら、次のきょろきょろを待たずに視線を動かし直す

        calmの待機は最長10秒あるので、驚いた瞬間に画面を確認しにいく動きが
        そこまで遅れないことを確かめる。
        """
        controller, fake = self._start_controller()
        try:
            # 待機モーションが動き出す＝各ディレクタが待機に入ってから仕掛ける
            assert _wait_until(
                lambda: any(
                    request.get("messageType") == "InjectParameterDataRequest"
                    for request in list(fake.requests)
                )
            )
            sentinel = 999.0
            controller.state["look_x"] = sentinel
            controller.set_emotion("surprised")
            assert _wait_until(
                lambda: controller.state["look_x"] != sentinel,
                timeout=2.0,
            ), "感情が変わっても視線の狙いが更新されません"
        finally:
            controller.stop()

    def test_threads_are_daemon(self) -> None:
        """連携スレッドはデーモンで、実況本体の終了を妨げない"""
        controller, _fake = self._start_controller()
        try:
            thread = controller._thread
            assert thread is not None
            assert thread.daemon is True
        finally:
            controller.stop()
