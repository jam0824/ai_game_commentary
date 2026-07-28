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


class TestFaceExpressionParams:
    """眉と口角(VTSのBrows / MouthSmile)のムード連動

    どちらも 0.5 が中立で、0 で眉が下がりきる・口角がへの字になる。
    送らないと 0 が使われてしまうため、常に値を出し続ける必要がある。
    """

    def test_brows_and_smile_are_sent_every_frame(self) -> None:
        """眉と口角は毎フレーム送る対象に入っている"""
        assert "Brows" in vtube.FACE_PARAMS
        assert "MouthSmile" in vtube.FACE_PARAMS

    def test_values_stay_in_range(self) -> None:
        """VTSの定格(0〜1)に収まる"""
        for name, params in vtube.MOODS.items():
            assert 0.0 <= params.brows <= 1.0, name
            assert 0.0 <= params.mouth_smile <= 1.0, name

    def test_no_mood_leaves_the_face_at_the_default_zero(self) -> None:
        """どのムードも中立(0.5)から極端に外れた顔で固定しない"""
        for name, params in vtube.MOODS.items():
            assert abs(params.brows - vtube.FACE_NEUTRAL) <= 0.4, name
            assert abs(params.mouth_smile - vtube.FACE_NEUTRAL) <= 0.4, name

    def test_moods_differ_in_the_face(self) -> None:
        """楽しいときは口角と眉が上がり、沈んだときは下がる"""
        calm = vtube.MOODS["calm"]
        assert vtube.MOODS["amused"].mouth_smile > calm.mouth_smile
        assert vtube.MOODS["excited"].brows > calm.brows
        assert vtube.MOODS["surprised"].brows > calm.brows
        assert vtube.MOODS["sad"].mouth_smile < calm.mouth_smile
        assert vtube.MOODS["tense"].brows < calm.brows

    def test_calm_and_amused_are_distinguishable(self) -> None:
        """表情ファイルを持たないcalmとamusedが、顔つきで区別できる"""
        calm = vtube.MOODS["calm"]
        amused = vtube.MOODS["amused"]
        difference = abs(calm.mouth_smile - amused.mouth_smile) + abs(
            calm.brows - amused.brows
        )
        assert difference >= 0.15


class TestIdleWaveform:
    """待機モーションの波形（ムードごとの個性）"""

    def test_periods_are_positive(self) -> None:
        """周期は正の値（0除算やゼロ周期にしない）"""
        for name, params in vtube.MOODS.items():
            assert params.period_x > 0.0, name
            assert params.period_y > 0.0, name
            assert params.period_z > 0.0, name

    def test_moods_are_not_the_same_dance_at_different_speeds(self) -> None:
        """ムードごとに軸の周期比が違う（速度違いの同じ振り付けにしない）"""
        ratios = {
            name: (
                round(params.period_y / params.period_x, 3),
                round(params.period_z / params.period_x, 3),
            )
            for name, params in vtube.MOODS.items()
        }
        assert len(set(ratios.values())) == len(ratios), ratios

    def test_period_changes_the_waveform(self) -> None:
        """周期を変えると波形が変わる（振幅だけの違いにならない）"""
        base = vtube.MOODS["calm"]
        faster = base.replace(period_x=base.period_x * 0.5)
        same_phase = 2.0
        assert vtube.idle_angles(same_phase, base)[0] != pytest.approx(
            vtube.idle_angles(same_phase, faster)[0]
        )

    def test_waveform_stays_continuous_while_the_mood_blends(self) -> None:
        """ムード切り替えで周期が変わっても、角度が飛ばない"""
        engine = vtube.MoodEngine("sad")
        engine.update(1.0)
        engine.set_mood("excited")
        previous = engine.angles()[:3]
        for _ in range(int(3 * vtube.FPS)):
            engine.update(1 / vtube.FPS)
            current = engine.angles()[:3]
            for before, after in zip(previous, current):
                assert abs(after - before) < 2.0, "波形が不連続に飛んでいます"
            previous = current


class TestGestures:
    """単発の仕草（うなずき・首かしげ・首振りなど）"""

    def test_gesture_starts_and_ends_at_zero(self) -> None:
        """仕草は始点と終点で0に戻る（待機モーションへ滑らかに戻す）"""
        for name, gesture in vtube.GESTURES.items():
            assert vtube.gesture_offset(gesture, 0.0) == 0.0, name
            assert vtube.gesture_offset(gesture, 1.0) == 0.0, name

    def test_gesture_never_exceeds_its_amplitude(self) -> None:
        """振幅を超えて首が飛ばない"""
        for name, gesture in vtube.GESTURES.items():
            for step in range(101):
                offset = vtube.gesture_offset(gesture, step / 100)
                assert abs(offset) <= abs(gesture.amp) + 1e-9, name

    def test_oscillating_gesture_swings_both_ways(self) -> None:
        """うなずき・首振りは往復する"""
        for name in ("nod", "shake"):
            gesture = vtube.GESTURES[name]
            values = [vtube.gesture_offset(gesture, s / 100) for s in range(101)]
            assert min(values) < -0.5 and max(values) > 0.5, name

    def test_gesture_stands_out_from_the_idle_sway(self) -> None:
        """仕草は待機の揺れよりはっきり大きい

        揺れに埋もれる振幅だと「何かしたのか分からない」動きになる。
        """
        calm = vtube.MOODS["calm"]
        sway = {
            "x": calm.sway_x + calm.sub_x,
            "y": calm.sway_y + calm.sub_y,
            "z": calm.sway_z + calm.sub_z,
        }
        for name, gesture in vtube.GESTURES.items():
            peak = max(
                abs(vtube.gesture_offset(gesture, s / 500)) for s in range(501)
            )
            assert peak >= sway[gesture.axis] * 2.0, (
                f"{name}が待機の揺れ({sway[gesture.axis]:.1f}度)に埋もれます: "
                f"{peak:.1f}度"
            )

    def test_oscillating_gesture_reaches_its_amplitude(self) -> None:
        """往復の山が削れず、指定した振幅まで振れる"""
        for name in ("nod", "shake"):
            gesture = vtube.GESTURES[name]
            peak = max(
                abs(vtube.gesture_offset(gesture, s / 500)) for s in range(501)
            )
            assert peak >= abs(gesture.amp) * 0.95, name

    def test_every_swing_is_visible(self) -> None:
        """往復のどの振りも見える大きさになる（最初と最後だけ削られない）"""
        for name, expected in (("nod", 3), ("shake", 5)):
            gesture = vtube.GESTURES[name]
            values = [vtube.gesture_offset(gesture, s / 500) for s in range(501)]
            big = 0
            for index in range(1, len(values) - 1):
                if (
                    abs(values[index]) >= abs(gesture.amp) * 0.8
                    and abs(values[index]) >= abs(values[index - 1])
                    and abs(values[index]) > abs(values[index + 1])
                ):
                    big += 1
            assert big >= expected, f"{name}の振りが{big}回しか見えません"

    def test_held_gesture_reaches_and_keeps_its_pose(self) -> None:
        """首かしげ・前のめりは姿勢を作ってしばらく保つ"""
        gesture = vtube.GESTURES["tilt"]
        middle = vtube.gesture_offset(gesture, 0.5)
        assert middle == pytest.approx(gesture.amp, abs=0.01)

    def test_every_mood_has_gestures(self) -> None:
        """全ムードに仕草が割り当てられている"""
        assert set(vtube.MOOD_GESTURES) == set(vtube.MOODS)
        for mood, weights in vtube.MOOD_GESTURES.items():
            assert weights, mood
            for name in weights:
                assert name in vtube.GESTURES, f"{mood}に未定義の仕草: {name}"

    def test_moods_use_different_gestures(self) -> None:
        """ムードごとに出る仕草が違う（沈んだときにうなずかない等）"""
        assert vtube.MOOD_GESTURES["sad"] != vtube.MOOD_GESTURES["excited"]
        assert "droop" in vtube.MOOD_GESTURES["sad"]
        assert "nod" in vtube.MOOD_GESTURES["excited"]

    def test_pick_gesture_respects_weights(self) -> None:
        """重みの大きい仕草がよく出る"""
        counts: dict[str, int] = {}
        for _ in range(2000):
            name = vtube.pick_gesture("excited", random)
            counts[name] = counts.get(name, 0) + 1
        assert set(counts) <= set(vtube.MOOD_GESTURES["excited"])
        assert counts["nod"] > counts["shake"], counts

    def test_engine_plays_a_gesture_and_returns_to_idle(self) -> None:
        """エンジンに仕草を積むと角度が動き、終われば待機へ戻る"""
        engine = vtube.MoodEngine("calm")
        engine.start_gesture("tilt")
        peak = 0.0
        for _ in range(int(vtube.GESTURES["tilt"].seconds * vtube.FPS)):
            engine.update(1 / vtube.FPS)
            peak = max(peak, abs(engine.gesture_angles()[2]))
        assert peak > 4.0, f"首かしげが小さすぎます: {peak}"
        engine.update(1 / vtube.FPS)
        assert engine.gesture_angles() == (0.0, 0.0, 0.0)

    def test_unknown_gesture_is_ignored(self) -> None:
        """未知の仕草名でも落ちない"""
        engine = vtube.MoodEngine("calm")
        engine.start_gesture("moonwalk")
        assert engine.gesture_angles() == (0.0, 0.0, 0.0)


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
            x, y, _hold = vtube.plan_gaze("screen", params, rng)[0]
            assert x <= -20.0, f"{mood}のチラ見が小さすぎます: x={x}"
            assert y >= 6.0, f"{mood}のチラ見が上を向いていません: y={y}"

    def test_screen_glance_stays_inside_vts_range(self) -> None:
        """振り幅を大きくしてもVTSのFaceAngle定格(±30)を超えない"""
        for pattern in vtube.GAZE_PATTERNS:
            for params in vtube.MOODS.values():
                for _ in range(50):
                    for x, y, hold in vtube.plan_gaze(pattern, params, random):
                        assert -30.0 < x < 30.0, pattern
                        assert -30.0 < y < 30.0, pattern
                        assert hold > 0.0, pattern

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
            x, y, _hold = vtube.plan_gaze("wander", params, rng)[0]
            assert abs(x) <= params.look_range_x
            assert abs(y) <= params.look_range_y


class TestMoodMix:
    """感情の強度（表情のON/OFFではなく連続値で送るための配合）"""

    def test_mix_covers_every_mood(self) -> None:
        """全ムード分の強度を持つ"""
        engine = vtube.MoodEngine("calm")
        assert set(engine.mix) == set(vtube.MOODS)

    def test_mix_eases_toward_the_current_mood(self) -> None:
        """今のムードの強度が上がり、他は下がる"""
        engine = vtube.MoodEngine("calm")
        engine.set_mood("sad")
        for _ in range(int(3 * vtube.FPS)):
            engine.update(1 / vtube.FPS)
        assert engine.mix["sad"] > 0.9
        assert engine.mix["calm"] < 0.1

    def test_mix_changes_continuously(self) -> None:
        """切り替えの瞬間に強度が飛ばない（表情がパチっと変わらない）"""
        engine = vtube.MoodEngine("calm")
        engine.set_mood("excited")
        previous = dict(engine.mix)
        for _ in range(int(2 * vtube.FPS)):
            engine.update(1 / vtube.FPS)
            for name, value in engine.mix.items():
                assert abs(value - previous[name]) < 0.2
            previous = dict(engine.mix)

    def test_surprise_spikes_immediately(self) -> None:
        """驚きは補間を待たずに立ち上がる（一瞬の演出なので）"""
        engine = vtube.MoodEngine("calm")
        engine.set_mood("surprised")
        engine.update(0.1)
        assert engine.mood_param_values()["surprised"] > 0.5

    def test_only_moods_with_a_parameter_are_sent(self) -> None:
        """パラメータを持つムードだけ強度を出す（calm/amusedは素の顔）"""
        engine = vtube.MoodEngine("calm")
        assert set(engine.mood_param_values()) == set(vtube.MOOD_PARAMS)
        assert "calm" not in vtube.MOOD_PARAMS


class TestContinuousMoodParams:
    """感情パラメータが割り当て済みかどうかで動作が切り替わるか"""

    def _start(self, assigned: bool):
        fake = _FakeVts(mood_params_assigned=assigned)

        async def connect():
            return fake

        controller = VTubeStudioController(connect=connect)
        assert controller.start(wait=5.0) is True
        return controller, fake

    def test_creates_the_custom_parameters(self) -> None:
        """起動時に感情用のカスタム入力パラメータを作る"""
        controller, fake = self._start(assigned=False)
        try:
            created = {
                request["data"]["parameterName"]
                for request in list(fake.requests)
                if request.get("messageType") == "ParameterCreationRequest"
            }
            assert set(vtube.MOOD_PARAMS.values()) <= created
        finally:
            controller.stop()

    def test_sends_intensity_when_assigned(self) -> None:
        """割り当て済みなら強度を毎フレーム送る"""
        controller, fake = self._start(assigned=True)
        try:
            controller.set_emotion("sad")

            def sent(name: str):
                for request in reversed(list(fake.requests)):
                    if request.get("messageType") != "InjectParameterDataRequest":
                        continue
                    for parameter in request["data"]["parameterValues"]:
                        if parameter["id"] == name:
                            return parameter["value"]
                return None

            assert _wait_until(
                lambda: (sent(vtube.MOOD_PARAMS["sad"]) or 0.0) > 0.8
            ), "悲しみの強度が送られていません"
        finally:
            controller.stop()

    def test_expression_file_is_not_used_when_assigned(self) -> None:
        """強度で表現できるムードは、表情ファイルを二重に出さない"""
        controller, fake = self._start(assigned=True)
        try:
            controller.set_emotion("sad")
            time.sleep(0.6)
            activated = [
                request
                for request in list(fake.requests)
                if request.get("messageType") == "ExpressionActivationRequest"
                and request["data"].get("active") is True
            ]
            assert not activated, "強度とファイルの両方で表情を出しています"
        finally:
            controller.stop()

    def test_falls_back_to_expressions_when_not_assigned(self) -> None:
        """未割り当てなら従来どおり表情ファイルを使う"""
        controller, fake = self._start(assigned=False)
        try:
            controller.set_emotion("sad")
            assert _wait_until(
                lambda: any(
                    request.get("messageType") == "ExpressionActivationRequest"
                    and request["data"].get("expressionFile") == "sad.exp3.json"
                    and request["data"].get("active") is True
                    for request in list(fake.requests)
                )
            ), "フォールバックの表情が出ていません"
        finally:
            controller.stop()


class TestGazePatterns:
    """名前のついた視線パターン（二度見・伏し目・ガン見など）"""

    def test_every_mood_has_gaze_patterns(self) -> None:
        """全ムードに視線パターンが割り当てられている"""
        assert set(vtube.MOOD_GAZE) == set(vtube.MOODS)
        for mood, weights in vtube.MOOD_GAZE.items():
            assert weights, mood
            for name in weights:
                assert name in vtube.GAZE_PATTERNS, f"{mood}に未定義: {name}"

    def test_double_take_looks_back_at_the_screen(self) -> None:
        """二度見は 画面→正面→もう一度画面 の順に動く"""
        steps = vtube.plan_gaze("double_take", vtube.MOODS["surprised"], random)
        assert len(steps) >= 3
        assert steps[0][0] < -15.0
        assert abs(steps[1][0]) < 6.0, "いったん視線を戻さないと二度見にならない"
        assert steps[-1][0] < -15.0

    def test_downcast_gaze_looks_down(self) -> None:
        """伏し目は下を向く"""
        x, y, _hold = vtube.plan_gaze("down", vtube.MOODS["sad"], random)[0]
        assert y < -3.0
        assert abs(x) < 6.0

    def test_stare_holds_longer_than_a_glance(self) -> None:
        """ガン見はチラ見より長く画面に留まる"""
        params = vtube.MOODS["tense"]
        rng = _AlwaysGlanceRng()
        stare = vtube.plan_gaze("stare", params, rng)[0]
        glance = vtube.plan_gaze("screen", params, rng)[0]
        assert stare[2] > glance[2]
        assert stare[0] < -15.0

    def test_audience_gaze_faces_the_camera(self) -> None:
        """カメラ目線は正面を向く"""
        x, y, _hold = vtube.plan_gaze("audience", vtube.MOODS["amused"], random)[0]
        assert abs(x) < 4.0 and abs(y) < 4.0

    def test_drifting_gaze_moves_in_several_steps(self) -> None:
        """考え中の泳ぐ視線は何段階かに分けて動く"""
        steps = vtube.plan_gaze("drift", vtube.MOODS["tense"], random)
        assert len(steps) >= 2
        assert len({(x, y) for x, y, _ in steps}) >= 2

    def test_mood_picks_its_own_pattern(self) -> None:
        """ムードごとに出やすい視線パターンが違う"""
        counts: dict[str, int] = {}
        for _ in range(2000):
            name = vtube.pick_gaze_pattern("tense", random)
            counts[name] = counts.get(name, 0) + 1
        assert counts.get("stare", 0) > counts.get("wander", 0), counts
        assert "down" in vtube.MOOD_GAZE["sad"]
        assert "double_take" in vtube.MOOD_GAZE["surprised"]

    def test_unknown_pattern_falls_back_to_wander(self) -> None:
        """未知のパターン名でも視線が止まらない"""
        steps = vtube.plan_gaze("moonwalk", vtube.MOODS["calm"], random)
        assert steps


class TestSaccade:
    """マイクロサッカード（視線の微細な揺れ）"""

    def test_offset_is_small(self) -> None:
        """目線が泳がない程度の微小な揺れに留める"""
        saccade = vtube.Saccade()
        for _ in range(500):
            x, y = saccade.update(1 / vtube.FPS, random)
            assert abs(x) <= vtube.SACCADE_AMP
            assert abs(y) <= vtube.SACCADE_AMP

    def test_offset_changes_over_time(self) -> None:
        """時間が経つと別の位置へ飛ぶ（固定値ではない）"""
        saccade = vtube.Saccade()
        seen = set()
        for _ in range(600):
            seen.add(saccade.update(1 / vtube.FPS, random))
        assert len(seen) >= 3

    def test_offset_is_held_between_jumps(self) -> None:
        """飛ぶまでは同じ位置に留まる（毎フレーム乱数だとブレて見える）"""
        saccade = vtube.Saccade()
        first = saccade.update(1 / vtube.FPS, random)
        assert saccade.update(1 / vtube.FPS, random) == first


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
    def __init__(self, mood_params_assigned: bool = False) -> None:
        self.vts_request = _FakeRequestFactory()
        self.requests: list[dict] = []
        self.active_files: set[str] = set()
        self.closed = False
        # VTS側で感情用カスタムパラメータが割り当て済みかどうか
        self.mood_params_assigned = mood_params_assigned
        self.injected: dict[str, float] = {}

    def _live2d_parameters(self) -> list[dict]:
        """割り当て済みなら、注入した感情パラメータがモデル側へ現れる"""
        values = {}
        for mood, name in vtube.MOOD_PARAMS.items():
            live2d_name = f"Param{mood.capitalize()}"
            values[live2d_name] = (
                self.injected.get(name, 0.0)
                if self.mood_params_assigned
                else 0.0
            )
        return [{"name": k, "value": v} for k, v in values.items()]

    async def request(self, payload):
        self.requests.append(payload)
        message_type = payload.get("messageType")
        if message_type == "InjectParameterDataRequest":
            for entry in payload["data"]["parameterValues"]:
                self.injected[entry["id"]] = entry["value"]
        if message_type == "Live2DParameterListRequest":
            return {
                "messageType": "Live2DParameterListResponse",
                "data": {"parameters": self._live2d_parameters()},
            }
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

    def test_head_angles_stay_inside_the_vts_range(self) -> None:
        """画面を見ながら首を振ってもFaceAngleが定格(±30)を超えない"""
        controller, fake = self._start_controller()
        try:
            controller.set_look_override((vtube.SCREEN_LOOK_X, vtube.SCREEN_LOOK_Y))
            for _ in range(3):
                controller.play_gesture("shake")
                time.sleep(0.4)
            angles = [
                parameter["value"]
                for request in list(fake.requests)
                if request.get("messageType") == "InjectParameterDataRequest"
                for parameter in request["data"]["parameterValues"]
                if parameter["id"] in ("FaceAngleX", "FaceAngleY", "FaceAngleZ")
            ]
            assert angles
            worst = max(angles, key=abs)
            assert abs(worst) <= vtube.FACE_ANGLE_LIMIT, f"定格超え: {worst}"
        finally:
            controller.stop()

    def test_gesture_moves_the_head_beyond_the_idle_sway(self) -> None:
        """仕草が待機の揺れを超えて首を動かす（配線）"""
        controller, fake = self._start_controller()
        try:

            def face_angle_y():
                for request in reversed(list(fake.requests)):
                    if request.get("messageType") != "InjectParameterDataRequest":
                        continue
                    for parameter in request["data"]["parameterValues"]:
                        if parameter["id"] == "FaceAngleY":
                            return parameter["value"]
                return None

            assert _wait_until(lambda: face_angle_y() is not None)
            controller.engine.start_gesture("nod")
            # calmの揺れ(sway_y+sub_y+呼吸)は最大3.0程度
            assert _wait_until(
                lambda: abs(face_angle_y() or 0.0) > 3.5,
                timeout=2.0,
            ), "うなずきが首の角度に出ていません"
        finally:
            controller.stop()

    def test_face_params_are_sent_with_the_mood(self) -> None:
        """ムードの眉・口角が実際に送信される（配線）"""
        controller, fake = self._start_controller()
        try:
            controller.set_emotion("amused")

            def sent(name: str):
                for request in reversed(list(fake.requests)):
                    if request.get("messageType") != "InjectParameterDataRequest":
                        continue
                    for parameter in request["data"]["parameterValues"]:
                        if parameter["id"] == name:
                            return parameter["value"]
                return None

            target = vtube.MOODS["amused"]
            assert _wait_until(
                lambda: (sent("MouthSmile") or 0.0) > vtube.FACE_NEUTRAL
            ), "楽しいときに口角が上がりません"
            assert _wait_until(
                lambda: abs((sent("Brows") or 0.0) - target.brows) < 0.1
            ), "眉がムードの値へ寄りません"
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
            # 待機モーションが動き出す＝各ディレクタが待機に入ってから仕掛ける。
            # 起動時の割り当て判定も注入を使うので、首の角度が来たかで見分ける
            assert _wait_until(
                lambda: any(
                    request.get("messageType") == "InjectParameterDataRequest"
                    and any(
                        parameter["id"] == "FaceAngleX"
                        for parameter in request["data"]["parameterValues"]
                    )
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
