import pytest

from game_window_ocr import vtube
from game_window_ocr.vtube_demo import (
    DemoStep,
    build_demo_steps,
    build_parser,
    run_steps,
)


class _FakeController:
    """VTubeStudioControllerのうち、デモが使う入口だけを記録するフェイク"""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple] = []
        self._fail_on = fail_on

    def set_emotion(self, emotion: str) -> None:
        self.calls.append(("emotion", emotion))

    def set_speaking(self, speaking: bool) -> None:
        self.calls.append(("speaking", speaking))

    def set_look_override(self, target) -> None:
        self.calls.append(("look", target))

    def play_gesture(self, name: str) -> None:
        self.calls.append(("gesture", name))


def _run(steps, controller=None):
    controller = controller or _FakeController()
    slept: list[float] = []
    run_steps(controller, steps, sleep=slept.append, log=lambda *_: None)
    return controller, slept


class TestDemoSteps:
    def test_covers_every_mood(self) -> None:
        """デモで全ムードの表情・動きを一通り確認できる"""
        emotions = {step.emotion for step in build_demo_steps()}
        moods = {vtube.resolve_emotion_mood(e) for e in emotions}
        assert moods == set(vtube.MOODS)

    def test_covers_thoughtful_alias(self) -> None:
        """実況プランで使うthoughtfulも実際に流して確認できる"""
        emotions = [step.emotion for step in build_demo_steps()]
        assert "thoughtful" in emotions

    def test_every_emotion_is_known(self) -> None:
        """デモの感情名はすべて定義済みムードへ解決される"""
        for step in build_demo_steps():
            assert vtube.resolve_emotion_mood(step.emotion) in vtube.MOODS

    def test_includes_screen_glance_step(self) -> None:
        """ゲーム画面のチラ見を狙って再現できるステップがある"""
        looks = [step.look for step in build_demo_steps() if step.look]
        assert (vtube.SCREEN_LOOK_X, vtube.SCREEN_LOOK_Y) in looks

    def test_hold_scales_step_length(self) -> None:
        """--holdでデモ全体の尺が変わる"""
        short = sum(step.seconds for step in build_demo_steps(hold=2.0))
        long = sum(step.seconds for step in build_demo_steps(hold=8.0))
        assert long > short

    def test_filtering_by_emotion(self) -> None:
        """感情を指定するとそのステップだけに絞れる"""
        steps = build_demo_steps(emotion="sad")
        assert steps
        assert {step.emotion for step in steps} == {"sad"}

    def test_unknown_emotion_is_rejected(self) -> None:
        """未知の感情名は黙って空にせずエラーにする"""
        with pytest.raises(ValueError):
            build_demo_steps(emotion="angry")

    def test_covers_every_gesture(self) -> None:
        """全ての仕草を1回ずつ確認できる"""
        played = {step.gesture for step in build_demo_steps() if step.gesture}
        assert played == set(vtube.GESTURES)

    def test_gesture_steps_name_the_gesture(self) -> None:
        """仕草ステップは何が起きるかを説明している"""
        for step in build_demo_steps():
            if step.gesture:
                assert step.gesture in vtube.GESTURES
                assert step.check

    def test_every_step_explains_what_to_watch(self) -> None:
        """各ステップに「何を確認するか」が書いてある"""
        for step in build_demo_steps():
            assert step.title
            assert step.check


class TestRunSteps:
    def test_drives_controller_in_order(self) -> None:
        """ステップ通りに感情・視線・口パクを駆動する（配線）"""
        steps = [
            DemoStep(
                title="チラ見",
                check="左上を向く",
                emotion="calm",
                look=(vtube.SCREEN_LOOK_X, vtube.SCREEN_LOOK_Y),
                speak=False,
                seconds=3.0,
            ),
            DemoStep(
                title="喋る",
                check="口が動く",
                emotion="excited",
                look=None,
                speak=True,
                seconds=4.0,
            ),
        ]
        controller, slept = _run(steps)
        assert controller.calls == [
            ("emotion", "calm"),
            ("look", (vtube.SCREEN_LOOK_X, vtube.SCREEN_LOOK_Y)),
            ("emotion", "excited"),
            ("look", None),
            ("speaking", True),
            ("speaking", False),
            ("look", None),
            ("speaking", False),
        ]
        assert slept == [3.0, 4.0]

    def test_plays_the_gesture_of_a_step(self) -> None:
        """仕草ステップはコントローラへ仕草を流す（配線）"""
        steps = [
            DemoStep(
                title="うなずき",
                check="首が縦に振れる",
                emotion="calm",
                look=None,
                speak=False,
                seconds=2.0,
                gesture="nod",
            )
        ]
        controller, _ = _run(steps)
        assert ("gesture", "nod") in controller.calls

    def test_steps_without_a_gesture_do_not_play_one(self) -> None:
        """仕草を指定しないステップでは仕草を出さない"""
        steps = [
            DemoStep(
                title="待機",
                check="揺れるだけ",
                emotion="calm",
                look=None,
                speak=False,
                seconds=1.0,
            )
        ]
        controller, _ = _run(steps)
        assert not [call for call in controller.calls if call[0] == "gesture"]

    def test_mouth_closes_after_each_speaking_step(self) -> None:
        """喋るステップは必ず口を閉じて終わる"""
        controller, _ = _run(build_demo_steps(hold=1.0))
        speaking = [value for kind, value in controller.calls if kind == "speaking"]
        assert speaking, "口パクを確認するステップがありません"
        assert speaking[-1] is False
        # True の直後は必ず False（開きっぱなしのステップが無い）
        for index, value in enumerate(speaking[:-1]):
            if value is True:
                assert speaking[index + 1] is False

    def test_releases_look_override_at_the_end(self) -> None:
        """デモ終了後は視線固定を解除して自動の動きへ戻す"""
        controller, _ = _run(build_demo_steps(hold=1.0))
        assert controller.calls[-2:] == [("look", None), ("speaking", False)]

    def test_cleans_up_when_interrupted(self) -> None:
        """Ctrl+Cで中断しても口と視線を元へ戻す"""
        controller = _FakeController()

        def interrupt(_seconds):
            raise KeyboardInterrupt

        steps = build_demo_steps(hold=1.0)
        with pytest.raises(KeyboardInterrupt):
            run_steps(controller, steps, sleep=interrupt, log=lambda *_: None)
        assert ("look", None) in controller.calls[-2:]
        assert ("speaking", False) in controller.calls[-2:]


class TestCli:
    def test_defaults(self) -> None:
        """既定では全ステップを1周する"""
        args = build_parser().parse_args([])
        assert args.emotion is None
        assert args.loop is False
        assert args.hold > 0

    def test_emotion_and_loop_options(self) -> None:
        """感情の指定と繰り返し実行を指定できる"""
        args = build_parser().parse_args(["--emotion", "sad", "--loop", "--hold", "2"])
        assert args.emotion == "sad"
        assert args.loop is True
        assert args.hold == 2.0
