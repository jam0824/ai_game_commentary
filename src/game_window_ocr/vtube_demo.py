"""VTube Studio連携の動作確認用スクリプト。

ゲームを起動しなくても、待機モーション・ゲーム画面のチラ見・まばたき・呼吸・
口パク・ムードごとの表情を一通り目で確認できる。モーションの数値を調整したあと、
実況を回さずに見た目を確かめるために使う。

    uv run game-vtube-demo              # 一通り流す
    uv run game-vtube-demo --emotion sad  # 特定の感情だけ
    uv run game-vtube-demo --loop       # Ctrl+Cまで繰り返す
    uv run game-vtube-demo --list       # 中身だけ表示（VTSへ接続しない）

VTube Studioを起動し、プラグインのAPIを有効にしてから実行すること。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .vtube import (
    EMOTION_TO_MOOD,
    MOOD_EXPRESSIONS,
    SCREEN_LOOK_X,
    SCREEN_LOOK_Y,
    VTubeStudioController,
    resolve_emotion_mood,
)

DEFAULT_HOLD = 6.0


@dataclass(frozen=True)
class DemoStep:
    """デモ1ステップ分の指示

    title/check は画面に出す説明。何が起きるべきか分かっていないと
    「動いていない」のか「そういう動きなのか」を判断できないため。
    """

    title: str
    check: str
    emotion: str
    look: tuple[float, float] | None  # None は自動のきょろきょろに任せる
    speak: bool
    seconds: float


def build_demo_steps(
    hold: float = DEFAULT_HOLD,
    emotion: str | None = None,
) -> list[DemoStep]:
    """デモの手順を組み立てる

    hold は1ステップの基準秒数。emotion を渡すとその感情のステップだけに絞る。
    """
    if emotion is not None and emotion.strip().casefold() not in EMOTION_TO_MOOD:
        raise ValueError(
            f"未知の感情です: {emotion}"
            f"（指定できるのは {', '.join(EMOTION_TO_MOOD)}）"
        )

    steps = [
        DemoStep(
            title="待機モーション",
            check="首がゆっくり揺れ、まばたきと呼吸の上下がある",
            emotion="calm",
            look=None,
            speak=False,
            seconds=hold,
        ),
        DemoStep(
            title="ゲーム画面のチラ見",
            check=(
                f"左上（X={SCREEN_LOOK_X:.0f}, Y={SCREEN_LOOK_Y:.0f}）へ"
                "はっきり首を振り、振った側へ首も傾く"
            ),
            emotion="calm",
            look=(SCREEN_LOOK_X, SCREEN_LOOK_Y),
            speak=False,
            seconds=hold,
        ),
        DemoStep(
            title="正面に戻る",
            check="カメラ目線へ戻る。目が先に動いて首が後から追いつく",
            emotion="calm",
            look=(0.0, 0.0),
            speak=False,
            seconds=hold * 0.5,
        ),
        DemoStep(
            title="自動のきょろきょろ",
            check="放っておくと勝手に視線が動き、ときどき画面をチラ見する",
            emotion="calm",
            look=None,
            speak=False,
            seconds=hold * 2.5,
        ),
    ]

    # ムードごとの表情と動きの違い。thoughtfulはtenseと同じ扱いになるかも見る
    mood_steps = [
        ("calm", "平常。表情なしの素の顔で喋る"),
        ("amused", "楽しそうに弾む。表情は素のまま少し速く"),
        ("excited", "大きく速い動き＋「興奮」表情"),
        ("tense", "動きが小さく細かく震える＋「真剣」表情。画面から目を離さない"),
        ("thoughtful", "tenseと同じ「真剣」表情になる"),
        ("sad", "半目でうつむき、動きが重い＋「悲しみ」表情"),
    ]
    for name, check in mood_steps:
        mood = resolve_emotion_mood(name)
        expression = MOOD_EXPRESSIONS.get(mood, "表情なし")
        steps.append(
            DemoStep(
                title=f"感情: {name}（ムード={mood} / 表情={expression}）",
                check=f"{check}。喋っている間だけ口が動く",
                emotion=name,
                look=None,
                speak=True,
                seconds=hold,
            )
        )

    # 驚きは一瞬のバースト演出。連続で来ても毎回頭から出し直せるかを見る
    for index in (1, 2):
        steps.append(
            DemoStep(
                title=f"感情: surprised（{index}回目）",
                check="のけぞって目を見開き、首がカクッと傾いてから戻る",
                emotion="surprised",
                look=None,
                speak=index == 2,
                seconds=hold * 0.6,
            )
        )

    steps.append(
        DemoStep(
            title="待機へ復帰",
            check="表情が消えて素の顔の待機モーションへ戻る",
            emotion="calm",
            look=None,
            speak=False,
            seconds=hold * 0.5,
        )
    )

    if emotion is None:
        return steps

    wanted = emotion.strip().casefold()
    return [step for step in steps if step.emotion == wanted]


def describe_step(index: int, total: int, step: DemoStep) -> str:
    return (
        f"[{index}/{total}] {step.title}（{step.seconds:.1f}秒）\n"
        f"        確認: {step.check}"
    )


def run_steps(
    controller,
    steps: Sequence[DemoStep],
    *,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> None:
    """デモ手順をコントローラへ流す

    中断されても口と視線を戻してから抜ける。開いた口や横を向いた首が
    残ったままVTSに置き去りにされないようにするため。
    """
    total = len(steps)
    try:
        for index, step in enumerate(steps, start=1):
            log(describe_step(index, total, step))
            controller.set_emotion(step.emotion)
            controller.set_look_override(step.look)
            if step.speak:
                controller.set_speaking(True)
                sleep(step.seconds)
                controller.set_speaking(False)
            else:
                sleep(step.seconds)
    finally:
        controller.set_look_override(None)
        controller.set_speaking(False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VTube Studioの待機モーション・視線・表情・口パクを"
            "ゲーム抜きで一通り確認します。"
        )
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=DEFAULT_HOLD,
        help=f"1ステップの基準秒数。既定は {DEFAULT_HOLD}。",
    )
    parser.add_argument(
        "--emotion",
        default=None,
        help=(
            "指定した感情のステップだけを流します。"
            f"選択肢: {', '.join(EMOTION_TO_MOOD)}"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Ctrl+Cで止めるまで繰り返します。",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="手順を表示するだけで、VTube Studioへは接続しません。",
    )
    return parser


def _print_steps(steps: Iterable[DemoStep]) -> None:
    steps = list(steps)
    for index, step in enumerate(steps, start=1):
        print(describe_step(index, len(steps), step))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        steps = build_demo_steps(hold=args.hold, emotion=args.emotion)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    if not steps:
        print("エラー: 実行するステップがありません。", file=sys.stderr)
        return 2

    if args.list_only:
        _print_steps(steps)
        return 0

    total_seconds = sum(step.seconds for step in steps)
    print(
        f"VTube Studioへ接続します（全{len(steps)}ステップ / "
        f"1周およそ{total_seconds:.0f}秒）。Ctrl+Cで中断できます。"
    )

    controller = VTubeStudioController()
    if not controller.start():
        # 接続失敗の警告はコントローラ側が表示済み
        print(
            "VTube Studioへ接続できないためデモを中止します。"
            "VTSを起動し、APIを有効にしてから再実行してください。",
            file=sys.stderr,
        )
        controller.stop()
        return 1

    try:
        while True:
            run_steps(controller, steps)
            if not args.loop:
                break
            print("--- もう1周します（Ctrl+Cで終了） ---")
    except KeyboardInterrupt:
        print("\n中断しました。")
    finally:
        controller.stop()
    print("デモを終了しました。")
    return 0


if __name__ == "__main__":  # pragma: no cover - 手動実行用
    raise SystemExit(main())
