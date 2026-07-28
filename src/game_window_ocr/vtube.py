"""VTube Studio連携（Live2Dモデルの感情モーション・表情・口パク）。

live2d_sample_code/default_motion.py の待機モーションをベースに、
同期処理の実況ツール本体から扱えるよう、デーモンスレッド上のasyncioループで
モーションを駆動する。接続失敗や途中切断では警告だけ出し、ゲーム実況は
止めない。
"""

from __future__ import annotations

import asyncio
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, fields, replace
from typing import Any, Awaitable, Callable

try:
    import pyvts
except ImportError:  # pragma: no cover - 実行環境に依存
    pyvts = None  # type: ignore[assignment]

PLUGIN_INFO = {
    "plugin_name": "SkynaController",
    "developer": "you",
    "authentication_token_path": "./token.txt",
}

FPS = 30

# ムード切り替えの滑らかさ(秒)。小さいほどキビキビ切り替わる
MOOD_BLEND_TAU = 0.9

# 驚きバースト(一瞬だけ乗る演出)の長さと、その後元のムードへ戻るまでの時間
SURPRISE_BURST = 0.9
SURPRISE_HOLD = 2.0
# 目の見開き量。素の開き(eye_open_base)から満開までのうち、どこまで見開くか。
# 1.0 でピーク時に目一杯。素の開きが 1.0 だと足しても頭打ちで何も起きないため、
# 各ムードの eye_open_base は満開より下げて余地を作ってある
SURPRISE_EYE = 1.0
SURPRISE_TILT = 5.0    # 首がカクッと傾く量(Z)
SURPRISE_LIFT = 3.0    # 一瞬のけぞる量(Y)

# FaceAngleY の符号。モデルによって上下が逆なら -1.0 にする
LOOK_Y_SIGN = 1.0

# 首の角度の上限。VTS の FaceAngleX/Y と、モデルの ParamAngle* が ±30 のため
FACE_ANGLE_LIMIT = 30.0

# ゲーム画面の方向。キャラは配信画面の右下に立つので、画面は「向かって左上」にある。
# 逆を向いたら符号を反転する。
# 配信画面ではキャラが小さく映るので、はっきり首を振らないと「画面を見た」ことが
# 伝わらない。VTS の FaceAngle は ±30 が定格なので、その範囲で大きめに取る
SCREEN_LOOK_X = -24.0  # 左
SCREEN_LOOK_Y = 9.0    # 上
SCREEN_JITTER = 3.0    # チラ見のたびに少しズラす(毎回同じ点を見ると機械的)

# 待機位置を画面側へ寄せる量の上限。
# チラ見を大きくしたぶん、これを掛けずに寄せると横を向いたまま固定されてしまう
REST_LEAN_MAX = 0.3

# 首の狙い位置に追いつく速さ。大きいほどキビキビ視線が動く。
# 振り幅を大きくしたので、鈍く見えないよう合わせて上げている
LOOK_FOLLOW_RATE = 2.8

# 振り向きに連動する首の傾き(Z)。
# 人は横を向くとき首も向いた側へ傾く。首の角度をそのまま平行移動させるだけだと
# 「顔が横にスライドした」ように見えるので、傾きを足して振り向きらしさを出す。
# 傾く向きが逆に見えるなら符号を反転する
LOOK_TO_TILT = 0.28
LOOK_TILT_MAX = 8.0

# 視線パターン。ムードごとに出やすいものを重みで決める
GAZE_PATTERNS = (
    "screen",       # ゲーム画面をチラ見する
    "wander",       # 何となく視線を動かす
    "stare",        # 画面から目を離さない
    "double_take",  # 二度見する
    "drift",        # 考えながら視線が泳ぐ
    "down",         # 伏し目・うつむく
    "audience",     # 視聴者(カメラ)を見る
)

MOOD_GAZE = {
    "calm": {"screen": 3.0, "wander": 3.0, "audience": 0.8, "drift": 0.5},
    "amused": {"screen": 3.0, "wander": 2.5, "audience": 1.5, "drift": 0.5},
    "excited": {"screen": 4.0, "wander": 2.0, "audience": 1.5, "double_take": 0.8},
    "surprised": {"double_take": 3.0, "screen": 3.0, "wander": 0.5},
    "tense": {"stare": 4.0, "screen": 2.0, "drift": 0.6, "wander": 0.4},
    "sad": {"down": 3.0, "drift": 1.5, "wander": 1.0, "screen": 1.0},
}

# 視線が泳ぐ(drift)ときの振れ幅と、見上げる高さ
DRIFT_RANGE_X = 6.0
DRIFT_LIFT = 5.0

# 伏し目(down)で視線を落とす高さ
DOWNCAST_Y = -7.0

# マイクロサッカード(眼球の細かい飛び)。生体感を出すための微小な揺れ
SACCADE_AMP = 0.025
SACCADE_INTERVAL = (0.25, 1.1)

# 首の角度を目線に変換する係数。
# 人は視線を動かすとき目が先に動き、首が後から追いつく。EYE_LEAD が「まだ首が
# 向いていない分を目で先行する量」、EYE_HOLD が「首の向きに合わせた定常のズレ」。
# 大きく振り向くようになったぶん、上げすぎると目が -1 に振り切れたまま固まり、
# 首が到着しても横目のままになる。首が着いたら目は中央側へ戻す値にしている
EYE_LEAD = 0.05
EYE_HOLD = 0.02

# 待機の揺れに対して目を逆へ回す量(前庭動眼反射)。
# 首が揺れても視線が一点に留まって見えるようにする。大きくすると目が泳ぐ
SWAY_EYE_COMP = 0.01

# まばたき1回の各段階の長さ(秒)。ムードの速さで縮むが、下限を割らない
BLINK_CLOSE = 0.07   # 閉じている
BLINK_HALF = 0.04    # 半開き
BLINK_OPEN = 0.12    # 開ききってから次へ
MIN_BLINK_STEP = 2.0 / FPS

# 感情の変化に気づくまでの間隔。待機中でもこの間隔で見に行く
MOOD_REACT_TICK = 0.2

# 感情が変わってから反応のまばたき・仕草を入れるまでの間(秒)
REACTION_BLINK_DELAY = (0.15, 0.5)
REACTION_GESTURE_DELAY = (0.2, 0.8)

# 呼吸を流し込むための入力パラメータ名。
# VTube Studio の既定パラメータには呼吸用が無いので、起動時にカスタム入力
# パラメータとして自動生成する。生成しただけでは何も動かず、VTube Studio の
# model config タブ内「VTS Parameter Setup」で、呼吸用 Live2D パラメータの
# input にこれを割り当てる必要がある(初回だけ)。Auto-breath は OFF にすること
BREATH_PARAM = "Breath"

# 呼吸で頭がわずかに上下する量。モデル側の呼吸パラメータと二重に効くので控えめに
BREATH_TO_Y = 0.3

# 副波の周期を主波の何倍にするか。割り切れない比にして、
# 主波と副波が同じ位相で揃う周期を長くする(繰り返しに見せない)
SUB_PERIOD_RATIO = 0.43

# 眉(Brows)と口角(MouthSmile)の中立値。
# どちらも 0.0〜1.0 で 0.5 が素の顔、0 で眉が下がりきり・口角がへの字になる。
# 送らないでいるとVTS側の既定値 0 が使われてしまうので、毎フレーム出し続ける
FACE_NEUTRAL = 0.5

# 毎フレーム送る顔まわりの入力パラメータ。値の並び順と対応している
FACE_PARAMS = (
    "FaceAngleX", "FaceAngleY", "FaceAngleZ",
    "EyeOpenLeft", "EyeOpenRight",
    "EyeLeftX", "EyeRightX", "EyeLeftY", "EyeRightY",
    "MouthOpen", "MouthSmile", "Brows",
)


@dataclass(frozen=True)
class MoodParams:
    """ムードごとの待機モーションの味付け。全フィールド float(補間できるように)"""

    sway_x: float          # 首の揺れ 主波の振幅
    sway_y: float
    sway_z: float
    sub_x: float           # 首の揺れ 副波の振幅(周期をずらして機械的に見せない)
    sub_y: float
    sub_z: float
    speed: float           # 揺れ・まばたき・追従の全体的な速さ倍率
    tremor: float          # 高周波の細かい震え(緊張感)
    breath_period: float   # 呼吸1周期の秒数
    breath_depth: float    # 呼吸の深さ(1.0=通常, 小さいほど浅い)
    blink_min: float       # まばたき間隔
    blink_max: float
    double_blink_chance: float
    eye_open_base: float   # 平常時の目の開き(1.0=満開。驚きで見開く余地を残す)
    look_range_x: float    # きょろきょろの振れ幅
    look_range_y: float
    look_bias_x: float     # 視線の基準位置オフセット(上向き/下向きなど)
    look_bias_y: float
    look_min: float        # 視線を動かす間隔
    look_max: float
    screen_glance_chance: float  # 視線を動かすとき、ゲーム画面を見にいく確率
    screen_hold: float           # ゲーム画面を見続ける秒数
    head_tilt: float       # 首の傾き(Z)の固定オフセット
    mouth_speed: float     # 口パクの速さ倍率
    mouth_amp: float       # 口の開き幅倍率
    mouth_smile: float     # 口角(0.5=中立、下げるとへの字。笑い目と頬も連動する)
    brows: float           # 眉の高さ(0.5=中立、下げると寄る)
    gesture_min: float     # 単発の仕草を出す間隔
    gesture_max: float
    gesture_amp: float     # 仕草の大きさ倍率(静かなムードでは小さく出す)
    period_x: float        # 首の揺れ 主波の周期(秒)。軸ごとに変えると動きに癖が出る
    period_y: float
    period_z: float

    def replace(self, **changes):
        return replace(self, **changes)


# 実況プランのemotionをムードへ割り当てる。
# 指定: calm/amusedはデフォルト状態(表情なし)、thoughtfulは真剣(tenseと同じ)
EMOTION_TO_MOOD = {
    "calm": "calm",
    "amused": "amused",
    "excited": "excited",
    "surprised": "surprised",
    "tense": "tense",
    "thoughtful": "tense",
    "sad": "sad",
}


def resolve_emotion_mood(emotion: str) -> str:
    """実況プランのemotionをムード名へ解決する。未知の値はcalmにする"""
    return EMOTION_TO_MOOD.get(str(emotion).strip().casefold(), "calm")


# ムードに対応させる表情。値は VTube Studio 上の表情の「表示名」。
# ここに無いムード(calm など)は表情なしの素の顔で動く。
# 表情を増やしたらここに1行足すだけでよい
MOOD_EXPRESSIONS = {
    "sad": "悲しみ",
    "tense": "真剣",
    "excited": "興奮",
    "surprised": "驚き",
}

# 表情を切り替えるときのフェード秒数。0 だとパッと切り替わる
EXPRESSION_FADE = 0.4

# 感情の「強度」を流し込むためのカスタム入力パラメータ名(ムード -> 名前)。
# モデルが Paramtense のような感情専用パラメータを持っている場合、表情ファイルの
# ON/OFF ではなく 0〜1 の連続値で表情を出せる。ムードの補間がそのまま表情の濃さに
# なるので、切り替わりが滑らかになり「少し真剣」のような中間も作れる。
# Breath と同じく、VTS の「VTS Parameter Setup」で対応する Live2D パラメータの
# input へ割り当てるまでは何も起きない。割り当て済みかは起動時に自動判定し、
# 未割り当てのムードは従来どおり表情ファイルで出す
MOOD_PARAMS = {
    "tense": "MoodTense",
    "sad": "MoodSad",
    "excited": "MoodExcited",
    "surprised": "MoodSurprised",
}

# 割り当て判定で「動いた」とみなすLive2Dパラメータの変化量
MOOD_PARAM_DETECT_DELTA = 0.05

MOODS = {
    # 平常。デフォルトの待機がこれ
    "calm": MoodParams(
        sway_x=2.6, sway_y=1.8, sway_z=2.2,
        sub_x=1.2, sub_y=0.8, sub_z=0.9,
        speed=1.0, tremor=0.0, breath_period=4.0, breath_depth=1.0,
        blink_min=2.0, blink_max=6.0, double_blink_chance=0.25,
        eye_open_base=0.88,
        look_range_x=7.0, look_range_y=3.0,
        look_bias_x=0.0, look_bias_y=0.0,
        look_min=4.0, look_max=10.0,
        screen_glance_chance=0.35, screen_hold=2.5,
        head_tilt=0.0, mouth_speed=1.0, mouth_amp=1.0,
        mouth_smile=0.45, brows=0.5,
        gesture_min=8.0, gesture_max=18.0,
        period_x=7.3, period_y=5.7, period_z=9.1,
        gesture_amp=1.0,
    ),
    # 笑ってる・楽しんでる。表情は素の顔のまま少し弾む
    "amused": MoodParams(
        sway_x=3.2, sway_y=2.4, sway_z=3.0,
        sub_x=1.5, sub_y=1.2, sub_z=1.3,
        speed=1.3, tremor=0.0, breath_period=3.2, breath_depth=1.0,
        blink_min=1.5, blink_max=4.0, double_blink_chance=0.45,
        eye_open_base=0.88,
        look_range_x=6.0, look_range_y=3.0,
        look_bias_x=0.0, look_bias_y=0.8,
        look_min=3.0, look_max=7.0,
        screen_glance_chance=0.45, screen_hold=2.5,
        head_tilt=1.5, mouth_speed=1.2, mouth_amp=1.0,
        mouth_smile=0.70, brows=0.60,
        gesture_min=5.0, gesture_max=12.0,
        period_x=5.9, period_y=3.8, period_z=7.3,
        gesture_amp=1.15,
    ),
    # 大興奮。振幅も速さも大きく、前のめり
    "excited": MoodParams(
        sway_x=4.6, sway_y=3.2, sway_z=3.8,
        sub_x=2.0, sub_y=1.6, sub_z=1.6,
        speed=1.75, tremor=0.15, breath_period=2.2, breath_depth=1.15,
        blink_min=1.0, blink_max=3.0, double_blink_chance=0.5,
        eye_open_base=0.88,
        look_range_x=9.0, look_range_y=4.0,
        look_bias_x=0.0, look_bias_y=1.5,
        look_min=2.0, look_max=5.0,
        screen_glance_chance=0.6, screen_hold=3.0,
        head_tilt=0.0, mouth_speed=1.6, mouth_amp=1.0,
        mouth_smile=0.62, brows=0.78,
        gesture_min=4.0, gesture_max=9.0,
        period_x=6.3, period_y=4.4, period_z=5.1,
        gesture_amp=1.5,
    ),
    # 驚き。持続状態としては「固まってる」。演出は別途バーストで乗る
    "surprised": MoodParams(
        sway_x=1.2, sway_y=0.9, sway_z=1.0,
        sub_x=0.5, sub_y=0.4, sub_z=0.4,
        speed=0.7, tremor=0.1, breath_period=6.0, breath_depth=0.4,
        blink_min=2.5, blink_max=6.0, double_blink_chance=0.15,
        eye_open_base=0.88,
        look_range_x=2.0, look_range_y=1.0,
        look_bias_x=0.0, look_bias_y=1.0,
        look_min=1.5, look_max=3.0,
        # 何が起きた?と真っ先に画面を確認する
        screen_glance_chance=0.7, screen_hold=1.5,
        head_tilt=0.0, mouth_speed=1.4, mouth_amp=1.0,
        mouth_smile=0.50, brows=0.85,
        gesture_min=6.0, gesture_max=14.0,
        period_x=6.8, period_y=8.2, period_z=9.6,
        gesture_amp=1.0,
    ),
    # 緊張・集中(thoughtfulもここへ割り当てる)。画面から目を離さず、細かく震える
    "tense": MoodParams(
        sway_x=1.2, sway_y=0.8, sway_z=1.0,
        sub_x=0.5, sub_y=0.35, sub_z=0.4,
        speed=0.85, tremor=0.35, breath_period=2.4, breath_depth=0.6,
        blink_min=1.5, blink_max=3.2, double_blink_chance=0.3,
        eye_open_base=0.84,
        look_range_x=2.5, look_range_y=1.2,
        look_bias_x=0.0, look_bias_y=-0.5,
        look_min=3.0, look_max=6.0,
        # 画面から目を離さない
        screen_glance_chance=0.85, screen_hold=5.0,
        head_tilt=0.0, mouth_speed=1.1, mouth_amp=0.85,
        mouth_smile=0.35, brows=0.30,
        gesture_min=8.0, gesture_max=20.0,
        period_x=5.2, period_y=6.9, period_z=8.4,
        gesture_amp=0.6,
    ),
    # 落ち込み。半目・下向き・動きが重い
    "sad": MoodParams(
        sway_x=1.3, sway_y=0.9, sway_z=1.1,
        sub_x=0.6, sub_y=0.4, sub_z=0.5,
        speed=0.55, tremor=0.0, breath_period=4.6, breath_depth=1.1,
        blink_min=3.0, blink_max=8.0, double_blink_chance=0.1,
        eye_open_base=0.72,
        look_range_x=3.0, look_range_y=1.5,
        look_bias_x=0.0, look_bias_y=-3.0,
        look_min=6.0, look_max=12.0,
        # うつむきがちで画面をあまり見ない
        screen_glance_chance=0.2, screen_hold=2.0,
        head_tilt=2.5, mouth_speed=0.8, mouth_amp=0.75,
        mouth_smile=0.20, brows=0.38,
        gesture_min=10.0, gesture_max=24.0,
        period_x=8.1, period_y=6.2, period_z=11.3,
        gesture_amp=0.8,
    ),
}


@dataclass(frozen=True)
class Gesture:
    """単発の仕草。待機モーションの上に一時的に足す動き

    cycles > 0 なら往復する動き(うなずき・首振り)、0 なら姿勢を作って
    しばらく保つ動き(首かしげ・前のめり)。amp の符号が最初に動く向き。
    """

    axis: str        # "x"(左右) / "y"(上下) / "z"(傾き)
    amp: float       # 振幅(度)
    seconds: float
    cycles: float = 0.0


# 振幅は「待機の揺れ(calmで3〜4度)より明らかに大きい」ことを基準に決める。
# 揺れと同じくらいだと、何かしたのか分からない動きになる
GESTURES = {
    "nod": Gesture(axis="y", amp=-12.0, seconds=0.9, cycles=1.5),    # うなずき
    # 速く振ると「いいえ」になる。仕草は発話内容と連動せず抽選で出るので、
    # 否定に読める動きは肯定的なセリフとぶつかる。「否定」と「感嘆」を分けるのは
    # 主に速さなので、ゆっくり振って「いやー…」「まいったな」に寄せる。
    # 振幅は下げられない(excited の待機の揺れに埋もれる)ので、速度だけを落とす
    "shake": Gesture(axis="x", amp=9.0, seconds=1.8, cycles=1.2),    # 首を振る
    "tilt": Gesture(axis="z", amp=9.0, seconds=2.6),                 # 首かしげ
    "lean_in": Gesture(axis="y", amp=-7.0, seconds=2.2),             # 前のめり
    "droop": Gesture(axis="y", amp=-8.0, seconds=3.2),               # うなだれる
}

# ムードごとに出る仕草と、その出やすさ。
# 補間はしない(仕草は抽選した瞬間に決まる離散的なもの)ので MoodParams とは別表
# 1つの仕草へ寄せすぎると、そのムードの間ずっと同じ動きに見える。
# 特に tense は配信時間の半分を占めるので、種類を散らしておく
MOOD_GESTURES = {
    "calm": {"nod": 1.0, "tilt": 0.8, "shake": 0.3},
    "amused": {"nod": 1.5, "tilt": 1.0, "shake": 0.5},
    "excited": {"nod": 2.0, "shake": 0.8, "lean_in": 1.2},
    "surprised": {"lean_in": 1.0, "tilt": 0.8},
    "tense": {"lean_in": 1.0, "tilt": 0.8, "nod": 0.5, "shake": 0.3},
    "sad": {"droop": 1.5, "shake": 0.6, "tilt": 0.8},
}

# 実況プランの mode ごとに出やすくする仕草の倍率。
# 反射的な反応(reaction)は「え？」という首かしげ、じっくり話す
# (extended)は相槌のうなずきや前のめり、というように傾向を分ける
MODE_GESTURE_BIAS = {
    # reaction の「え？」は tilt で足りる。遅くした shake を重ねると反応が鈍く見える
    "reaction": {"tilt": 3.0, "shake": 1.0, "nod": 0.4},
    "quick": {"nod": 1.5, "tilt": 1.0},
    "extended": {"nod": 2.0, "lean_in": 1.5, "shake": 0.7},
}

# 実況プランの intensity(0〜1) を仕草の大きさ・間隔へ変換する範囲。
# 強度0でも動きを止めず、強度1でも暴れすぎない幅にする
INTENSITY_AMP_MIN = 0.7
INTENSITY_AMP_MAX = 1.3
INTENSITY_INTERVAL_MIN = 0.7
INTENSITY_INTERVAL_MAX = 1.3

# 仕草の立ち上がり・戻りにかける割合。
# EASE は姿勢を保つ仕草(首かしげなど)、EDGE_FADE は往復する仕草の端の丸め。
# 往復する仕草に全体窓を掛けると、最初と最後の振りが削れて「1回動いただけ」に
# 見えてしまうので、端だけ短く繋ぐ
GESTURE_EASE = 0.25
GESTURE_EDGE_FADE = 0.15


def clamp(value, low, high):
    return max(low, min(high, value))


def smoothstep(t: float) -> float:
    """0→1 をなめらかに繋ぐ(端で速度が0になる)"""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gesture_offset(gesture: Gesture, progress: float) -> float:
    """仕草の進行度(0〜1)に対する角度

    両端で必ず0に戻す。途中で待機モーションへ足しても継ぎ目が出ないように。
    """
    if progress <= 0.0 or progress >= 1.0:
        return 0.0
    if gesture.cycles > 0.0:
        # 往復する動き。端だけ丸めて、途中の振りは削らない
        fade = smoothstep(
            min(progress, 1.0 - progress) / GESTURE_EDGE_FADE
        )
        return gesture.amp * math.sin(
            2 * math.pi * gesture.cycles * progress
        ) * fade
    # 姿勢を作って保つ動き
    if progress < GESTURE_EASE:
        ease = smoothstep(progress / GESTURE_EASE)
    elif progress > 1.0 - GESTURE_EASE:
        ease = smoothstep((1.0 - progress) / GESTURE_EASE)
    else:
        ease = 1.0
    return gesture.amp * ease


def gesture_amp_scale(intensity: float) -> float:
    """強度から仕草の大きさの倍率"""
    ratio = clamp(intensity, 0.0, 1.0)
    return INTENSITY_AMP_MIN + (INTENSITY_AMP_MAX - INTENSITY_AMP_MIN) * ratio


def gesture_interval_scale(intensity: float) -> float:
    """強度から仕草の間隔の倍率(高いほど詰まる)"""
    ratio = clamp(intensity, 0.0, 1.0)
    return INTENSITY_INTERVAL_MAX - (
        INTENSITY_INTERVAL_MAX - INTENSITY_INTERVAL_MIN
    ) * ratio


def pick_gesture(mood: str, rng, bias: dict[str, float] | None = None) -> str:
    """そのムードで出る仕草を重み付きで選ぶ

    bias を渡すと、実況の mode に合わせて出やすさを傾ける。
    ムードに無い仕草は bias があっても出さない(ムードの性格を壊さないため)。
    """
    weights = MOOD_GESTURES.get(mood) or MOOD_GESTURES["calm"]
    if bias:
        weights = {
            name: weight * bias.get(name, 1.0) for name, weight in weights.items()
        }
        if not any(weights.values()):
            weights = MOOD_GESTURES.get(mood) or MOOD_GESTURES["calm"]
    total = sum(weights.values())
    threshold = rng.random() * total
    upto = 0.0
    for name, weight in weights.items():
        upto += weight
        if threshold <= upto:
            return name
    return next(iter(weights))


def lerp_params(a: MoodParams, b: MoodParams, ratio: float) -> MoodParams:
    """MoodParams 同士を線形補間する(切り替えをガクつかせないため)"""
    inv = 1.0 - ratio
    return MoodParams(
        **{
            f.name: getattr(a, f.name) * inv + getattr(b, f.name) * ratio
            for f in fields(MoodParams)
        }
    )


def surprise_envelope(u: float) -> float:
    """驚きバーストの強さ。u=0〜1 で、立ち上がりが鋭く後は減衰する"""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    if u < 0.10:
        return u / 0.10
    return math.exp(-(u - 0.10) * 4.0)


def eye_look_offset(head: float, residual: float, sway: float = 0.0) -> float:
    """目線のオフセット

    head は今の首の向き、residual は狙いに対して首がまだ向いていない残りの角度。
    人は視線を動かすとき目が先に動いて首が後から追いつくので、residual が大きい
    振り向きはじめに目を大きく振り、首が追いついたらズレを小さくする。
    sway は待機の揺れ。人は首が揺れても視線は同じ点に留まる(前庭動眼反射)ので、
    揺れの分だけ目を逆へ回して、一点を見ている感じを出す。
    VTS の Eye*X / Eye*Y は -1〜1 なので、はみ出さないよう丸める。
    """
    return clamp(
        head * EYE_HOLD + residual * EYE_LEAD - sway * SWAY_EYE_COMP,
        -1.0,
        1.0,
    )


def vertical_gaze(
    head: float,
    residual: float,
    sway: float = 0.0,
) -> tuple[float, float]:
    """上下方向の (首の角度, 目線) を返す

    LOOK_Y_SIGN をどちらにも同じように掛けるための入口。首だけ反転させると
    「顔は上、目は下」というちぐはぐな向きになるので、ここで揃えておく。
    """
    return (
        head * LOOK_Y_SIGN,
        eye_look_offset(head, residual, sway) * LOOK_Y_SIGN,
    )


def surprise_eye_bonus(params: MoodParams, strength: float) -> float:
    """驚いたときに目を見開く量

    素の開きから満開までの残りを埋める形にする。こうしておくと、半目のムード
    (sad など)から驚いても、素の顔から驚いても、同じだけ「見開いた」に見える。
    """
    return (1.0 - params.eye_open_base) * SURPRISE_EYE * strength


def eye_open_value(blink: float, params: MoodParams, bonus: float) -> float:
    """送信する目の開き

    まばたき(blink)を後から掛けるのは、驚いている最中でも目が閉じきるように
    するため。足し算にすると見開き量のぶん閉じ残る。
    """
    return clamp(blink * (params.eye_open_base + bonus), 0.0, 1.0)


def blink_phase(seconds: float, speed: float) -> float:
    """まばたき1段階の長さ

    ムードが速いほど短くするが、30fps で送っている以上、1フレームを切ると
    まばたきが抜けたりチラついたりする。最低2フレームは持たせる。
    """
    return max(seconds / max(speed, 0.3), MIN_BLINK_STEP)


def input_parameter_names(response: Any) -> set[str]:
    """InputParameterListRequest のレスポンスからパラメータ名を集める

    モデルが受け付ける入力パラメータの確認用。APIError でも落ちないようにする。
    """
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, dict):
        return set()

    names = set()
    for key in ("defaultParameters", "customParameters"):
        for entry in data.get(key) or []:
            name = entry.get("name") if isinstance(entry, dict) else None
            if name:
                names.add(name)
    return names


def screen_point(rng) -> tuple[float, float]:
    """ゲーム画面の方を見るときの狙い。毎回わずかにズラす"""
    return (
        SCREEN_LOOK_X + rng.uniform(-SCREEN_JITTER, SCREEN_JITTER),
        SCREEN_LOOK_Y + rng.uniform(-SCREEN_JITTER, SCREEN_JITTER),
    )


def pick_gaze_pattern(mood: str, rng) -> str:
    """そのムードで出る視線パターンを重み付きで選ぶ"""
    weights = MOOD_GAZE.get(mood) or MOOD_GAZE["calm"]
    total = sum(weights.values())
    threshold = rng.random() * total
    upto = 0.0
    for name, weight in weights.items():
        upto += weight
        if threshold <= upto:
            return name
    return next(iter(weights))


def plan_gaze(
    pattern: str,
    params: MoodParams,
    rng,
) -> list[tuple[float, float, float]]:
    """視線パターンを (狙いX, 狙いY, 見ている秒数) の並びへ展開する

    1点を見るだけでなく、二度見のように複数段階の動きを表せるようにしている。
    rng を引数に取るのは、テストで揺らぎを固定できるようにするため。
    """
    speed = max(params.speed, 0.3)

    if pattern == "screen":
        x, y = screen_point(rng)
        hold = rng.uniform(params.screen_hold * 0.6, params.screen_hold * 1.4)
        return [(x, y, hold)]

    if pattern == "stare":
        # 画面から目を離さない。チラ見よりずっと長く留める
        x, y = screen_point(rng)
        return [(x, y, rng.uniform(params.screen_hold * 2.0, params.screen_hold * 3.5))]

    if pattern == "double_take":
        # 一度見て、正面へ戻して、もう一度見る
        first_x, first_y = screen_point(rng)
        again_x, again_y = screen_point(rng)
        return [
            (first_x, first_y, rng.uniform(0.3, 0.5)),
            (rng.uniform(-2.0, 2.0), rng.uniform(-1.5, 1.5), rng.uniform(0.25, 0.45)),
            (again_x, again_y, rng.uniform(params.screen_hold * 0.8, params.screen_hold * 1.6)),
        ]

    if pattern == "drift":
        # 考えているときに視線が泳ぐ。上寄りを何回か彷徨う
        return [
            (
                DRIFT_RANGE_X * side + rng.uniform(-1.5, 1.5),
                DRIFT_LIFT + rng.uniform(-1.5, 1.5),
                rng.uniform(0.6, 1.4) / speed,
            )
            for side in (-1.0, 0.4, -0.7)
        ]

    if pattern == "down":
        # 伏し目。うつむいて視線を落とす
        return [
            (
                rng.uniform(-2.5, 2.5),
                DOWNCAST_Y + rng.uniform(-1.5, 1.5),
                rng.uniform(2.0, 4.0) / speed,
            )
        ]

    if pattern == "audience":
        # 視聴者の方を見る(カメラ目線)。同意を求めるような間になる
        return [
            (
                rng.uniform(-1.5, 1.5),
                rng.uniform(-1.0, 1.0),
                rng.uniform(1.5, 3.0) / speed,
            )
        ]

    # wander(既定): ムードの範囲で何となく視線を動かす
    return [
        (
            rng.uniform(-params.look_range_x, params.look_range_x),
            rng.uniform(-params.look_range_y, params.look_range_y),
            rng.uniform(2.0, 4.0) / speed,
        )
    ]


class Saccade:
    """眼球の細かい飛び(マイクロサッカード)

    人の目は一点を見ているときも小刻みに位置を変えている。毎フレーム乱数だと
    ただのノイズに見えるので、次に飛ぶまでは同じ位置に留める。
    """

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self._left = 0.0

    def update(self, dt: float, rng) -> tuple[float, float]:
        self._left -= dt
        if self._left <= 0.0:
            self._left = rng.uniform(*SACCADE_INTERVAL)
            self.x = rng.uniform(-SACCADE_AMP, SACCADE_AMP)
            self.y = rng.uniform(-SACCADE_AMP, SACCADE_AMP)
        return self.x, self.y


def rest_look_target(params: MoodParams, rng) -> tuple[float, float]:
    """視線を戻す先

    画面をよく見るムードほど正面ではなく画面寄りで待機させる。
    そうしないとチラ見のあと毎回カメラ目線に戻ってしまい、
    「画面に集中している」感じが消える。
    """
    lean = min(params.screen_glance_chance * 0.5, REST_LEAN_MAX)
    x = SCREEN_LOOK_X * lean + rng.uniform(-1.5, 1.5)
    y = SCREEN_LOOK_Y * lean + rng.uniform(-1.0, 1.0)
    return x, y


def turn_tilt(head_x: float) -> float:
    """振り向いた角度に対して、連動して首を傾ける量(Z)"""
    return clamp(head_x * LOOK_TO_TILT, -LOOK_TILT_MAX, LOOK_TILT_MAX)


def idle_angles(
    phase: float,
    params: MoodParams,
    breath_phase: float | None = None,
) -> tuple[float, float, float, float]:
    """複数のサイン波を重ねて、周期の読めない自然な揺れを作る

    phase は「速さ倍率を掛けたあとの経過時間」。実時間ではないので、
    ムードが変わって speed が変化しても波形が飛ばない。
    """
    p = params
    x = (
        p.sway_x * math.sin(phase * 2 * math.pi / p.period_x)
        + p.sub_x * math.sin(phase * 2 * math.pi / (p.period_x * SUB_PERIOD_RATIO))
        + p.tremor * math.sin(phase * 2 * math.pi / 0.19)
    )
    y = (
        p.sway_y * math.sin(phase * 2 * math.pi / p.period_y)
        + p.sub_y * math.sin(phase * 2 * math.pi / (p.period_y * SUB_PERIOD_RATIO))
        + p.tremor * 0.7 * math.sin(phase * 2 * math.pi / 0.23)
    )
    z = (
        p.sway_z * math.sin(phase * 2 * math.pi / p.period_z)
        + p.sub_z * math.sin(phase * 2 * math.pi / (p.period_z * SUB_PERIOD_RATIO))
        + p.tremor * 0.5 * math.sin(phase * 2 * math.pi / 0.13)
        + p.head_tilt
    )

    if breath_phase is None:
        breath_phase = phase / p.breath_period
    # 深さを上げると上下で頭打ちになり、息を吸いきる「間」ができる
    breath = clamp(
        0.5 + 0.5 * p.breath_depth * math.sin(breath_phase * 2 * math.pi), 0.0, 1.0
    )
    return x, y, z, breath


def expression_files_by_name(response: Any) -> dict[str, str]:
    """ExpressionStateRequest のレスポンスから 表示名 -> ファイル名 の対応を作る

    コードでは表示名(「悲しみ」など)で指定したいが、API はファイル名を要求するため。
    """
    try:
        expressions = response["data"]["expressions"]
    except (TypeError, KeyError):
        return {}
    if not isinstance(expressions, list):
        return {}

    return {
        e["name"]: e["file"]
        for e in expressions
        if isinstance(e, dict) and e.get("name") and e.get("file")
    }


def resolve_mood_expressions(by_name: dict[str, str]) -> dict[str, str]:
    """ムード名 -> 表情ファイル の対応を作る

    VTS 側に登録されていない表情は黙って外す。1つ足りないだけで
    他のムードまで表情なしになると原因が分かりにくいので。
    """
    return {
        mood: by_name[label]
        for mood, label in MOOD_EXPRESSIONS.items()
        if label in by_name
    }


def missing_mood_expressions(by_name: dict[str, str]) -> list[str]:
    """VTS 側に見つからなかった表情の表示名(起動時の警告用)"""
    return [label for label in MOOD_EXPRESSIONS.values() if label not in by_name]


def active_expression_files(response: Any) -> list[str]:
    """ExpressionStateRequest のレスポンスから、今ONの表情ファイル名を取り出す

    モデル未読み込みや APIError が返ってきても落ちないようにする。
    """
    try:
        expressions = response["data"]["expressions"]
    except (TypeError, KeyError):
        return []
    if not isinstance(expressions, list):
        return []
    return [e["file"] for e in expressions if e.get("active")]


def parameter_created(response: Any) -> bool:
    """ParameterCreationRequest が成功したかどうか"""
    return (
        isinstance(response, dict)
        and response.get("messageType") == "ParameterCreationResponse"
    )


class MoodEngine:
    """現在のムードを保持し、毎フレーム分の待機モーションを計算する"""

    def __init__(self, mood: str = "calm") -> None:
        self.mood = mood if mood in MOODS else "calm"
        self.params = MOODS[self.mood]
        self.phase = 0.0
        self.breath_phase = 0.0
        self._prev_mood = self.mood
        self._hold_left = 0.0
        self._burst_t: float | None = None
        self._gesture: Gesture | None = None
        self._gesture_t = 0.0
        # ムードの配合。今のムードへ 1.0 で寄っていく(表情の濃さに使う)
        self.mix = {name: (1.0 if name == self.mood else 0.0) for name in MOODS}
        self.speaking = False
        # 実況プランの強度とモード。仕草の大きさ・間隔・傾向に効く。
        # 0.5 が素の大きさ(倍率1.0)で、そこから上下する
        self.intensity = 0.5
        self.mode: str | None = None

    def set_intensity(self, intensity: float) -> None:
        """実況プランの強度(0〜1)を反映する"""
        self.intensity = clamp(float(intensity), 0.0, 1.0)

    def set_mood(self, name: str, hold: float | None = None) -> None:
        """ムードを切り替える。hold を渡すとその秒数後に元のムードへ戻る"""
        name = name if name in MOODS else "calm"

        if name == "surprised":
            # 実況中は驚きが連発する。演出は毎回頭から出し直す
            self._burst_t = 0.0
            if hold is None:
                hold = SURPRISE_HOLD
        elif name == self.mood:
            return

        if hold is not None:
            # 一時ムードの入れ子で戻り先を見失わないようにする
            if self._hold_left <= 0.0:
                self._prev_mood = self.mood
            self._hold_left = hold
        else:
            self._hold_left = 0.0

        self.mood = name

    def set_speaking(self, speaking: bool) -> None:
        """発話の開始・終了を伝える

        set_emotion は音声の「生成前」に呼ばれるので、驚きの演出をそのまま流すと
        声が出る頃には終わっている。喋り始めで演出を出し直し、喋っている間は
        元のムードへ戻さないようにする。
        """
        speaking = bool(speaking)
        if speaking and not self.speaking and self.mood == "surprised":
            self._burst_t = 0.0
        self.speaking = speaking

    def start_gesture(self, name: str) -> None:
        """単発の仕草を積む。実行中の仕草は差し替える

        未知の名前は黙って無視する。仕草を増やすときに表の書き間違いで
        モーション全体が止まると困るため。
        """
        gesture = GESTURES.get(name)
        if gesture is None:
            return
        self._gesture = gesture
        self._gesture_t = 0.0

    def gesture_angles(self) -> tuple[float, float, float]:
        """実行中の仕草による (x, y, z) の加算量"""
        if self._gesture is None:
            return 0.0, 0.0, 0.0
        # ムードの落ち着き(params.gesture_amp)と実況の強度の両方で大きさを決める
        offset = (
            gesture_offset(self._gesture, self._gesture_t / self._gesture.seconds)
            * self.params.gesture_amp
            * gesture_amp_scale(self.intensity)
        )
        if self._gesture.axis == "x":
            return offset, 0.0, 0.0
        if self._gesture.axis == "y":
            return 0.0, offset, 0.0
        return 0.0, 0.0, offset

    def update(self, dt: float) -> None:
        """dt 秒ぶん時間を進める。パラメータ補間と位相の更新はここだけ"""
        # 一時ムード(驚きなど)の残り時間を消化して元へ戻す。
        # 喋っている間は止めておく。言い終わる前に平常へ戻ると、
        # 「うわっ!」と言いながら素の顔という食い違いが起きる
        if self._hold_left > 0.0 and not self.speaking:
            self._hold_left -= dt
            if self._hold_left <= 0.0:
                self._hold_left = 0.0
                self.mood = self._prev_mood

        # 目標ムードへ滑らかに寄せる。切り替えの瞬間にガクッと動かないように
        target = MOODS[self.mood]
        blend = 1.0 - math.exp(-dt / MOOD_BLEND_TAU)
        self.params = lerp_params(self.params, target, blend)
        for name in self.mix:
            goal = 1.0 if name == self.mood else 0.0
            self.mix[name] += (goal - self.mix[name]) * blend

        if self._burst_t is not None:
            self._burst_t += dt
            if self._burst_t >= SURPRISE_BURST:
                self._burst_t = None

        if self._gesture is not None:
            self._gesture_t += dt
            if self._gesture_t >= self._gesture.seconds:
                self._gesture = None
                self._gesture_t = 0.0

        env = self.burst_strength()
        self.phase += dt * self.params.speed
        # 驚いた瞬間は息を止める
        self.breath_phase += dt / self.params.breath_period * (1.0 - env)

    def burst_strength(self) -> float:
        if self._burst_t is None:
            return 0.0
        return surprise_envelope(self._burst_t / SURPRISE_BURST)

    def mood_param_values(self) -> dict[str, float]:
        """感情パラメータへ送る強度(ムード -> 0〜1)

        驚きだけは補間を待たずにバーストの強さで立ち上げる。一瞬の演出なので、
        滑らかに寄せていると山が過ぎてしまう。
        """
        values = {
            mood: clamp(self.mix.get(mood, 0.0), 0.0, 1.0) for mood in MOOD_PARAMS
        }
        if "surprised" in values:
            values["surprised"] = max(values["surprised"], self.burst_strength())
        return values

    def eye_open_bonus(self) -> float:
        """驚いた直後だけ目を見開く量"""
        return surprise_eye_bonus(self.params, self.burst_strength())

    def angles(self) -> tuple[float, float, float, float]:
        """この瞬間の (x, y, z, breath) を返す"""
        x, y, z, breath = idle_angles(self.phase, self.params, self.breath_phase)
        env = self.burst_strength()
        if env > 0.0:
            z += SURPRISE_TILT * env
            y += SURPRISE_LIFT * env
        gesture_x, gesture_y, gesture_z = self.gesture_angles()
        return x + gesture_x, y + gesture_y, z + gesture_z, breath


class VTubeStudioController:
    """VTube Studioへ待機モーション・表情・口パクを送り続けるコントローラ

    実況本体（同期処理）から使う入口は start / set_emotion / set_speaking /
    stop の4つだけ。VTSが起動していない・途中で落ちた場合も例外を外へ
    漏らさず、警告表示のみで実況を継続させる。
    """

    def __init__(
        self,
        *,
        plugin_info: dict[str, str] | None = None,
        connect: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self.engine = MoodEngine("calm")
        self.state = {
            "eye_open": 1.0,     # 0.0=閉じる 1.0=開く(まばたき用。ムードの半目とは別)
            "mouth_open": 0.0,   # 0.0=閉じる 1.0=開く
            "is_speaking": False,
            "look_x": 0.0,       # 視線・首の狙い位置(ゆっくり動く目標値)
            "look_y": 0.0,
            # (x, y) を入れている間は自動のきょろきょろを止めてそこを見続ける
            "look_override": None,
        }
        # 起動時の確認で埋まる。モデル/VTS に無いものは送らない
        self.runtime: dict[str, Any] = {
            "breath_param": None,   # 呼吸パラメータ名(見つからなければ None)
            "expressions": {},      # ムード名 -> 表情ファイル
            "mood_params": {},      # ムード名 -> 感情パラメータ名(割り当て済みのみ)
        }
        self._plugin_info = dict(plugin_info or PLUGIN_INFO)
        self._connect = connect
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._vts_lock: asyncio.Lock | None = None
        self._attempt_finished = threading.Event()
        self._connected = threading.Event()
        self._stop_requested = threading.Event()

    # ---- 実況本体から呼ぶ同期API ----

    def start(self, wait: float = 10.0) -> bool:
        """接続スレッドを開始する

        wait 秒までに接続失敗が確定したら False。初回認証待ちなどで
        まだ結果が出ていない場合は True を返し、接続でき次第モーションを始める。
        """
        if self._thread is not None:
            return self._connected.is_set() or not self._attempt_finished.is_set()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="vtube-studio",
            daemon=True,
        )
        self._thread.start()
        self._attempt_finished.wait(wait)
        if self._attempt_finished.is_set():
            return self._connected.is_set()
        print(
            "VTube Studioの接続待ちのまま実況を開始します"
            "（初回はVTS側でプラグインの許可が必要です）。"
        )
        return True

    def set_emotion(
        self,
        emotion: str,
        intensity: float | None = None,
        mode: str | None = None,
    ) -> None:
        """実況プランの emotion / intensity / mode をモーションへ反映する

        intensity は仕草の大きさと間隔、mode は感情が動いた直後に出る仕草の
        傾向に効く。省略した場合は前の値のまま。
        """
        mood = resolve_emotion_mood(emotion)

        def apply() -> None:
            if intensity is not None:
                self.engine.set_intensity(intensity)
            if mode is not None:
                self.engine.mode = str(mode).strip().casefold()
            self.engine.set_mood(mood)

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(apply)
        else:
            apply()

    def play_gesture(self, name: str) -> None:
        """単発の仕草(うなずきなど)をその場で出す"""
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self.engine.start_gesture, name)
        else:
            self.engine.start_gesture(name)

    def set_look_override(self, target: tuple[float, float] | None) -> None:
        """視線の狙いを固定する。None を渡すと自動のきょろきょろへ戻る

        角度はムードごとの基準位置(look_bias)を足さない絶対値。
        動作確認用スクリプトのように、狙った向きを確実に見せたいときに使う。
        """
        if target is None:
            self.state["look_override"] = None
            return
        x, y = target
        self.state["look_override"] = (float(x), float(y))

    def set_speaking(self, speaking: bool) -> None:
        """音声再生中フラグ。Trueの間だけ口パクする

        口パクだけでなく、驚きの演出を声に合わせ直すためにも使う。
        """
        self.state["is_speaking"] = bool(speaking)
        if not speaking:
            self.state["mouth_open"] = 0.0
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self.engine.set_speaking, bool(speaking))
        else:
            self.engine.set_speaking(bool(speaking))

    def stop(self, timeout: float = 3.0) -> None:
        """モーションを止め、表情をリセットして切断する"""
        self._stop_requested.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    # ---- 接続スレッド側 ----

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            print(
                f"警告: VTube Studio連携が停止しました（実況は継続します）: {exc}",
                file=sys.stderr,
            )
        finally:
            self._loop = None
            self._connected.clear()
            self._attempt_finished.set()

    async def _open(self) -> Any:
        if self._connect is not None:
            return await self._connect()
        if pyvts is None:
            raise RuntimeError(
                "pyvtsがインストールされていません（uv sync を実行してください）"
            )
        vts = pyvts.vts(plugin_info=self._plugin_info)
        await vts.connect()
        await vts.request_authenticate_token()
        await vts.request_authenticate()
        return vts

    async def _run(self) -> None:
        try:
            vts = await self._open()
        except Exception as exc:
            print(
                "警告: VTube Studioへ接続できないため、"
                f"モーション連携なしで実況を続けます: {exc}",
                file=sys.stderr,
            )
            self._attempt_finished.set()
            return

        self._vts_lock = asyncio.Lock()
        try:
            await self._reset_expressions(vts)
            names = await self._setup_parameters(vts)
            self.runtime["mood_params"] = await self._setup_mood_params(vts, names)
            await self._setup_expressions(vts)
        except Exception as exc:
            print(
                "警告: VTube Studioの初期化が一部失敗しましたが、"
                f"可能な範囲でモーションを続けます: {exc}",
                file=sys.stderr,
            )

        self._loop = asyncio.get_running_loop()
        self._connected.set()
        self._attempt_finished.set()
        print("VTube Studioへ接続しました。感情モーションと口パクを開始します。")

        tasks = [
            asyncio.create_task(coroutine)
            for coroutine in (
                self._sender_loop(vts),
                self._blink_director(),
                self._look_director(),
                self._gesture_director(),
                self._mouth_director(),
                self._expression_director(vts),
                self._stop_watcher(),
            )
        ]
        try:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
        finally:
            self._loop = None
            self._connected.clear()
            try:
                await asyncio.wait_for(self._shutdown(vts), timeout=2.0)
            except Exception:
                pass

    async def _shutdown(self, vts: Any) -> None:
        try:
            await self._reset_expressions(vts, fade=0.0)
        finally:
            await vts.close()

    async def wait_for_mood_change(self, seconds: float) -> bool:
        """seconds 待つ。途中で感情が変わったら待たずに返る（戻り値: 変わったか）

        待機の間隔はムードごとに最大10秒あるので、素直に眠ると「驚いたのに
        画面を確認しにいかない」ような取りこぼしが出る。刻んで見に行く。
        """
        mood = self.engine.mood
        remaining = seconds
        while remaining > 0.0:
            await asyncio.sleep(min(MOOD_REACT_TICK, remaining))
            remaining -= MOOD_REACT_TICK
            if self.engine.mood != mood:
                return True
        return False

    async def _stop_watcher(self) -> None:
        while not self._stop_requested.is_set():
            await asyncio.sleep(0.1)

    async def _request(self, vts: Any, payload: Any) -> Any:
        assert self._vts_lock is not None
        async with self._vts_lock:
            return await vts.request(payload)

    # ---- 起動時セットアップ ----

    async def _setup_parameters(self, vts: Any) -> set[str]:
        """入力パラメータを確認し、呼吸用が無ければ作る"""
        response = await self._request(
            vts, vts.vts_request.requestTrackingParameterList()
        )

        names = input_parameter_names(response)
        if not names:
            print(
                "警告: VTSの入力パラメータ一覧を取得できませんでした",
                file=sys.stderr,
            )
            return names

        if BREATH_PARAM in names:
            self.runtime["breath_param"] = BREATH_PARAM
        elif await self._create_breath_parameter(vts):
            self.runtime["breath_param"] = BREATH_PARAM
            names = names | {BREATH_PARAM}
            print(
                f"呼吸パラメータ「{BREATH_PARAM}」を作成しました。"
                "VTube Studioの「VTS Parameter Setup」で呼吸用Live2Dパラメータの"
                f"inputへ「{BREATH_PARAM}」を割り当ててください"
                "（Auto-breathはOFFに）。"
            )
        else:
            self.runtime["breath_param"] = None
            print(
                f"警告: 呼吸パラメータ「{BREATH_PARAM}」を作成できませんでした。"
                "頭の上下だけで呼吸を表現します",
                file=sys.stderr,
            )

        missing = [p for p in FACE_PARAMS if p not in names]
        if missing:
            print(
                "警告: 次のVTS入力パラメータが見つかりません: "
                + ", ".join(missing),
                file=sys.stderr,
            )
        return names

    async def _live2d_values(self, vts: Any) -> dict[str, float]:
        """モデル側のLive2Dパラメータの現在値"""
        response = await self._request(
            vts, vts.vts_request.BaseRequest("Live2DParameterListRequest", {})
        )
        try:
            entries = response["data"]["parameters"]
        except (TypeError, KeyError):
            return {}
        if not isinstance(entries, list):
            return {}
        return {
            e["name"]: float(e["value"])
            for e in entries
            if isinstance(e, dict) and e.get("name") and e.get("value") is not None
        }

    async def _push(self, vts: Any, values: dict[str, float], frames: int = 3) -> None:
        """指定の入力パラメータを数フレームだけ押し込む(割り当て判定用)"""
        for _ in range(frames):
            await self._request(
                vts,
                vts.vts_request.requestSetMultiParameterValue(
                    parameters=list(values),
                    values=list(values.values()),
                    face_found=True,
                ),
            )
            await asyncio.sleep(1 / FPS)

    async def _moves_model(
        self,
        vts: Any,
        before: dict[str, float],
        active: dict[str, float],
    ) -> bool:
        """入力を押したときにモデル側が動くか＝割り当て済みかを調べる

        どのLive2Dパラメータへ割り当てたかは利用者の自由なので、名前を決め打ちせず
        「何かが動いたか」で判定する。before は全部0で押したときの値。
        """
        idle = {name: 0.0 for name in MOOD_PARAMS.values()}
        await self._push(vts, {**idle, **active})
        after = await self._live2d_values(vts)
        return any(
            abs(value - before[name]) > MOOD_PARAM_DETECT_DELTA
            for name, value in after.items()
            if name in before
        )

    async def _setup_mood_params(self, vts: Any, names: set[str]) -> dict[str, str]:
        """感情パラメータを用意し、VTS側で割り当て済みのものだけ採用する"""
        for param in MOOD_PARAMS.values():
            if param in names:
                continue
            await self._request(
                vts,
                vts.vts_request.requestCustomParameter(
                    param, min=0.0, max=1.0, default_value=0.0, info="感情の強度"
                ),
            )

        idle = {param: 0.0 for param in MOOD_PARAMS.values()}
        await self._push(vts, idle)
        before = await self._live2d_values(vts)

        # まとめて押して何も動かなければ、1本ずつ試すまでもなく未割り当て
        if not await self._moves_model(
            vts, before, {param: 1.0 for param in MOOD_PARAMS.values()}
        ):
            await self._push(vts, idle)
            print(
                "感情パラメータ（"
                + "、".join(MOOD_PARAMS.values())
                + "）を作成しました。VTube Studioの「VTS Parameter Setup」で"
                "感情用Live2Dパラメータのinputへ割り当てると、表情の切り替えが"
                "ON/OFFではなく強度の変化になります（未割り当ての間は"
                "従来どおり表情ファイルを使います）。"
            )
            return {}

        assigned = {}
        for mood, param in MOOD_PARAMS.items():
            if await self._moves_model(vts, before, {param: 1.0}):
                assigned[mood] = param
        await self._push(vts, idle)
        print(
            "感情パラメータを強度で送ります: "
            + "、".join(f"{mood}={param}" for mood, param in assigned.items())
        )
        return assigned

    async def _create_breath_parameter(self, vts: Any) -> bool:
        """呼吸用のカスタム入力パラメータを作る

        VTube Studio の既定パラメータには呼吸用が無いので、入口を自分で用意する。
        値域はモーション側の呼吸 (0〜1) に合わせる。
        """
        response = await self._request(
            vts,
            vts.vts_request.requestCustomParameter(
                BREATH_PARAM,
                min=0.0,
                max=1.0,
                default_value=0.5,
                info="待機モーションの呼吸",
            ),
        )
        return parameter_created(response)

    async def _request_expression_state(self, vts: Any) -> Any:
        return await self._request(
            vts,
            vts.vts_request.BaseRequest(
                "ExpressionStateRequest", {"details": False}
            ),
        )

    async def _set_expression(
        self,
        vts: Any,
        file: str,
        active: bool,
        fade: float = EXPRESSION_FADE,
    ) -> None:
        """表情を明示的に ON / OFF する

        ホットキーはトグルなので取りこぼすと状態がズレるが、こちらは何度呼んでも
        指定した状態になる。
        """
        await self._request(
            vts,
            vts.vts_request.BaseRequest(
                "ExpressionActivationRequest",
                {"expressionFile": file, "active": active, "fadeTime": fade},
            ),
        )

    async def _setup_expressions(self, vts: Any) -> dict[str, str]:
        """ムードと表情ファイルの対応を作る

        コード側は表示名(「悲しみ」など)で持っておき、ここでファイル名に解決する。
        VTS 側の登録漏れや表示名の違いはここで警告する。
        """
        by_name = expression_files_by_name(
            await self._request_expression_state(vts)
        )
        expressions = resolve_mood_expressions(by_name)
        # 強度で出せるムードは表情ファイルを使わない(同じ顔を二重に動かさない)
        for mood in self.runtime["mood_params"]:
            expressions.pop(mood, None)
        self.runtime["expressions"] = expressions

        missing = missing_mood_expressions(by_name)
        if missing:
            print(
                "警告: 次の表情がVTube Studioに見つかりません: "
                + ", ".join(missing)
                + "（該当ムードは素の顔で動きます）",
                file=sys.stderr,
            )
        return self.runtime["expressions"]

    async def _reset_expressions(self, vts: Any, fade: float = 0.0) -> list[str]:
        """ONのまま残っている表情を全部OFFにする

        前回の実行が表情を出したまま落ちていることがあるので、起動時と終了時に通す。
        """
        files = active_expression_files(
            await self._request_expression_state(vts)
        )
        for file in files:
            await self._set_expression(vts, file, False, fade=fade)
        return files

    # ---- 常駐コルーチン ----

    async def _sender_loop(self, vts: Any) -> None:
        """唯一の送信担当。全パラメータを毎フレーム送る"""
        prev = time.perf_counter()
        # 首の狙い位置を滑らかに追従させるための現在値
        cur_x, cur_y = 0.0, 0.0
        saccade = Saccade()

        while True:
            now = time.perf_counter()
            dt = now - prev
            prev = now

            self.engine.update(dt)
            p = self.engine.params
            sway_x, sway_y, sway_z, breath = self.engine.angles()

            # look_x/y に近づける(イージング)。テンションが高いほど機敏に
            override = self.state["look_override"]
            if override is None:
                aim_x = self.state["look_x"] + p.look_bias_x
                aim_y = self.state["look_y"] + p.look_bias_y
            else:
                aim_x, aim_y = override
            follow = 1.0 - math.exp(-dt * LOOK_FOLLOW_RATE * p.speed)
            cur_x += (aim_x - cur_x) * follow
            cur_y += (aim_y - cur_y) * follow

            # 呼吸で頭もわずかに上下させる(モデルの呼吸パラメータとは別口の味付け)
            breath_y = (breath - 0.5) * 2.0 * BREATH_TO_Y

            eye = eye_open_value(
                self.state["eye_open"], p, self.engine.eye_open_bonus()
            )
            # 首がまだ向いていない分(aim - cur)を目で先行させ、
            # 待機の揺れ(sway)のぶんは目を逆へ回して視線を一点に留める
            eye_x = eye_look_offset(cur_x, aim_x - cur_x, sway_x)
            # 上下は首と目の向きを揃える。呼吸は目線に乗せない
            # (呼吸のたびに目玉が上下すると落ち着かない)
            look_y, eye_y = vertical_gaze(cur_y, aim_y - cur_y, sway_y)
            # 一点を見ているときも眼球は細かく飛んでいる
            saccade_x, saccade_y = saccade.update(dt, random)
            eye_x = clamp(eye_x + saccade_x, -1.0, 1.0)
            eye_y = clamp(eye_y + saccade_y, -1.0, 1.0)

            params = list(FACE_PARAMS)
            values = [
                # 揺れ・視線・仕草が重なると定格を超えうるので、ここで丸める
                clamp(sway_x + cur_x, -FACE_ANGLE_LIMIT, FACE_ANGLE_LIMIT),
                clamp(
                    look_y + (sway_y + breath_y) * LOOK_Y_SIGN,
                    -FACE_ANGLE_LIMIT,
                    FACE_ANGLE_LIMIT,
                ),
                clamp(
                    sway_z + turn_tilt(cur_x),
                    -FACE_ANGLE_LIMIT,
                    FACE_ANGLE_LIMIT,
                ),
                eye, eye,
                # 目線も首の動きに連動させると生きてる感が出る
                eye_x, eye_x, eye_y, eye_y,
                self.state["mouth_open"],
                # 眉と口角はムードの値をそのまま。送らないとVTS側の既定値0が
                # 使われ、眉が下がりきった顔になってしまう
                clamp(p.mouth_smile, 0.0, 1.0),
                clamp(p.brows, 0.0, 1.0),
            ]

            if self.runtime["breath_param"]:
                params.append(self.runtime["breath_param"])
                values.append(breath)

            if self.runtime["mood_params"]:
                intensity = self.engine.mood_param_values()
                for mood, param in self.runtime["mood_params"].items():
                    params.append(param)
                    values.append(intensity.get(mood, 0.0))

            await self._request(
                vts,
                vts.vts_request.requestSetMultiParameterValue(
                    parameters=params, values=values, face_found=True
                ),
            )

            await asyncio.sleep(1 / FPS)

    async def _blink_director(self) -> None:
        """まばたきのタイミングを決めて state を書き換えるだけ"""
        while True:
            p = self.engine.params
            if await self.wait_for_mood_change(
                random.uniform(p.blink_min, p.blink_max)
            ):
                # 感情が動いた直後は一拍おいて反応のまばたきを入れる。
                # 変化と同時に閉じると機械的に見えるので少しずらす
                await asyncio.sleep(random.uniform(*REACTION_BLINK_DELAY))

            # 落ち込み時はゆっくり、興奮時はパチパチと速く閉じる
            p = self.engine.params

            # たまに二連続まばたき
            count = 2 if random.random() < p.double_blink_chance else 1
            for _ in range(count):
                self.state["eye_open"] = 0.0
                await asyncio.sleep(blink_phase(BLINK_CLOSE, p.speed))
                self.state["eye_open"] = 0.5
                await asyncio.sleep(blink_phase(BLINK_HALF, p.speed))
                self.state["eye_open"] = 1.0
                await asyncio.sleep(blink_phase(BLINK_OPEN, p.speed))

    async def _look_director(self) -> None:
        """たまに視線・首の向きを変える(きょろきょろ / ゲーム画面のチラ見)

        ムードごとの基準位置(look_bias)は _sender_loop 側で足されるので、
        ここではその基準からのブレだけを決める。
        """
        while True:
            p = self.engine.params
            # 感情が変わったら待たずに視線を動かし直す。驚いた瞬間に画面を
            # 確認しにいく動きが、次のきょろきょろまで遅れないように
            await self.wait_for_mood_change(
                random.uniform(p.look_min, p.look_max)
            )

            if self.state["look_override"] is not None:
                # 視線を固定中は自動のきょろきょろを出さない
                continue

            p = self.engine.params
            pattern = pick_gaze_pattern(self.engine.mood, random)
            interrupted = False
            for x, y, hold in plan_gaze(pattern, p, random):
                self.state["look_x"] = x
                self.state["look_y"] = y
                # 視線を固定されたら自動の動きは引っ込める
                if self.state["look_override"] is not None:
                    interrupted = True
                    break
                # ここでも感情が動けば、途中でもパターンを組み直す
                if await self.wait_for_mood_change(hold):
                    interrupted = True
                    break
            if interrupted:
                continue
            self.state["look_x"], self.state["look_y"] = rest_look_target(
                p, random
            )

    async def _gesture_director(self) -> None:
        """たまに単発の仕草(うなずき・首かしげなど)を出す

        待機の揺れは振幅と速さしか変わらないので、そのままだとどのムードも
        同じ動きに見える。ムードごとに違う仕草を混ぜて印象を分ける。
        """
        while True:
            p = self.engine.params
            bias = None
            if await self.wait_for_mood_change(
                random.uniform(p.gesture_min, p.gesture_max)
                * gesture_interval_scale(self.engine.intensity)
            ):
                # 感情が動いた直後は、その感情らしい仕草で反応する。
                # 話し方(mode)に合わせて出す仕草も傾ける
                await asyncio.sleep(random.uniform(*REACTION_GESTURE_DELAY))
                bias = MODE_GESTURE_BIAS.get(self.engine.mode or "")
            self.engine.start_gesture(
                pick_gesture(self.engine.mood, random, bias)
            )

    async def _mouth_director(self) -> None:
        """喋っている間だけ口をパクパクさせる

        音声の再生終了は開閉サイクルの途中でも届くので、各ステップの前に
        発話中かを確かめ直す。そうしないと再生後最大0.16秒ほど口が開いたまま残る。
        """
        while True:
            if not self.state["is_speaking"]:
                self.state["mouth_open"] = 0.0
                await asyncio.sleep(0.05)
                continue

            p = self.engine.params
            pace = 1.0 / max(p.mouth_speed, 0.3)
            for amplitude in (
                random.uniform(0.6, 1.0),
                random.uniform(0.0, 0.15),
            ):
                if not self.state["is_speaking"]:
                    break
                self.state["mouth_open"] = amplitude * p.mouth_amp
                await asyncio.sleep(random.uniform(0.08, 0.16) * pace)
            if not self.state["is_speaking"]:
                self.state["mouth_open"] = 0.0

    async def _expression_director(self, vts: Any) -> None:
        """ムードが変わったら表情を切り替える

        表情を持たないムード(calm など)へ移ると、出ていた表情を消して素の顔に戻す。
        """
        current: str | None = None
        while True:
            desired = self.runtime["expressions"].get(self.engine.mood)
            if desired != current:
                if current is not None:
                    await self._set_expression(vts, current, False)
                if desired is not None:
                    await self._set_expression(vts, desired, True)
                current = desired
            await asyncio.sleep(0.1)
