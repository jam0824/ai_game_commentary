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

# 感情が変わってから反応のまばたきを入れるまでの間(秒)
REACTION_BLINK_DELAY = (0.15, 0.5)

# 呼吸を流し込むための入力パラメータ名。
# VTube Studio の既定パラメータには呼吸用が無いので、起動時にカスタム入力
# パラメータとして自動生成する。生成しただけでは何も動かず、VTube Studio の
# model config タブ内「VTS Parameter Setup」で、呼吸用 Live2D パラメータの
# input にこれを割り当てる必要がある(初回だけ)。Auto-breath は OFF にすること
BREATH_PARAM = "Breath"

# 呼吸で頭がわずかに上下する量。モデル側の呼吸パラメータと二重に効くので控えめに
BREATH_TO_Y = 0.3

# 毎フレーム送る顔まわりの入力パラメータ。値の並び順と対応している
FACE_PARAMS = (
    "FaceAngleX", "FaceAngleY", "FaceAngleZ",
    "EyeOpenLeft", "EyeOpenRight",
    "EyeLeftX", "EyeRightX", "EyeLeftY", "EyeRightY",
    "MouthOpen",
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
    ),
}


def clamp(value, low, high):
    return max(low, min(high, value))


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


def pick_look_target(params: MoodParams, rng) -> tuple[float, float, float]:
    """次に視線を向ける先と、そこを見ている秒数を決める

    たまにゲーム画面(向かって左上)をチラ見させると、
    「実況者が画面を見ている」動きになって視線に意味が出る。
    rng を引数に取るのは、テストで揺らぎを固定できるようにするため。
    """
    if rng.random() < params.screen_glance_chance:
        x = SCREEN_LOOK_X + rng.uniform(-SCREEN_JITTER, SCREEN_JITTER)
        y = SCREEN_LOOK_Y + rng.uniform(-SCREEN_JITTER, SCREEN_JITTER)
        hold = rng.uniform(params.screen_hold * 0.6, params.screen_hold * 1.4)
        return x, y, hold

    x = rng.uniform(-params.look_range_x, params.look_range_x)
    y = rng.uniform(-params.look_range_y, params.look_range_y)
    hold = rng.uniform(2.0, 4.0) / max(params.speed, 0.3)
    return x, y, hold


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
        p.sway_x * math.sin(phase * 2 * math.pi / 7.3)
        + p.sub_x * math.sin(phase * 2 * math.pi / 3.1)
        + p.tremor * math.sin(phase * 2 * math.pi / 0.19)
    )
    y = (
        p.sway_y * math.sin(phase * 2 * math.pi / 5.7)
        + p.sub_y * math.sin(phase * 2 * math.pi / 2.3)
        + p.tremor * 0.7 * math.sin(phase * 2 * math.pi / 0.23)
    )
    z = (
        p.sway_z * math.sin(phase * 2 * math.pi / 9.1)
        + p.sub_z * math.sin(phase * 2 * math.pi / 4.7)
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

    def update(self, dt: float) -> None:
        """dt 秒ぶん時間を進める。パラメータ補間と位相の更新はここだけ"""
        # 一時ムード(驚きなど)の残り時間を消化して元へ戻す
        if self._hold_left > 0.0:
            self._hold_left -= dt
            if self._hold_left <= 0.0:
                self._hold_left = 0.0
                self.mood = self._prev_mood

        # 目標ムードへ滑らかに寄せる。切り替えの瞬間にガクッと動かないように
        target = MOODS[self.mood]
        self.params = lerp_params(
            self.params, target, 1.0 - math.exp(-dt / MOOD_BLEND_TAU)
        )

        if self._burst_t is not None:
            self._burst_t += dt
            if self._burst_t >= SURPRISE_BURST:
                self._burst_t = None

        env = self.burst_strength()
        self.phase += dt * self.params.speed
        # 驚いた瞬間は息を止める
        self.breath_phase += dt / self.params.breath_period * (1.0 - env)

    def burst_strength(self) -> float:
        if self._burst_t is None:
            return 0.0
        return surprise_envelope(self._burst_t / SURPRISE_BURST)

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
        return x, y, z, breath


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

    def set_emotion(self, emotion: str) -> None:
        """実況プランのemotionをモーションへ反映する"""
        mood = resolve_emotion_mood(emotion)
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self.engine.set_mood, mood)
        else:
            self.engine.set_mood(mood)

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
        """音声再生中フラグ。Trueの間だけ口パクする"""
        self.state["is_speaking"] = bool(speaking)
        if not speaking:
            self.state["mouth_open"] = 0.0

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
            await self._setup_parameters(vts)
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
        self.runtime["expressions"] = resolve_mood_expressions(by_name)

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

            params = list(FACE_PARAMS)
            values = [
                sway_x + cur_x,
                look_y + (sway_y + breath_y) * LOOK_Y_SIGN,
                sway_z + turn_tilt(cur_x),
                eye, eye,
                # 目線も首の動きに連動させると生きてる感が出る
                eye_x, eye_x, eye_y, eye_y,
                self.state["mouth_open"],
            ]

            if self.runtime["breath_param"]:
                params.append(self.runtime["breath_param"])
                values.append(breath)

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
            x, y, hold = pick_look_target(p, random)
            self.state["look_x"] = x
            self.state["look_y"] = y

            # しばらくしたら待機位置に戻る。ここでも感情が動けば見直す
            if await self.wait_for_mood_change(hold):
                continue
            self.state["look_x"], self.state["look_y"] = rest_look_target(
                p, random
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
