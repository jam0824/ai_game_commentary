import asyncio
import math
import random
import time
from dataclasses import dataclass, fields, replace

import pyvts

plugin_info = {
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
SURPRISE_EYE = 0.45    # 目の見開き量
SURPRISE_TILT = 5.0    # 首がカクッと傾く量(Z)
SURPRISE_LIFT = 3.0    # 一瞬のけぞる量(Y)

# FaceAngleY の符号。モデルによって上下が逆なら -1.0 にする
LOOK_Y_SIGN = 1.0

# ゲーム画面の方向。キャラは配信画面の右下に立つので、画面は「向かって左上」にある。
# 逆を向いたら符号を反転する
SCREEN_LOOK_X = -14.0  # 左
SCREEN_LOOK_Y = 7.0    # 上
SCREEN_JITTER = 2.0    # チラ見のたびに少しズラす(毎回同じ点を見ると機械的)

# 首の狙い位置に追いつく速さ。大きいほどキビキビ視線が動く
LOOK_FOLLOW_RATE = 2.2

# 首の角度を目線に変換する係数。
# 人は視線を動かすとき目が先に動き、首が後から追いつく。EYE_LEAD が「まだ首が
# 向いていない分を目で先行する量」、EYE_HOLD が「首の向きに合わせた定常のズレ」。
EYE_LEAD = 0.075
EYE_HOLD = 0.030

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
    eye_open_base: float   # 目の開きの上限(1.0未満で半目)
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


MOOD_NAMES = (
    "calm",
    "amused",
    "excited",
    "surprised",
    "tense",
    "thoughtful",
    "sad",
)

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
    # 平常。今までのデフォルト待機がこれ
    "calm": MoodParams(
        sway_x=2.6, sway_y=1.8, sway_z=2.2,
        sub_x=1.2, sub_y=0.8, sub_z=0.9,
        speed=1.0, tremor=0.0, breath_period=4.0, breath_depth=1.0,
        blink_min=2.0, blink_max=6.0, double_blink_chance=0.25,
        eye_open_base=1.0,
        look_range_x=7.0, look_range_y=3.0,
        look_bias_x=0.0, look_bias_y=0.0,
        look_min=4.0, look_max=10.0,
        screen_glance_chance=0.35, screen_hold=2.5,
        head_tilt=0.0, mouth_speed=1.0, mouth_amp=1.0,
    ),
    # 笑ってる・楽しんでる。少し弾む
    "amused": MoodParams(
        sway_x=3.2, sway_y=2.4, sway_z=3.0,
        sub_x=1.5, sub_y=1.2, sub_z=1.3,
        speed=1.3, tremor=0.0, breath_period=3.2, breath_depth=1.0,
        blink_min=1.5, blink_max=4.0, double_blink_chance=0.45,
        eye_open_base=1.0,
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
        eye_open_base=1.0,
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
        eye_open_base=1.0,
        look_range_x=2.0, look_range_y=1.0,
        look_bias_x=0.0, look_bias_y=1.0,
        look_min=1.5, look_max=3.0,
        # 何が起きた?と真っ先に画面を確認する
        screen_glance_chance=0.7, screen_hold=1.5,
        head_tilt=0.0, mouth_speed=1.4, mouth_amp=1.0,
    ),
    # 緊張・集中。画面から目を離さず、細かく震える
    "tense": MoodParams(
        sway_x=1.2, sway_y=0.8, sway_z=1.0,
        sub_x=0.5, sub_y=0.35, sub_z=0.4,
        speed=0.85, tremor=0.35, breath_period=2.4, breath_depth=0.6,
        blink_min=1.5, blink_max=3.2, double_blink_chance=0.3,
        eye_open_base=0.95,
        look_range_x=2.5, look_range_y=1.2,
        look_bias_x=0.0, look_bias_y=-0.5,
        look_min=3.0, look_max=6.0,
        # 画面から目を離さない
        screen_glance_chance=0.85, screen_hold=5.0,
        head_tilt=0.0, mouth_speed=1.1, mouth_amp=0.85,
    ),
    # 考え込み。首をかしげて斜め上を見がち
    "thoughtful": MoodParams(
        sway_x=1.6, sway_y=1.1, sway_z=1.3,
        sub_x=0.7, sub_y=0.5, sub_z=0.6,
        speed=0.6, tremor=0.0, breath_period=4.8, breath_depth=0.9,
        blink_min=3.0, blink_max=7.0, double_blink_chance=0.15,
        eye_open_base=0.95,
        look_range_x=4.0, look_range_y=2.5,
        look_bias_x=-1.8, look_bias_y=2.6,
        look_min=5.0, look_max=11.0,
        # 画面より宙を見て考え込む
        screen_glance_chance=0.25, screen_hold=2.0,
        head_tilt=6.0, mouth_speed=0.85, mouth_amp=0.8,
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


def resolve_mood(name):
    """未知のムード名が来ても落ちないように calm へフォールバックする"""
    return MOODS.get(name, MOODS["calm"])


def lerp_params(a, b, ratio):
    """MoodParams 同士を線形補間する(切り替えをガクつかせないため)"""
    inv = 1.0 - ratio
    return MoodParams(
        **{
            f.name: getattr(a, f.name) * inv + getattr(b, f.name) * ratio
            for f in fields(MoodParams)
        }
    )


def surprise_envelope(u):
    """驚きバーストの強さ。u=0〜1 で、立ち上がりが鋭く後は減衰する"""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    if u < 0.10:
        return u / 0.10
    return math.exp(-(u - 0.10) * 4.0)


def eye_look_offset(head, residual):
    """目線のオフセット

    head は今の首の向き、residual は狙いに対して首がまだ向いていない残りの角度。
    人は視線を動かすとき目が先に動いて首が後から追いつくので、residual が大きい
    振り向きはじめに目を大きく振り、首が追いついたらズレを小さくする。
    VTS の Eye*X / Eye*Y は -1〜1 なので、はみ出さないよう丸める。
    """
    return clamp(head * EYE_HOLD + residual * EYE_LEAD, -1.0, 1.0)


def input_parameter_names(response):
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


def pick_look_target(params, rng):
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


def rest_look_target(params, rng):
    """視線を戻す先

    画面をよく見るムードほど正面ではなく画面寄りで待機させる。
    そうしないとチラ見のあと毎回カメラ目線に戻ってしまい、
    「画面に集中している」感じが消える。
    """
    lean = params.screen_glance_chance * 0.5
    x = SCREEN_LOOK_X * lean + rng.uniform(-1.5, 1.5)
    y = SCREEN_LOOK_Y * lean + rng.uniform(-1.0, 1.0)
    return x, y


def idle_angles(phase, params, breath_phase=None):
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


class MoodEngine:
    """現在のムードを保持し、毎フレーム分の待機モーションを計算する"""

    def __init__(self, mood="calm"):
        self.mood = mood if mood in MOODS else "calm"
        self.params = MOODS[self.mood]
        self.phase = 0.0
        self.breath_phase = 0.0
        self._prev_mood = self.mood
        self._hold_left = 0.0
        self._burst_t = None

    def set_mood(self, name, hold=None):
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

    def update(self, dt):
        """dt 秒ぶん時間を進める。パラメータ補間と位相の更新はここだけ"""
        # 一時ムード(驚きなど)の残り時間を消化して元へ戻す
        if self._hold_left > 0.0:
            self._hold_left -= dt
            if self._hold_left <= 0.0:
                self._hold_left = 0.0
                self.mood = self._prev_mood

        # 目標ムードへ滑らかに寄せる。切り替えの瞬間にガクッと動かないように
        target = MOODS[self.mood]
        self.params = lerp_params(self.params, target, 1.0 - math.exp(-dt / MOOD_BLEND_TAU))

        if self._burst_t is not None:
            self._burst_t += dt
            if self._burst_t >= SURPRISE_BURST:
                self._burst_t = None

        env = self.burst_strength()
        self.phase += dt * self.params.speed
        # 驚いた瞬間は息を止める
        self.breath_phase += dt / self.params.breath_period * (1.0 - env)

    def burst_strength(self):
        if self._burst_t is None:
            return 0.0
        return surprise_envelope(self._burst_t / SURPRISE_BURST)

    def eye_open_bonus(self):
        """驚いた直後だけ目を見開く量"""
        return SURPRISE_EYE * self.burst_strength()

    def angles(self):
        """この瞬間の (x, y, z, breath) を返す"""
        x, y, z, breath = idle_angles(self.phase, self.params, self.breath_phase)
        env = self.burst_strength()
        if env > 0.0:
            z += SURPRISE_TILT * env
            y += SURPRISE_LIFT * env
        return x, y, z, breath


# ---- 状態(各コルーチンはここを書き換えるだけ) ----
engine = MoodEngine("calm")

state = {
    "eye_open": 1.0,     # 0.0=閉じる 1.0=開く(まばたき用。ムードの半目とは別)
    "mouth_open": 0.0,   # 0.0=閉じる 1.0=開く
    "is_speaking": False,
    "look_x": 0.0,       # 視線・首の狙い位置(ゆっくり動く目標値)
    "look_y": 0.0,
}

vts_lock = asyncio.Lock()  # ホットキー等、別系統のリクエスト用

# 起動時の確認で埋まる。モデル/VTS に無いものは送らない
runtime = {
    "breath_param": None,   # 呼吸パラメータ名(見つからなければ None)
    "expressions": {},      # ムード名 -> 表情ファイル
}


def set_mood(name, hold=None):
    """外(実況ツール側)から感情を伝えるための入口"""
    engine.set_mood(name, hold)


async def sender_loop(vts):
    """唯一の送信担当。全パラメータを毎フレーム送る"""
    prev = time.perf_counter()
    # 首の狙い位置を滑らかに追従させるための現在値
    cur_x, cur_y = 0.0, 0.0

    while True:
        now = time.perf_counter()
        dt = now - prev
        prev = now

        engine.update(dt)
        p = engine.params
        sway_x, sway_y, sway_z, breath = engine.angles()

        # look_x/y に近づける(イージング)。テンションが高いほど機敏に
        aim_x = state["look_x"] + p.look_bias_x
        aim_y = state["look_y"] + p.look_bias_y
        follow = 1.0 - math.exp(-dt * LOOK_FOLLOW_RATE * p.speed)
        cur_x += (aim_x - cur_x) * follow
        cur_y += (aim_y - cur_y) * follow

        # 呼吸で頭もわずかに上下させる(モデルの呼吸パラメータとは別口の味付け)
        breath_y = (breath - 0.5) * 2.0 * BREATH_TO_Y

        eye = clamp(state["eye_open"] * p.eye_open_base + engine.eye_open_bonus(), 0.0, 1.0)
        # 首がまだ向いていない分(aim - cur)を目で先行させる
        eye_x = eye_look_offset(cur_x, aim_x - cur_x)
        eye_y = eye_look_offset(cur_y, aim_y - cur_y)

        params = list(FACE_PARAMS)
        values = [
            sway_x + cur_x,
            (sway_y + cur_y + breath_y) * LOOK_Y_SIGN,
            sway_z,
            eye, eye,
            # 目線も首の動きに連動させると生きてる感が出る
            eye_x, eye_x, eye_y, eye_y,
            state["mouth_open"],
        ]

        if runtime["breath_param"]:
            params.append(runtime["breath_param"])
            values.append(breath)

        async with vts_lock:
            await vts.request(vts.vts_request.requestSetMultiParameterValue(
                parameters=params, values=values, face_found=True
            ))

        await asyncio.sleep(1 / FPS)


async def blink_director():
    """まばたきのタイミングを決めて state を書き換えるだけ"""
    while True:
        p = engine.params
        await asyncio.sleep(random.uniform(p.blink_min, p.blink_max))

        # 落ち込み時はゆっくり、興奮時はパチパチと速く閉じる
        p = engine.params
        pace = 1.0 / max(p.speed, 0.3)

        # たまに二連続まばたき
        count = 2 if random.random() < p.double_blink_chance else 1
        for _ in range(count):
            state["eye_open"] = 0.0
            await asyncio.sleep(0.07 * pace)
            state["eye_open"] = 0.5
            await asyncio.sleep(0.04 * pace)
            state["eye_open"] = 1.0
            await asyncio.sleep(0.12 * pace)


async def look_director():
    """たまに視線・首の向きを変える(きょろきょろ / ゲーム画面のチラ見)

    ムードごとの基準位置(look_bias)は sender_loop 側で足されるので、
    ここではその基準からのブレだけを決める。
    """
    while True:
        p = engine.params
        await asyncio.sleep(random.uniform(p.look_min, p.look_max))

        p = engine.params
        x, y, hold = pick_look_target(p, random)
        state["look_x"] = x
        state["look_y"] = y

        # しばらくしたら待機位置に戻る
        await asyncio.sleep(hold)
        state["look_x"], state["look_y"] = rest_look_target(p, random)


async def mouth_director():
    """喋っている間だけ口をパクパクさせる"""
    while True:
        if state["is_speaking"]:
            p = engine.params
            pace = 1.0 / max(p.mouth_speed, 0.3)
            state["mouth_open"] = random.uniform(0.6, 1.0) * p.mouth_amp
            await asyncio.sleep(random.uniform(0.08, 0.16) * pace)
            state["mouth_open"] = random.uniform(0.0, 0.15) * p.mouth_amp
            await asyncio.sleep(random.uniform(0.08, 0.16) * pace)
        else:
            state["mouth_open"] = 0.0
            await asyncio.sleep(0.1)


def expression_files_by_name(response):
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


def resolve_mood_expressions(by_name):
    """ムード名 -> 表情ファイル の対応を作る

    VTS 側に登録されていない表情は黙って外す。1つ足りないだけで
    他のムードまで表情なしになると原因が分かりにくいので。
    """
    return {
        mood: by_name[label]
        for mood, label in MOOD_EXPRESSIONS.items()
        if label in by_name
    }


def missing_mood_expressions(by_name):
    """VTS 側に見つからなかった表情の表示名(起動時の警告用)"""
    return [label for label in MOOD_EXPRESSIONS.values() if label not in by_name]


def active_expression_files(response):
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


def parameter_created(response):
    """ParameterCreationRequest が成功したかどうか"""
    return (
        isinstance(response, dict)
        and response.get("messageType") == "ParameterCreationResponse"
    )


async def create_breath_parameter(vts):
    """呼吸用のカスタム入力パラメータを作る

    VTube Studio の既定パラメータには呼吸用が無いので、入口を自分で用意する。
    値域はモーション側の呼吸 (0〜1) に合わせる。
    """
    async with vts_lock:
        response = await vts.request(vts.vts_request.requestCustomParameter(
            BREATH_PARAM,
            min=0.0,
            max=1.0,
            default_value=0.5,
            info="待機モーションの呼吸",
        ))
    return parameter_created(response)


async def setup_parameters(vts):
    """入力パラメータを確認し、呼吸用が無ければ作る"""
    async with vts_lock:
        response = await vts.request(vts.vts_request.requestTrackingParameterList())

    names = input_parameter_names(response)
    if not names:
        print("警告: 入力パラメータ一覧を取得できませんでした")
        return names

    if BREATH_PARAM in names:
        runtime["breath_param"] = BREATH_PARAM
        print(f"呼吸パラメータ「{BREATH_PARAM}」を使います")
    elif await create_breath_parameter(vts):
        runtime["breath_param"] = BREATH_PARAM
        names = names | {BREATH_PARAM}
        print(
            f"呼吸パラメータ「{BREATH_PARAM}」を作成しました。\n"
            "  → VTube Studio の model config タブ内「VTS Parameter Setup」で、\n"
            f"    呼吸用 Live2D パラメータ(例: ParamBreath)の input に「{BREATH_PARAM}」を割り当ててください\n"
            "    (Auto-breath が ON だと二重に動くので、割り当てたら OFF にしてください)。\n"
            "    割り当てるまで呼吸は動きません"
        )
    else:
        runtime["breath_param"] = None
        print(
            f"警告: 呼吸パラメータ「{BREATH_PARAM}」を作成できませんでした。"
            "頭の上下だけで呼吸を表現します"
        )

    missing = [p for p in FACE_PARAMS if p not in names]
    if missing:
        print(f"警告: 次の入力パラメータが見つかりません: {', '.join(missing)}")

    return names


async def request_expression_state(vts):
    async with vts_lock:
        return await vts.request(
            vts.vts_request.BaseRequest("ExpressionStateRequest", {"details": False})
        )


async def set_expression(vts, file, active, fade=EXPRESSION_FADE):
    """表情を明示的に ON / OFF する

    ホットキーはトグルなので取りこぼすと状態がズレるが、こちらは何度呼んでも
    指定した状態になる。
    """
    async with vts_lock:
        await vts.request(vts.vts_request.BaseRequest(
            "ExpressionActivationRequest",
            {"expressionFile": file, "active": active, "fadeTime": fade},
        ))


async def setup_expressions(vts):
    """ムードと表情ファイルの対応を作る

    コード側は表示名(「悲しみ」など)で持っておき、ここでファイル名に解決する。
    VTS 側の登録漏れや表示名の違いはここで警告する。
    """
    by_name = expression_files_by_name(await request_expression_state(vts))
    runtime["expressions"] = resolve_mood_expressions(by_name)

    for mood, file in runtime["expressions"].items():
        print(f"{mood} -> 表情「{MOOD_EXPRESSIONS[mood]}」({file})")

    missing = missing_mood_expressions(by_name)
    if missing:
        print(
            f"警告: 次の表情が VTube Studio に見つかりません: {', '.join(missing)}"
            "(該当ムードは素の顔で動きます)"
        )
    return runtime["expressions"]


async def expression_director(vts):
    """ムードが変わったら表情を切り替える

    表情を持たないムード(calm など)へ移ると、出ていた表情を消して素の顔に戻す。
    """
    current = None
    while True:
        desired = runtime["expressions"].get(engine.mood)
        if desired != current:
            if current is not None:
                await set_expression(vts, current, False)
            if desired is not None:
                await set_expression(vts, desired, True)
                print(f"表情 -> {MOOD_EXPRESSIONS.get(engine.mood, engine.mood)}")
            current = desired
        await asyncio.sleep(0.1)


async def reset_expressions(vts):
    """ONのまま残っている表情を全部OFFにする

    前回の実行が表情を出したまま落ちていることがあるので、起動時と終了時に通す。
    """
    files = active_expression_files(await request_expression_state(vts))
    if not files:
        print("ONになっている表情はありません")
        return files

    for file in files:
        # 終了時のリセットはフェードを待たずに落としたいので即時
        await set_expression(vts, file, False, fade=0.0)
        print(f"表情をリセット: {file}")
    return files


async def trigger_hotkey(vts, name, hold_time=None):
    """表情ホットキーを発動(hold_time を渡すとその秒数後に解除)"""
    async with vts_lock:
        response = await vts.request(vts.vts_request.requestHotKeyList())
    target = next(
        (h for h in response["data"]["availableHotkeys"] if h["name"] == name), None
    )
    if target is None:
        print(f"ホットキー「{name}」が見つかりません")
        return

    async with vts_lock:
        await vts.request(vts.vts_request.requestTriggerHotKey(target["hotkeyID"]))
    print(f"{name} ON")

    if hold_time is not None:
        await asyncio.sleep(hold_time)
        async with vts_lock:
            await vts.request(vts.vts_request.requestTriggerHotKey(target["hotkeyID"]))
        print(f"{name} OFF")


# ---- 動作確認用 ----
async def demo(vts):
    """各ムードを順に切り替えて見た目を確認する

    表情は expression_director がムードに追従して切り替えるので、
    ここで set_mood を呼ぶだけでよい。
    """
    await asyncio.sleep(3)
    state["is_speaking"] = True

    for mood in ("calm", "amused", "excited", "tense", "thoughtful", "sad"):
        set_mood(mood)
        print(f"mood -> {mood}")
        await asyncio.sleep(8)

    set_mood("calm")
    await asyncio.sleep(2)
    print("mood -> surprised (数秒で自動的に calm へ戻る)")
    set_mood("surprised")

    await asyncio.sleep(5)
    state["is_speaking"] = False
    print("デモ終了")


async def connect_vts():
    vts = pyvts.vts(plugin_info=plugin_info)
    await vts.connect()
    await vts.request_authenticate_token()
    await vts.request_authenticate()
    return vts


async def main():
    vts = await connect_vts()

    # 前回の実行が表情を出したまま落ちている場合があるので、まず素の顔に戻す
    await reset_expressions(vts)
    await setup_parameters(vts)
    await setup_expressions(vts)

    print("接続成功。ムード連動の待機モーションを開始します(Ctrl+Cで停止)")

    await asyncio.gather(
        sender_loop(vts),
        blink_director(),
        look_director(),
        mouth_director(),
        expression_director(vts),
        demo(vts),
    )


async def cleanup():
    """Ctrl+C 後の後始末。イベントループが畳まれた後なので繋ぎ直す"""
    vts = await connect_vts()
    try:
        await reset_expressions(vts)
    finally:
        await vts.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n停止します。表情をリセットします")
        asyncio.run(cleanup())
