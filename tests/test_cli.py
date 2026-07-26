import json

import pytest
from PIL import Image, ImageDraw

import game_window_ocr.commentary as commentary_module
import game_window_ocr.windows as windows_module
from game_window_ocr.cli import clean_ocr_text, parse_crop, prepare_ocr_image
from game_window_ocr.commentary import (
    ChoiceOption,
    ChoicePlan,
    CommentaryPlan,
    SpeechResult,
    TextResult,
    apply_commentary_intensity_boost,
    build_choice_prompt,
    build_commentary_prompt,
    build_commentary_speech_prompt,
    build_narration_prompt,
    choice_plan_issue,
    choice_utterance,
    collapse_visual_line_breaks,
    commentary_plan_issue,
    detect_advance_marker,
    extract_choice_options,
    extract_incremental_text,
    narration_matches,
    parse_choice_plan,
    parse_commentary_plan,
    speak_choice_with_retries,
    verify_choice_speech,
)
from game_window_ocr.windows import normalize_title


def test_title_normalization_matches_ascii_and_full_width() -> None:
    assert normalize_title("かまいたちの夜x3") == normalize_title(
        "かまいたちの夜×３"
    )


def test_select_choice_sends_down_for_index_then_enter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys: list[tuple[int, int, bool]] = []
    monkeypatch.setattr(windows_module, "activate_window", lambda hwnd, **kwargs: None)
    monkeypatch.setattr(
        windows_module,
        "_post_key",
        lambda hwnd, virtual_key, scan_code, *, extended=False: keys.append(
            (virtual_key, scan_code, extended)
        ),
    )
    monkeypatch.setattr(windows_module.time, "sleep", lambda seconds: None)

    windows_module.select_choice(123, 2)

    assert keys == [
        (windows_module.VK_DOWN, windows_module.DOWN_SCAN_CODE, True),
        (windows_module.VK_DOWN, windows_module.DOWN_SCAN_CODE, True),
        (windows_module.VK_RETURN, windows_module.ENTER_SCAN_CODE, False),
    ]


def test_parse_crop() -> None:
    assert parse_crop("10, 20, 300, 200") == (10, 20, 300, 200)


def test_obs_capture_window_is_enabled_by_default() -> None:
    args = commentary_module._build_parser().parse_args([])
    assert args.no_obs_window is False


def test_obs_capture_window_can_be_disabled() -> None:
    args = commentary_module._build_parser().parse_args(["--no-obs-window"])
    assert args.no_obs_window is True


def test_obs_capture_window_opens_before_ocr_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []

    class FakeObsWindow:
        error = None
        is_open = True

        def __init__(self, *, enabled: bool) -> None:
            assert enabled is True

        def __enter__(self):
            events.append("obs_window")
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            events.append("obs_window_closed")

    class StoppingOcr:
        def __init__(self) -> None:
            events.append("ocr_initialization")
            raise RuntimeError("テスト終了")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(commentary_module, "ObsCaptureWindow", FakeObsWindow)
    monkeypatch.setattr(commentary_module, "PersistentNdlOcr", StoppingOcr)
    monkeypatch.setattr(
        commentary_module,
        "find_window",
        lambda title: commentary_module.WindowInfo(1, title, 960, 540),
    )

    result = commentary_module.main(["--output", str(tmp_path)])

    assert result == 1
    assert events == [
        "obs_window",
        "ocr_initialization",
        "obs_window_closed",
    ]


def test_prepare_ocr_image_crops_and_scales() -> None:
    image = Image.new("RGB", (640, 360))
    prepared = prepare_ocr_image(
        image,
        crop=(10, 20, 310, 220),
        scale=2,
    )
    assert prepared.size == (600, 400)


def test_prepare_ocr_image_rejects_out_of_bounds_crop() -> None:
    image = Image.new("RGB", (640, 360))
    with pytest.raises(ValueError):
        prepare_ocr_image(image, crop=(0, 0, 641, 360), scale=1)


def test_clean_ocr_text_filters_low_confidence_cursor(tmp_path) -> None:
    result = {
        "contents": [
            [
                {
                    "text": "止まった。",
                    "confidence": 0.768,
                    "isTextline": "true",
                },
                {
                    "text": "44",
                    "confidence": 0.432,
                    "isTextline": "true",
                },
            ]
        ]
    }
    json_path = tmp_path / "capture.json"
    json_path.write_text(json.dumps(result), encoding="utf-8")
    assert clean_ocr_text(json_path, min_confidence=0.5) == "止まった。"


def test_narration_prompt_wraps_source_as_verbatim_json() -> None:
    prompt = build_narration_prompt("ぼくの名前は、透。")
    assert '"response_text": "ぼくの名前は、透。"' in prompt
    assert '"require_repeat_verbatim": true' in prompt


def test_narration_prompt_collapses_visual_line_wraps() -> None:
    assert collapse_visual_line_breaks("そういう意味で\nも、ま、太った") == (
        "そういう意味でも、ま、太った"
    )
    prompt = build_narration_prompt("そういう意味で\nも、ま、太った")
    assert '"response_text": "そういう意味でも、ま、太った"' in prompt


def test_commentary_prompt_limits_length_and_repetition() -> None:
    prompt = build_commentary_prompt(
        "雪が降っていた。",
        page_text="山道を歩いていた。雪が降っていた。",
        advance_marker="book",
        page_has_spoken=False,
        must_speak=True,
    )
    assert '"new_game_text": "雪が降っていた。"' in prompt
    assert "silent" in prompt
    assert "reaction" in prompt
    assert "8～35文字" in prompt
    assert "最大2文・合計90文字" in prompt
    assert "毎画面しゃべる必要はありません" in prompt
    assert '"mode":"silent"' in prompt
    assert "同じ相づち・冒頭・語尾を連続させず" in prompt
    assert "triangleではquickを使わない" in prompt
    assert '"advance_marker": "book"' in prompt
    assert '"must_speak": true' in prompt
    assert "20代くらいの女性" in prompt
    assert "短めの2～3文・合計28～90文字" in prompt
    assert "頼りになりそうだよね" in prompt


def test_extract_choice_options_keeps_multiline_option_text() -> None:
    text = (
        "〉〉\n"
        "-A:「な、何言ってんだよ!」\n"
        "ぼくはドギマギした。\n"
        "B:「愚問だよ、ハニー」\n"
        "ぼくは指を振った。\n"
        "C:「待ってて、じっくり吟味するから」\n"
        "ぼくは視線を向けた。"
    )
    assert extract_choice_options(text) == (
        ChoiceOption(
            label="A",
            text="「な、何言ってんだよ!」ぼくはドギマギした。",
        ),
        ChoiceOption(
            label="B",
            text="「愚問だよ、ハニー」ぼくは指を振った。",
        ),
        ChoiceOption(
            label="C",
            text="「待ってて、じっくり吟味するから」ぼくは視線を向けた。",
        ),
    )


def test_extract_choice_options_accepts_full_width_labels() -> None:
    assert extract_choice_options("Ａ：「一つ目」\nＢ：「二つ目」") == (
        ChoiceOption("A", "「一つ目」"),
        ChoiceOption("B", "「二つ目」"),
    )


def test_extract_choice_options_rejects_non_contiguous_labels() -> None:
    assert extract_choice_options("A: 一つ目\nC: 三つ目") == ()


def test_choice_plan_and_utterance_require_opinion_before_declaration() -> None:
    options = (
        ChoiceOption("A", "正直に答える"),
        ChoiceOption("B", "強がって答える"),
        ChoiceOption("C", "考え込む"),
    )
    plan = parse_choice_plan(
        '{"selected_label":"Ｂ","opinion":"強がった後の反応が面白そうだよね。",'
        '"emotion":"amused","intensity":0.7,"pace":"normal"}'
    )
    assert plan == ChoicePlan(
        selected_label="B",
        opinion="強がった後の反応が面白そうだよね。",
        emotion="amused",
        intensity=0.7,
        pace="normal",
    )
    assert choice_plan_issue(plan, options) is None
    assert choice_utterance(plan) == (
        "強がった後の反応が面白そうだよね。 ここはBを選ぶね。"
    )


def test_choice_plan_rejects_unknown_label_and_embedded_declaration() -> None:
    options = (
        ChoiceOption("A", "一つ目"),
        ChoiceOption("B", "二つ目"),
    )
    unknown = ChoicePlan("C", "こっちの続きが気になるよね。", "calm", 0.6, "normal")
    declared = ChoicePlan("B", "面白そうだからBにするね。", "amused", 0.7, "normal")
    assert "表示中の選択肢" in (choice_plan_issue(unknown, options) or "")
    assert "選択宣言" in (choice_plan_issue(declared, options) or "")


def test_choice_speech_accepts_orthographic_variation_from_transcript() -> None:
    plan = ChoicePlan(
        "B",
        "このキザなノリ、逆に笑えていい味出てる。",
        "amused",
        0.7,
        "normal",
    )
    verification = verify_choice_speech(
        plan,
        "このキザなノリ、逆に笑えて良い味出てる。ここはBを選ぶね。",
    )
    assert verification.matches is True
    assert verification.exact is False
    assert verification.similarity == pytest.approx(1.0)
    assert verification.selection_declared is True


def test_choice_speech_rejects_wrong_selection_or_missing_opinion() -> None:
    plan = ChoicePlan(
        "B",
        "この返事の後が気になるよね。",
        "thoughtful",
        0.6,
        "normal",
    )
    wrong = verify_choice_speech(
        plan,
        "この返事の後が気になるよね。ここはCを選ぶね。",
    )
    no_opinion = verify_choice_speech(plan, "ここはBを選ぶね。")
    assert wrong.matches is False
    assert "選択先B" in (wrong.reason or "")
    assert no_opinion.matches is False
    assert "意見" in (no_opinion.reason or "")


def test_choice_speech_retries_until_selection_is_confirmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    plan = ChoicePlan(
        "B",
        "この返事の後が気になるよね。",
        "thoughtful",
        0.6,
        "normal",
    )
    transcripts = iter(
        [
            "この返事の後が気になるよね。ここはCを選ぶね。",
            "この返事の後が気になるよね。ここはBを選ぶね。",
        ]
    )

    class FakeRealtime:
        def speak(self, **kwargs) -> SpeechResult:
            return SpeechResult(
                phase=str(kwargs["phase"]),
                transcript=next(transcripts),
                audio_bytes=10,
                response_id=None,
            )

    monkeypatch.setattr(commentary_module.time, "sleep", lambda seconds: None)
    speech, verification, retries = speak_choice_with_retries(
        FakeRealtime(),
        plan,
        turn_dir=tmp_path,
        playback=False,
        retries=2,
        retry_delay=0.5,
    )
    assert speech.transcript.endswith("Bを選ぶね。")
    assert verification.matches is True
    assert retries == 1
    assert (tmp_path / "choice_transcript.txt").exists()
    assert (tmp_path / "choice_retry_001_transcript.txt").exists()


def test_choice_prompt_contains_only_available_labels() -> None:
    prompt = build_choice_prompt(
        (
            ChoiceOption("A", "一つ目"),
            ChoiceOption("B", "二つ目"),
        )
    )
    assert "A, B" in prompt
    assert '"label": "A"' in prompt
    assert '"label": "B"' in prompt
    assert "選択宣言はプログラムが" in prompt


def test_main_speaks_choice_before_sending_selection_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []
    choice_text = "A: 正直に答える\nB: 強がって答える\nC: じっくり考える"
    expected_utterance = (
        "意外な返事のほうが面白い展開になりそうだよね。 "
        "ここはCを選ぶね。"
    )

    class FakeObsWindow:
        error = None
        is_open = False

        def __init__(self, *, enabled: bool) -> None:
            assert enabled is False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

    class FakeRealtimeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def record_game_text(self, *, turn_number: int, text: str) -> None:
            events.append("record")

        def generate_text(self, **kwargs) -> TextResult:
            events.append("plan")
            return TextResult(
                text=(
                    '{"selected_label":"C",'
                    '"opinion":"意外な返事のほうが面白い展開になりそうだよね。",'
                    '"emotion":"amused","intensity":0.7,"pace":"normal"}'
                ),
                response_id="choice-plan",
            )

        def speak(self, **kwargs) -> SpeechResult:
            events.append("speak")
            return SpeechResult(
                phase="choice",
                transcript=expected_utterance,
                audio_bytes=100,
                response_id="choice-speech",
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(commentary_module, "ObsCaptureWindow", FakeObsWindow)
    monkeypatch.setattr(
        commentary_module,
        "RealtimeSpeechClient",
        FakeRealtimeClient,
    )
    monkeypatch.setattr(
        commentary_module,
        "find_window",
        lambda title: commentary_module.WindowInfo(123, title, 960, 540),
    )
    monkeypatch.setattr(
        commentary_module,
        "select_choice",
        lambda hwnd, index: events.append(f"select:{index}"),
    )

    result = commentary_module.main(
        [
            "--text",
            choice_text,
            "--press-enter",
            "--no-playback",
            "--no-obs-window",
            "--model",
            "same-model",
            "--commentary-model",
            "same-model",
            "--output",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert events == ["record", "plan", "speak", "select:2"]
    summary = json.loads(
        (tmp_path / "turn_001" / "result.json").read_text(encoding="utf-8")
    )
    assert summary["choice_plan"]["selected_label"] == "C"
    assert summary["selection_performed"] is True


@pytest.mark.parametrize(
    ("kind", "rows"),
    [
        ("triangle", commentary_module._TRIANGLE_MARKER_ROWS),
        ("book", commentary_module._BOOK_MARKER_ROWS),
    ],
)
def test_detect_advance_marker_from_template(
    kind: str,
    rows: tuple[str, ...],
) -> None:
    canvas = Image.new("L", (960, 540), 0)
    template = Image.fromarray(commentary_module._marker_template(rows))
    canvas.paste(template, (300, 200))
    marker = detect_advance_marker(canvas.convert("RGB"))
    assert marker.kind == kind
    assert marker.score == pytest.approx(1.0)


def test_detect_advance_marker_returns_none_for_blank_image() -> None:
    marker = detect_advance_marker(Image.new("RGB", (960, 540), "black"))
    assert marker.kind == "none"


def test_detect_triangle_marker_on_bright_background() -> None:
    canvas = Image.new("L", (960, 540), 180)
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(147, 213), (169, 228), (147, 245)], fill=0)
    draw.polygon([(150, 216), (165, 228), (150, 242)], fill=255)

    marker = detect_advance_marker(canvas.convert("RGB"))

    assert marker.kind == "triangle"
    assert marker.score == pytest.approx(0.90)
    assert marker.location is not None
    assert abs(marker.location[0] - 150) <= 1
    assert abs(marker.location[1] - 216) <= 1


def test_marker_wait_retries_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    blank = Image.new("RGB", (960, 540), "black")
    ready = Image.new("RGB", (960, 540), "white")
    results = iter(
        [
            (
                blank,
                commentary_module.AdvanceMarker(
                    "none",
                    0.60,
                    (100, 100),
                    waited_seconds=12.0,
                ),
            ),
            (
                ready,
                commentary_module.AdvanceMarker(
                    "triangle",
                    0.90,
                    (150, 216),
                    waited_seconds=0.2,
                ),
            ),
        ]
    )
    monkeypatch.setattr(
        commentary_module,
        "wait_for_advance_marker",
        lambda *args, **kwargs: next(results),
    )
    monkeypatch.setattr(commentary_module.time, "sleep", lambda seconds: None)
    timestamps = iter([10.0, 22.2])
    monkeypatch.setattr(
        commentary_module.time,
        "monotonic",
        lambda: next(timestamps),
    )
    timeout_capture = tmp_path / "marker_timeout_latest.png"

    image, marker = commentary_module.wait_for_advance_marker_with_retries(
        object(),
        activate=False,
        timeout=12.0,
        poll_interval=0.15,
        threshold=0.78,
        retries=0,
        retry_delay=1.0,
        retry_capture_path=timeout_capture,
    )

    assert image is ready
    assert marker.kind == "triangle"
    assert marker.retry_count == 1
    assert marker.waited_seconds == pytest.approx(12.2)
    assert timeout_capture.exists()


def test_extract_incremental_text_returns_only_appended_screen_text() -> None:
    previous = "ぼくは真理を見つめた。"
    current = "ぼくは真理を見つめた。\n彼女は注目の的だった。"
    assert extract_incremental_text(previous, current) == "彼女は注目の的だった。"


def test_extract_incremental_text_ignores_ocr_whitespace_changes() -> None:
    previous = (
        "真理とは、今年の四月に大学で知り合った。\n"
        "果敢かつ執ようなアタックで、何度かデートをする\n"
        "関係にまでこぎつけることができたのは、この秋の\n"
        "ことだ。"
    )
    current = (
        "真理とは、今年の四月に大学で知り合った。\n"
        "果敢かつ執ようなアタックで、 何度かデートをする\n"
        "関係にまでこぎつけることができたのは、この秋の\n"
        "ことだ。\n"
        "しかし、 押しても押しても手応えがなかった。"
    )
    assert extract_incremental_text(previous, current) == (
        "しかし、 押しても押しても手応えがなかった。"
    )


def test_extract_incremental_text_returns_empty_for_whitespace_only_change() -> None:
    assert extract_incremental_text(
        "アタックで、何度かデートをした。",
        "アタックで、　何度かデートをした。",
    ) == ""


def test_extract_incremental_text_keeps_replaced_screen_text() -> None:
    assert extract_incremental_text("前のページ。", "新しいページ。") == (
        "新しいページ。"
    )


def test_parse_commentary_plan() -> None:
    plan = parse_commentary_plan(
        '{"comment":"これは怪しい……","emotion":"tense",'
        '"intensity":0.75,"pace":"slow"}'
    )
    assert plan == CommentaryPlan(
        comment="これは怪しい……",
        mode="quick",
        emotion="tense",
        intensity=0.75,
        pace="slow",
    )


def test_parse_commentary_plan_handles_fence_and_invalid_values() -> None:
    plan = parse_commentary_plan(
        '```json\n{"comment":"気になるね","emotion":"unknown",'
        '"intensity":2,"pace":"medium"}\n```'
    )
    assert plan == CommentaryPlan(
        comment="気になるね",
        mode="quick",
        emotion="thoughtful",
        intensity=1.0,
        pace="normal",
    )


def test_parse_commentary_plan_normalizes_silent_delivery() -> None:
    plan = parse_commentary_plan(
        '{"mode":"silent","comment":"しゃべらない","emotion":"tense",'
        '"intensity":0.9,"pace":"fast"}'
    )
    assert plan == CommentaryPlan(
        comment="",
        mode="silent",
        emotion="calm",
        intensity=0.0,
        pace="normal",
    )
    assert commentary_plan_issue(plan) is None


def test_commentary_plan_issue_rejects_long_reaction() -> None:
    plan = CommentaryPlan(
        comment="えっ、ちょっと待って今のはいったい何なの!?",
        mode="reaction",
        emotion="surprised",
        intensity=0.8,
        pace="fast",
    )
    assert "上限12文字" in (commentary_plan_issue(plan) or "")


def test_commentary_policy_requires_speech_at_unspoken_page_end() -> None:
    silent = CommentaryPlan(
        comment="",
        mode="silent",
        emotion="calm",
        intensity=0.0,
        pace="normal",
    )
    assert "silentは禁止" in (
        commentary_plan_issue(
            silent,
            must_speak=True,
            advance_marker="book",
        )
        or ""
    )


def test_commentary_policy_disallows_quick_at_triangle_marker() -> None:
    quick = CommentaryPlan(
        comment="へぇ、そうなんだ。",
        mode="quick",
        emotion="calm",
        intensity=0.3,
        pace="normal",
    )
    assert "quickを使いません" in (
        commentary_plan_issue(
            quick,
            advance_marker="triangle",
        )
        or ""
    )


def test_commentary_policy_keeps_reaction_at_triangle_marker() -> None:
    reaction = CommentaryPlan(
        comment="うわっ！",
        mode="reaction",
        emotion="surprised",
        intensity=0.8,
        pace="fast",
    )
    assert commentary_plan_issue(
        reaction,
        advance_marker="triangle",
    ) is None


def test_commentary_policy_accepts_two_sentence_page_comment() -> None:
    page_comment = CommentaryPlan(
        comment=(
            "この子、落ち着いてて頼りになりそうだよね。"
            "眼鏡も似合ってるし、まとめ役っぽいかも。"
        ),
        mode="quick",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    assert commentary_plan_issue(
        page_comment,
        must_speak=True,
        advance_marker="book",
    ) is None


def test_commentary_policy_rejects_short_page_comment() -> None:
    short_comment = CommentaryPlan(
        comment="頼りになりそうだよね。",
        mode="quick",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    assert "短すぎます" in (
        commentary_plan_issue(
            short_comment,
            must_speak=True,
            advance_marker="book",
        )
        or ""
    )


def test_commentary_policy_rejects_reaction_only_at_page_end() -> None:
    reaction = CommentaryPlan(
        comment="うわっ！",
        mode="reaction",
        emotion="surprised",
        intensity=0.9,
        pace="fast",
    )
    assert "2～3文" in (
        commentary_plan_issue(
            reaction,
            must_speak=True,
            advance_marker="book",
        )
        or ""
    )


def test_commentary_policy_requires_soft_ending_at_page_end() -> None:
    flat = CommentaryPlan(
        comment=(
            "この子は落ち着いていて仕事もできそうに見える。"
            "みんなをまとめる役として期待できる。"
        ),
        mode="quick",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    assert "柔らかい語尾" in (
        commentary_plan_issue(
            flat,
            must_speak=True,
            advance_marker="book",
        )
        or ""
    )


def test_commentary_policy_rejects_blunt_sentence_ending() -> None:
    blunt = CommentaryPlan(
        comment="この子、落ち着いてて頼りになりそうだな。",
        mode="quick",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    assert "ぶっきらぼうな語尾" in (commentary_plan_issue(blunt) or "")


@pytest.mark.parametrize(
    ("mode", "input_intensity", "expected_intensity"),
    [
        ("silent", 0.0, 0.0),
        ("reaction", 0.3, 0.85),
        ("quick", 0.3, 0.55),
        ("extended", 0.3, 0.70),
        ("reaction", 0.95, 0.95),
    ],
)
def test_commentary_intensity_boost_uses_mode_floor(
    mode: str,
    input_intensity: float,
    expected_intensity: float,
) -> None:
    plan = CommentaryPlan(
        comment="" if mode == "silent" else "えっ、マジ!?",
        mode=mode,
        emotion="calm" if mode == "silent" else "surprised",
        intensity=input_intensity,
        pace="normal",
    )
    boosted, changed = apply_commentary_intensity_boost(plan)
    assert boosted.intensity == expected_intensity
    assert changed is (expected_intensity != input_intensity)


def test_commentary_plan_issue_rejects_long_quick_comment() -> None:
    plan = CommentaryPlan(
        comment="これは一息では言えないくらい長い通常場面の説明的な感想になっていて短く直す必要があります。",
        mode="quick",
        emotion="calm",
        intensity=0.3,
        pace="normal",
    )
    assert "上限35文字" in (commentary_plan_issue(plan) or "")


def test_commentary_plan_issue_accepts_natural_quick_comment() -> None:
    plan = CommentaryPlan(
        comment="いや、そういうものなの？",
        mode="quick",
        emotion="amused",
        intensity=0.3,
        pace="normal",
    )
    assert commentary_plan_issue(plan) is None


def test_commentary_plan_issue_rejects_nominal_slogan_ending() -> None:
    plan = CommentaryPlan(
        comment="ゴーグルの向こう、可愛い予感。",
        mode="quick",
        emotion="amused",
        intensity=0.3,
        pace="normal",
    )
    assert "体言止め" in (commentary_plan_issue(plan) or "")


def test_parse_commentary_plan_repairs_missing_outer_braces() -> None:
    plan = parse_commentary_plan(
        '"comment":"真理を信じる",\n'
        '"emotion":"thoughtful",\n'
        '"intensity":0.3,\n'
        '"pace":"normal"'
    )
    assert plan == CommentaryPlan(
        comment="真理を信じる",
        mode="quick",
        emotion="thoughtful",
        intensity=0.3,
        pace="normal",
    )


def test_parse_commentary_plan_recovers_fields_from_malformed_wrapper() -> None:
    plan = parse_commentary_plan(
        '["comment":"静かに忍び寄る違和感。",'
        '"emotion":"tense","intensity":0.65,"pace":"slow"}'
    )
    assert plan == CommentaryPlan(
        comment="静かに忍び寄る違和感。",
        mode="quick",
        emotion="tense",
        intensity=0.65,
        pace="slow",
    )


def test_commentary_speech_prompt_uses_selected_delivery() -> None:
    prompt = build_commentary_speech_prompt(
        CommentaryPlan(
            comment="えっ、今の何!?",
            mode="reaction",
            emotion="surprised",
            intensity=0.8,
            pace="fast",
        )
    )
    assert '"response_text": "えっ、今の何!?"' in prompt
    assert "思わず声が跳ねる大きな驚き" in prompt
    assert "20代くらいの女性" in prompt
    assert "相手へ話しかける柔らかい抑揚" in prompt
    assert "普段の会話よりリアクションを一段大きく" in prompt
    assert "少し速め" in prompt
    assert "0.80" in prompt


def test_narration_match_ignores_spaces_and_punctuation() -> None:
    assert narration_matches(
        "ぼくの名前は、透。\n東京の大学に通う学生だ。",
        "ぼくの名前は透 東京の大学に通う学生だ",
    )
    assert not narration_matches("透です。", "真理です。")


def test_narration_match_accepts_good_orthographic_variation() -> None:
    assert narration_matches(
        "このノリ、いい味出てる。",
        "このノリ、良い味出てる。",
    )


def test_clean_ocr_text_strips_page_cursor_attached_to_last_line(tmp_path) -> None:
    result = {
        "contents": [
            [
                {
                    "text": "真理もゴーグルをはずし、笑顔を見せた。山",
                    "confidence": 0.915,
                    "isTextline": "true",
                }
            ]
        ]
    }
    json_path = tmp_path / "capture.json"
    json_path.write_text(json.dumps(result), encoding="utf-8")
    assert (
        clean_ocr_text(json_path, min_confidence=0.5)
        == "真理もゴーグルをはずし、笑顔を見せた。"
    )


def test_clean_ocr_text_keeps_normal_trailing_kanji(tmp_path) -> None:
    result = {
        "contents": [
            [
                {
                    "text": "遠くに山",
                    "confidence": 0.9,
                    "isTextline": "true",
                }
            ]
        ]
    }
    json_path = tmp_path / "capture.json"
    json_path.write_text(json.dumps(result), encoding="utf-8")
    assert clean_ocr_text(json_path, min_confidence=0.5) == "遠くに山"
