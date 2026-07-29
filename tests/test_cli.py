import argparse
import json
import time
from threading import Event

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
    apply_ocr_replacements,
    build_choice_prompt,
    build_commentary_prompt,
    build_commentary_speech_prompt,
    build_narration_prompt,
    choice_plan_issue,
    choice_utterance,
    collapse_visual_line_breaks,
    commentary_plan_fallback,
    commentary_plan_issue,
    commentary_plan_style_notes,
    detect_advance_marker,
    extract_choice_options,
    extract_incremental_text,
    load_commentator_persona,
    load_ocr_replacements,
    narration_matches,
    parse_choice_plan,
    parse_commentary_plan,
    speak_choice_with_retries,
    verify_choice_speech,
)
from game_window_ocr.windows import normalize_title


def _subtitle_cue_lines(subtitles: str) -> list[list[str]]:
    return [
        block.splitlines()[2:]
        for block in subtitles.strip().split("\n\n")
        if block.strip()
    ]


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


def test_commentary_config_is_loaded_and_cli_takes_precedence(tmp_path) -> None:
    config_path = tmp_path / "commentary.toml"
    config_path.write_text(
        "ocr_interval = 0.75\n"
        "stable_ocr_samples = 4\n"
        "session_duration_minutes = 30.0\n"
        "ending_grace_minutes = 2.5\n"
        'summary_model = "gpt-5.6-terra"\n'
        'memory_dir = "memories"\n'
        'persona_file = "personas/curious-ai.md"\n'
        'initial_intro_file = "prompts/initial.txt"\n'
        'ocr_replacements_file = "config/replacements.txt"\n',
        encoding="utf-8",
    )

    configured = commentary_module._parse_args(
        ["--config", str(config_path)]
    )
    overridden = commentary_module._parse_args(
        [
            "--config",
            str(config_path),
            "--ocr-interval",
            "0.25",
            "--session-duration-minutes",
            "15",
        ]
    )

    assert configured.ocr_interval == pytest.approx(0.75)
    assert configured.stable_ocr_samples == 4
    assert configured.session_duration_minutes == pytest.approx(30.0)
    assert configured.ending_grace_minutes == pytest.approx(2.5)
    assert configured.summary_model == "gpt-5.6-terra"
    assert configured.memory_dir == (tmp_path / "memories").resolve()
    assert configured.persona_file == (
        tmp_path / "personas" / "curious-ai.md"
    ).resolve()
    assert configured.initial_intro_file == (
        tmp_path / "prompts" / "initial.txt"
    ).resolve()
    assert configured.ocr_replacements_file == (
        tmp_path / "config" / "replacements.txt"
    ).resolve()
    assert overridden.ocr_interval == pytest.approx(0.25)
    assert overridden.session_duration_minutes == pytest.approx(15.0)


def test_commentary_defaults_to_timed_unlimited_live_session() -> None:
    args = commentary_module._build_parser().parse_args([])

    assert args.max_turns == 0
    assert args.session_duration_minutes == pytest.approx(20.0)
    assert args.ending_grace_minutes == pytest.approx(5.0)
    assert args.unchanged_screen_retries == 3
    assert args.summary_model == "gpt-5.6-luna"
    assert args.initialize_memory is False


def test_memory_model_defaults_apart_from_summary_model() -> None:
    """記憶生成のモデルは締めとは別に指定できる"""
    defaults = commentary_module._build_parser().parse_args([])
    overridden = commentary_module._build_parser().parse_args(
        ["--memory-model", "gpt-5.6-terra"]
    )

    assert defaults.summary_model == "gpt-5.6-luna"
    assert defaults.memory_model == "gpt-5.6-sol"
    assert overridden.summary_model == "gpt-5.6-luna"
    assert overridden.memory_model == "gpt-5.6-terra"


def test_memory_model_is_loaded_from_config(tmp_path) -> None:
    """設定ファイルのmemory_modelを読み込む"""
    config_path = tmp_path / "commentary.toml"
    config_path.write_text(
        'summary_model = "gpt-5.6-luna"\nmemory_model = "gpt-5.6-sol"\n',
        encoding="utf-8",
    )

    configured = commentary_module._parse_args(["--config", str(config_path)])

    assert configured.summary_model == "gpt-5.6-luna"
    assert configured.memory_model == "gpt-5.6-sol"


def test_empty_memory_model_is_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--memory-model が空なら起動せずエラーにする"""
    result = commentary_module.main(["--memory-model", "  "])

    assert result == 2
    assert "--memory-model" in capsys.readouterr().err


def test_commentary_memory_can_be_initialized_from_cli() -> None:
    args = commentary_module._build_parser().parse_args(
        ["--initialize-memory"]
    )

    assert args.initialize_memory is True


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--max-turns", "-1"),
        ("--session-duration-minutes", "0"),
        ("--ending-grace-minutes", "-0.1"),
    ],
)
def test_invalid_session_ending_options_are_rejected(
    option: str,
    value: str,
) -> None:
    assert commentary_module.main([option, value]) == 2


def test_commentator_persona_is_loaded_as_utf8_markdown(tmp_path) -> None:
    persona_path = tmp_path / "persona.md"
    persona_path.write_text(
        "\ufeff# 人格\n\n人間を知りたい好奇心旺盛なAIです。\n",
        encoding="utf-8",
    )

    assert load_commentator_persona(persona_path) == (
        "# 人格\n\n人間を知りたい好奇心旺盛なAIです。"
    )


def test_ocr_replacements_are_loaded_and_applied_in_file_order(
    tmp_path,
) -> None:
    replacements_path = tmp_path / "replacements.txt"
    replacements_path.write_text(
        "\ufeff# 人名の読み\n\n真理 = マリ\n透=トオル\nマリ=まり",
        encoding="utf-8",
    )

    replacements = load_ocr_replacements(replacements_path)

    assert replacements == [
        ("真理", "マリ"),
        ("透", "トオル"),
        ("マリ", "まり"),
    ]
    assert apply_ocr_replacements(
        "真理は透に尋ねた。",
        replacements,
    ) == "まりはトオルに尋ねた。"


def test_luna_speech_plan_text_uses_the_same_replacements() -> None:
    replacements = [("真理", "マリ"), ("透", "トオル")]

    commentary_plan = commentary_module.apply_commentary_plan_replacements(
        CommentaryPlan(
            comment="真理と透が気になるね。",
            mode="quick",
            emotion="thoughtful",
            intensity=0.6,
            pace="normal",
        ),
        replacements,
    )
    choice_plan = commentary_module.apply_choice_plan_replacements(
        ChoicePlan(
            selected_label="B",
            opinion="透を信じてみたい。",
            emotion="thoughtful",
            intensity=0.6,
            pace="normal",
        ),
        replacements,
    )
    closing_plan = commentary_module.apply_closing_plan_replacements(
        commentary_module.ClosingPlan(
            ending_line="真理と透の話はここまで。",
            session_impression="透の行動が気になったね。",
            call_to_action="次回も真理を見守ってね。",
            emotion="thoughtful",
            intensity=0.6,
            pace="normal",
        ),
        replacements,
    )

    assert commentary_plan.comment == "マリとトオルが気になるね。"
    assert choice_plan.opinion == "トオルを信じてみたい。"
    assert closing_plan.message == (
        "マリとトオルの話はここまで。"
        "トオルの行動が気になったね。"
        "次回もマリを見守ってね。"
    )


def test_invalid_ocr_replacement_lines_do_not_block_valid_replacements(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    replacements_path = tmp_path / "replacements.txt"
    replacements_path.write_text(
        "区切りなし\n=空の置換前\n真理=マリ\n",
        encoding="utf-8",
    )

    replacements = load_ocr_replacements(replacements_path)

    assert replacements == [("真理", "マリ")]
    assert "不正な行を無視します" in capsys.readouterr().err


def test_missing_ocr_replacements_falls_back_to_original_text(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    replacements = load_ocr_replacements(tmp_path / "missing.txt")

    assert replacements == []
    assert apply_ocr_replacements("真理と透", replacements) == "真理と透"
    assert "置換せずに続行します" in capsys.readouterr().err


def test_recognize_capture_replaces_cleaned_ocr_and_keeps_original(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeOcrEngine:
        def recognize(self, *_args, **_kwargs) -> None:
            pass

    args = commentary_module._build_parser().parse_args([])
    args.scale = 1.0
    args.ocr_replacements = [("真理", "マリ"), ("透", "トオル")]
    monkeypatch.setattr(
        commentary_module,
        "clean_ocr_text",
        lambda *_args, **_kwargs: "真理は透に尋ねた。",
    )

    text = commentary_module._recognize_capture(
        Image.new("RGB", (16, 16)),
        tmp_path,
        args,
        FakeOcrEngine(),
    )

    assert text == "マリはトオルに尋ねた。"
    assert (tmp_path / "source.txt").read_text(
        encoding="utf-8"
    ) == "マリはトオルに尋ねた。\n"
    assert (tmp_path / "source_ocr_original.txt").read_text(
        encoding="utf-8"
    ) == "真理は透に尋ねた。\n"


def test_original_ocr_audit_save_failure_does_not_stop_replaced_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeOcrEngine:
        def recognize(self, *_args, **_kwargs) -> None:
            pass

    args = commentary_module._build_parser().parse_args([])
    args.scale = 1.0
    args.ocr_replacements = [("真理", "マリ")]
    monkeypatch.setattr(
        commentary_module,
        "clean_ocr_text",
        lambda *_args, **_kwargs: "真理が来た。",
    )
    original_write_text = commentary_module.Path.write_text

    def fail_only_audit_file(path, *write_args, **write_kwargs):
        if path.name == "source_ocr_original.txt":
            raise OSError("audit unavailable")
        return original_write_text(path, *write_args, **write_kwargs)

    monkeypatch.setattr(
        commentary_module.Path,
        "write_text",
        fail_only_audit_file,
    )

    text = commentary_module._recognize_capture(
        Image.new("RGB", (16, 16)),
        tmp_path,
        args,
        FakeOcrEngine(),
    )

    assert text == "マリが来た。"
    assert (tmp_path / "source.txt").read_text(
        encoding="utf-8"
    ) == "マリが来た。\n"
    assert "置換後の本文で続行します" in capsys.readouterr().err


def test_missing_commentator_persona_falls_back_without_error(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing.md"

    assert load_commentator_persona(missing_path) == ""
    assert "既存の基本口調で続行します" in capsys.readouterr().err


def test_invalid_utf8_commentator_persona_falls_back_without_error(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    persona_path = tmp_path / "broken.md"
    persona_path.write_bytes(b"\x81")

    assert load_commentator_persona(persona_path) == ""
    assert "既存の基本口調で続行します" in capsys.readouterr().err


def test_initial_intro_is_loaded_from_file_with_title_placeholder(
    tmp_path,
) -> None:
    intro_path = tmp_path / "intro.txt"
    intro_path.write_text(
        "ごきげんよう。\nスカイナです。\n今回は『{title}』です。",
        encoding="utf-8",
    )

    assert commentary_module.load_initial_intro(
        intro_path,
        title="かまいたちの夜",
    ) == "ごきげんよう。スカイナです。今回は『かまいたちの夜』です。"


def test_initial_intro_fallback_starts_with_fixed_greeting(tmp_path) -> None:
    errors: list[str] = []
    message = commentary_module.load_initial_intro(
        tmp_path / "missing.txt",
        title="かまいたちの夜",
        fallback_errors=errors,
    )

    assert message.startswith("ごきげんよう。")
    assert "スカイナ" in message
    assert "かまいたちの夜" in message
    assert errors


def test_default_ocr_interval_is_half_a_second() -> None:
    args = commentary_module._build_parser().parse_args([])
    assert args.ocr_interval == pytest.approx(0.5)


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
    assert (
        tmp_path / "subtitles" / "commentary.srt"
    ).read_bytes() == b"\xef\xbb\xbf"


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


@pytest.mark.parametrize("noise", ["4A", "ｱｲ"])
def test_clean_ocr_text_strips_two_trailing_halfwidth_characters(
    tmp_path,
    noise: str,
) -> None:
    result = {
        "contents": [
            [
                {
                    "text": f"雪山へ向かったのだ{noise}",
                    "confidence": 0.9,
                    "isTextline": "true",
                }
            ]
        ]
    }
    json_path = tmp_path / "capture.json"
    json_path.write_text(json.dumps(result), encoding="utf-8")

    assert clean_ocr_text(json_path, min_confidence=0.5) == "雪山へ向かったのだ"


def test_clean_ocr_text_keeps_one_trailing_halfwidth_character(tmp_path) -> None:
    result = {
        "contents": [
            [
                {
                    "text": "合言葉はX",
                    "confidence": 0.9,
                    "isTextline": "true",
                }
            ]
        ]
    }
    json_path = tmp_path / "capture.json"
    json_path.write_text(json.dumps(result), encoding="utf-8")

    assert clean_ocr_text(json_path, min_confidence=0.5) == "合言葉はX"


def test_narration_prompt_wraps_source_as_verbatim_json() -> None:
    prompt = build_narration_prompt("ぼくの名前は、透。")
    assert '"response_text": "ぼくの名前は、透。"' in prompt
    assert '"require_repeat_verbatim": true' in prompt
    assert "先頭文字から直ちに話し始め" in prompt
    assert "そのまま読み上げますね" in prompt
    assert "前置きや終了報告を絶対に付けない" in prompt


def test_narration_prompt_collapses_visual_line_wraps() -> None:
    assert collapse_visual_line_breaks("そういう意味で\nも、ま、太った") == (
        "そういう意味でも、ま、太った"
    )
    prompt = build_narration_prompt("そういう意味で\nも、ま、太った")
    assert '"response_text": "そういう意味でも、ま、太った"' in prompt


def test_commentary_prompt_limits_length_and_repetition() -> None:
    prompt = build_commentary_prompt(
        "雪が降っていた。",
        persona="人間を知りたいAI。20代くらいの女性の声で話す。",
        page_text="山道を歩いていた。雪が降っていた。",
        advance_marker="book",
        page_has_spoken=False,
        must_speak=True,
    )
    assert '"new_game_text": "雪が降っていた。"' in prompt
    assert "silent" in prompt
    assert "reaction" in prompt
    assert "1～20文字" in prompt
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
    assert "今回は開始挨拶ではない" in prompt
    assert "「ごきげんよう」" in prompt
    assert "choice_selection_performed" in prompt
    assert "初めて知ったように驚かない" in prompt
    assert "選択した台詞の表示だけ" in prompt
    assert "予想外の結果、新事実、他の人物の反応" in prompt


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
    attempts: list[SpeechResult] = []
    utterance = choice_utterance(plan)
    subtitle_writer = commentary_module.SrtSubtitleWriter(
        tmp_path / "subtitles" / "commentary.srt"
    )

    def record_attempt(speech: SpeechResult) -> None:
        attempts.append(speech)
        commentary_module._append_speech_subtitle(
            subtitle_writer,
            utterance,
            speech,
        )

    class FakeRealtime:
        def speak(self, **kwargs) -> SpeechResult:
            attempt_number = len(attempts)
            return SpeechResult(
                phase=str(kwargs["phase"]),
                transcript=next(transcripts),
                audio_bytes=10,
                response_id=None,
                started_at_seconds=float(attempt_number + 1),
                ended_at_seconds=float(attempt_number + 2),
            )

    monkeypatch.setattr(commentary_module.time, "sleep", lambda seconds: None)
    speech, verification, retries = speak_choice_with_retries(
        FakeRealtime(),
        plan,
        turn_dir=tmp_path,
        playback=False,
        retries=2,
        retry_delay=0.5,
        on_speech=record_attempt,
    )
    assert speech.transcript.endswith("Bを選ぶね。")
    assert verification.matches is True
    assert retries == 1
    assert [attempt.phase for attempt in attempts] == [
        "choice",
        "choice_retry",
    ]
    assert (tmp_path / "choice_transcript.txt").exists()
    assert (tmp_path / "choice_retry_001_transcript.txt").exists()
    subtitles = (
        tmp_path / "subtitles" / "commentary.srt"
    ).read_text(encoding="utf-8-sig")
    subtitle_utterance = utterance.replace("。 ", "。\n")
    assert subtitles.count(subtitle_utterance) == 2
    assert subtitles.count(" --> ") == 2


def test_regular_speech_subtitle_is_not_split_into_multiple_cues(
    tmp_path,
) -> None:
    subtitle_path = tmp_path / "subtitles" / "commentary.srt"
    writer = commentary_module.SrtSubtitleWriter(subtitle_path)
    speech = SpeechResult(
        phase="commentary",
        transcript="一文目。二文目。三文目。",
        audio_bytes=100,
        response_id=None,
        started_at_seconds=1.0,
        ended_at_seconds=4.0,
    )

    commentary_module._append_speech_subtitle(
        writer,
        "一文目。二文目。三文目。",
        speech,
    )

    subtitles = subtitle_path.read_text(encoding="utf-8-sig")
    assert subtitles.count(" --> ") == 1
    assert _subtitle_cue_lines(subtitles) == [
        ["一文目。", "二文目。", "三文目。"]
    ]


def test_speech_subtitle_with_missing_timing_is_skipped(tmp_path) -> None:
    subtitle_path = tmp_path / "subtitles" / "commentary.srt"
    writer = commentary_module.SrtSubtitleWriter(subtitle_path)
    speech = SpeechResult(
        phase="startup",
        transcript="開始します。",
        audio_bytes=100,
        response_id=None,
    )

    commentary_module._append_speech_subtitle(
        writer,
        "開始します。",
        speech,
        max_lines_per_cue=2,
    )

    assert subtitle_path.read_bytes() == b"\xef\xbb\xbf"


def test_choice_prompt_contains_only_available_labels() -> None:
    prompt = build_choice_prompt(
        (
            ChoiceOption("A", "一つ目"),
            ChoiceOption("B", "二つ目"),
        ),
        persona="人間の選択に興味津々なAI。",
    )
    assert "A, B" in prompt
    assert '"label": "A"' in prompt
    assert '"label": "B"' in prompt
    assert "選択宣言はプログラムが" in prompt
    assert "人間の選択に興味津々なAI" in prompt


def test_main_narrates_choice_text_then_speaks_before_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
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

        def speak(self, **kwargs) -> SpeechResult:
            phase = kwargs["phase"]
            events.append(phase)
            if phase == "narration":
                return SpeechResult(
                    phase=phase,
                    transcript=collapse_visual_line_breaks(choice_text),
                    audio_bytes=200,
                    response_id="choice-narration",
                    started_at_seconds=0.25,
                    ended_at_seconds=1.0,
                )
            return SpeechResult(
                phase=phase,
                transcript=expected_utterance,
                audio_bytes=100,
                response_id="choice-speech",
                started_at_seconds=1.25,
                ended_at_seconds=2.5,
            )

    class FakePlanner:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

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

        def record_confirmed_choice(
            self,
            *,
            plan: ChoicePlan,
            selected_option: ChoiceOption,
        ) -> None:
            assert plan.selected_label == "C"
            assert selected_option == ChoiceOption("C", "じっくり考える")
            events.append("confirmed:C")
            raise RuntimeError("履歴記録失敗")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(commentary_module, "ObsCaptureWindow", FakeObsWindow)
    monkeypatch.setattr(
        commentary_module,
        "RealtimeSpeechClient",
        FakeRealtimeClient,
    )
    monkeypatch.setattr(
        commentary_module,
        "ResponsesCommentaryPlanner",
        FakePlanner,
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
    assert events == [
        "narration",
        "plan",
        "choice",
        "select:2",
        "confirmed:C",
    ]
    summary = json.loads(
        (tmp_path / "turn_001" / "result.json").read_text(encoding="utf-8")
    )
    assert summary["narration_matches"] is True
    assert summary["narration"]["response_id"] == "choice-narration"
    assert summary["choice_plan"]["selected_label"] == "C"
    assert summary["selection_performed"] is True
    assert summary["choice_speech"]["started_at_seconds"] == 1.25
    assert summary["choice_speech"]["ended_at_seconds"] == 2.5
    assert "ゲーム進行を続けます" in capsys.readouterr().err
    subtitles = (
        tmp_path / "subtitles" / "commentary.srt"
    ).read_text(encoding="utf-8-sig")
    assert "00:00:01,250 --> 00:00:02,500" in subtitles
    assert expected_utterance.replace("。 ", "。\n") in subtitles
    assert choice_text not in subtitles


def test_main_plans_commentary_while_narration_is_playing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []
    plan_started = Event()
    narration_started = Event()

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

        def speak(self, **kwargs) -> SpeechResult:
            phase = str(kwargs["phase"])
            if phase == "narration":
                assert plan_started.wait(timeout=1)
                events.append("narration")
                narration_started.set()
                transcript = "以前交わした約束を、ついに果たした。"
                started_at_seconds = 0.5
                ended_at_seconds = 1.5
            else:
                events.append("commentary")
                transcript = "前の約束が、ここにつながったんだね。"
                started_at_seconds = 2.0
                ended_at_seconds = 3.25
            return SpeechResult(
                phase=phase,
                transcript=transcript,
                audio_bytes=100,
                response_id=f"{phase}-response",
                started_at_seconds=started_at_seconds,
                ended_at_seconds=ended_at_seconds,
            )

    class FakePlanner:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def generate_text(self, **kwargs) -> TextResult:
            events.append("plan_start")
            plan_started.set()
            assert narration_started.wait(timeout=1)
            return TextResult(
                text=(
                    '{"mode":"quick",'
                    '"comment":"前の約束がここにつながったんだね。",'
                    '"emotion":"thoughtful","intensity":0.6,"pace":"normal"}'
                ),
                response_id="plan-response",
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
        "ResponsesCommentaryPlanner",
        FakePlanner,
    )

    result = commentary_module.main(
        [
            "--text",
            "以前交わした約束を、ついに果たした。",
            "--no-playback",
            "--no-obs-window",
            "--output",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert events == ["plan_start", "narration", "commentary"]
    summary = json.loads(
        (tmp_path / "turn_001" / "result.json").read_text(encoding="utf-8")
    )
    assert summary["model"] == "gpt-realtime-2.1-mini"
    assert summary["commentary_model"] == "gpt-5.6-luna"
    assert summary["commentary_api"] == "responses"
    assert (
        summary["commentary_plan"]["wait_after_narration_seconds"]
        is not None
    )
    assert summary["commentary"]["started_at_seconds"] == 2.0
    assert summary["commentary"]["ended_at_seconds"] == 3.25
    subtitles = (
        tmp_path / "subtitles" / "commentary.srt"
    ).read_text(encoding="utf-8-sig")
    assert "00:00:02,000 --> 00:00:03,250" in subtitles
    assert "前の約束がここにつながったんだね。" in subtitles
    assert "以前交わした約束を、ついに果たした。" not in subtitles
    assert "前の約束が、ここにつながったんだね。" not in subtitles


def test_main_excludes_narration_and_silent_turn_from_subtitles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source_text = "静かな廊下を歩いた。"

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

        def speak(self, **kwargs) -> SpeechResult:
            assert kwargs["phase"] == "narration"
            return SpeechResult(
                phase="narration",
                transcript=source_text,
                audio_bytes=100,
                response_id="narration-response",
                started_at_seconds=1.0,
                ended_at_seconds=2.0,
            )

    class FakePlanner:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def generate_text(self, **kwargs) -> TextResult:
            return TextResult(
                text=(
                    '{"mode":"silent","comment":"","emotion":"calm",'
                    '"intensity":0.0,"pace":"normal"}'
                ),
                response_id="plan-response",
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
        "ResponsesCommentaryPlanner",
        FakePlanner,
    )

    result = commentary_module.main(
        [
            "--text",
            source_text,
            "--no-playback",
            "--no-obs-window",
            "--output",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert (
        tmp_path / "subtitles" / "commentary.srt"
    ).read_bytes() == b"\xef\xbb\xbf"


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


def test_detect_advance_marker_keeps_candidate_below_threshold() -> None:
    canvas = Image.new("L", (960, 540), 0)
    template = Image.fromarray(
        commentary_module._marker_template(
            commentary_module._BOOK_MARKER_ROWS
        )
    )
    canvas.paste(template, (300, 200))

    marker = detect_advance_marker(
        canvas.convert("RGB"),
        threshold=1.01,
    )

    assert marker.kind == "none"
    assert marker.candidate_kind == "book"
    assert marker.score == pytest.approx(1.0)


def test_capture_and_ocr_uses_stable_text_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image = Image.new("RGB", (960, 540), "black")
    waits = iter(
        [
            (
                image,
                commentary_module.AdvanceMarker(
                    "none",
                    score,
                    (645, 213),
                    candidate_kind="book",
                ),
            )
            for score in (0.68, 0.70, 0.712)
        ]
    )
    recognized: list[str] = []
    current_text = "前の本文。\n新しく表示された本文。"

    monkeypatch.setattr(
        commentary_module,
        "wait_for_advance_marker",
        lambda *args, **kwargs: next(waits),
    )

    def fake_recognize(*args, **kwargs) -> str:
        recognized.append(current_text)
        return current_text

    monkeypatch.setattr(
        commentary_module,
        "_recognize_capture",
        fake_recognize,
    )
    args = commentary_module._build_parser().parse_args([])

    text, marker = commentary_module._capture_and_ocr(
        object(),
        tmp_path,
        args,
        object(),
        previous_text="前の本文。",
    )

    assert text == current_text
    assert len(recognized) == commentary_module.STABLE_OCR_REQUIRED_SAMPLES
    assert marker.kind == "book"
    assert marker.score == pytest.approx(0.712)
    assert marker.candidate_kind == "book"
    assert marker.fallback_reason == "stable_ocr"


def test_capture_and_ocr_requires_consecutive_stable_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image = Image.new("RGB", (960, 540), "black")
    wait_calls = 0
    texts = iter(
        [
            "前の本文。\n表示途中",
            "前の本文。\n表示が完了した。",
            "前の本文。\n表示が完了した。",
            "前の本文。\n表示が完了した。",
        ]
    )

    def fake_wait(*args, **kwargs):
        nonlocal wait_calls
        wait_calls += 1
        return (
            image,
            commentary_module.AdvanceMarker(
                "none",
                0.40,
                (100, 100),
                candidate_kind="triangle",
            ),
        )

    monkeypatch.setattr(
        commentary_module,
        "wait_for_advance_marker",
        fake_wait,
    )
    monkeypatch.setattr(
        commentary_module,
        "_recognize_capture",
        lambda *args, **kwargs: next(texts),
    )
    args = commentary_module._build_parser().parse_args([])

    text, marker = commentary_module._capture_and_ocr(
        object(),
        tmp_path,
        args,
        object(),
        previous_text="前の本文。",
    )

    assert wait_calls == 4
    assert text == "前の本文。\n表示が完了した。"
    assert marker.kind == "stable_text"
    assert marker.fallback_reason == "stable_ocr"


def test_capture_and_ocr_returns_unchanged_text_for_enter_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image = Image.new("RGB", (960, 540), "black")
    waits = iter(
        [
            (
                image,
                commentary_module.AdvanceMarker(
                    "none",
                    0.70,
                    (645, 213),
                    candidate_kind="book",
                ),
            ),
            (
                image,
                commentary_module.AdvanceMarker(
                    "none",
                    0.70,
                    (645, 213),
                    candidate_kind="book",
                ),
            ),
            (
                image,
                commentary_module.AdvanceMarker(
                    "none",
                    0.70,
                    (645, 213),
                    candidate_kind="book",
                ),
            ),
        ]
    )
    recognized: list[str] = []
    previous_text = "前の画面と同じ本文。"

    monkeypatch.setattr(
        commentary_module,
        "wait_for_advance_marker",
        lambda *args, **kwargs: next(waits),
    )

    def fake_recognize(*args, **kwargs) -> str:
        recognized.append(previous_text)
        return previous_text

    monkeypatch.setattr(
        commentary_module,
        "_recognize_capture",
        fake_recognize,
    )
    args = commentary_module._build_parser().parse_args([])

    text, marker = commentary_module._capture_and_ocr(
        object(),
        tmp_path,
        args,
        object(),
        previous_text=previous_text,
    )

    assert text == previous_text
    assert len(recognized) == commentary_module.STABLE_OCR_REQUIRED_SAMPLES
    assert marker.kind == "book"
    assert marker.fallback_reason == "unchanged_ocr"


def test_capture_and_ocr_does_not_fallback_on_partial_choice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image = Image.new("RGB", (960, 540), "black")
    waits = iter(
        [
            (
                image,
                commentary_module.AdvanceMarker(
                    "none",
                    0.70,
                    (645, 213),
                    candidate_kind="book",
                ),
            ),
            (
                image,
                commentary_module.AdvanceMarker(
                    "none",
                    0.70,
                    (645, 213),
                    candidate_kind="book",
                ),
            ),
            (
                image,
                commentary_module.AdvanceMarker(
                    "none",
                    0.70,
                    (645, 213),
                    candidate_kind="book",
                ),
            ),
            (
                image,
                commentary_module.AdvanceMarker(
                    "triangle",
                    0.90,
                    (150, 216),
                    candidate_kind="triangle",
                ),
            ),
        ]
    )
    recognized: list[str] = []
    partial_choice = "A: 左へ進む"

    monkeypatch.setattr(
        commentary_module,
        "wait_for_advance_marker",
        lambda *args, **kwargs: next(waits),
    )

    def fake_recognize(*args, **kwargs) -> str:
        recognized.append(partial_choice)
        return partial_choice

    monkeypatch.setattr(
        commentary_module,
        "_recognize_capture",
        fake_recognize,
    )
    args = commentary_module._build_parser().parse_args([])

    text, marker = commentary_module._capture_and_ocr(
        object(),
        tmp_path,
        args,
        object(),
    )

    assert text == partial_choice
    assert len(recognized) == 4
    assert marker.kind == "triangle"
    assert marker.fallback_reason is None


def test_main_stable_text_fallback_sends_enter_once_after_narration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []
    source_text = "文字表示が完了した。"
    image = Image.new("RGB", (960, 540), "black")
    wait_calls = 0
    wait_timeouts: list[float] = []

    class FakeObsWindow:
        error = None
        is_open = False

        def __init__(self, *, enabled: bool) -> None:
            assert enabled is False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

    class FakeOcr:
        initialization_seconds = 0.0

    class FakeRealtimeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def speak(self, **kwargs) -> SpeechResult:
            phase = kwargs["phase"]
            events.append(phase)
            return SpeechResult(
                phase=phase,
                transcript="本文とは異なる転写。",
                audio_bytes=100,
                response_id=f"{phase}-response",
            )

    def fake_wait(*args, **kwargs):
        nonlocal wait_calls
        wait_calls += 1
        wait_timeouts.append(kwargs["timeout"])
        return (
            image,
            commentary_module.AdvanceMarker(
                "none",
                0.712,
                (645, 213),
                candidate_kind="book",
            ),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(commentary_module, "ObsCaptureWindow", FakeObsWindow)
    monkeypatch.setattr(commentary_module, "PersistentNdlOcr", FakeOcr)
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
        "wait_for_advance_marker",
        fake_wait,
    )
    monkeypatch.setattr(
        commentary_module,
        "_recognize_capture",
        lambda *args, **kwargs: source_text,
    )
    monkeypatch.setattr(
        commentary_module,
        "press_enter",
        lambda hwnd: events.append("enter"),
    )

    result = commentary_module.main(
        [
            "--press-enter",
            "--max-turns",
            "1",
            "--narration-only",
            "--no-playback",
            "--no-obs-window",
            "--output",
            str(tmp_path),
            "--memory-dir",
            str(tmp_path / "memory"),
        ]
    )

    assert result == 0
    assert wait_calls == commentary_module.STABLE_OCR_REQUIRED_SAMPLES
    assert wait_timeouts == [0.5, 0.5, 0.5]
    assert events == ["startup", "narration", "enter"]
    summary = json.loads(
        (tmp_path / "turn_001" / "result.json").read_text(encoding="utf-8")
    )
    assert summary["advance_marker"]["kind"] == "book"
    assert summary["advance_marker"]["fallback_reason"] == "stable_ocr"
    assert summary["narration_matches"] is False
    assert summary["enter_pressed"] is True
    assert summary["enter_blocked_reason"] is None


def test_main_resends_enter_without_repeating_speech_for_unchanged_screen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []
    captured_texts = iter(
        [
            "最初の本文。",
            "最初の本文。",
            "最初の本文。",
            "最初の本文。",
            "次の本文。",
        ]
    )
    narration_transcripts = iter(["最初の本文。", "次の本文。"])
    image = Image.new("RGB", (960, 540), "black")
    marker_results = iter(
        [
            commentary_module.AdvanceMarker(
                "book",
                0.90,
                (377, 258),
                candidate_kind="book",
            ),
            *[
                commentary_module.AdvanceMarker(
                    "none",
                    0.712,
                    (645, 213),
                    candidate_kind="book",
                )
                for _ in range(commentary_module.STABLE_OCR_REQUIRED_SAMPLES)
            ],
            commentary_module.AdvanceMarker(
                "book",
                0.90,
                (377, 258),
                candidate_kind="book",
            ),
        ]
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

    class FakeOcr:
        initialization_seconds = 0.0

    class FakeRealtimeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def speak(self, **kwargs) -> SpeechResult:
            phase = kwargs["phase"]
            events.append(phase)
            transcript = (
                next(narration_transcripts)
                if phase == "narration"
                else "開始します。"
            )
            return SpeechResult(
                phase=phase,
                transcript=transcript,
                audio_bytes=100,
                response_id=f"{phase}-response",
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(commentary_module, "ObsCaptureWindow", FakeObsWindow)
    monkeypatch.setattr(commentary_module, "PersistentNdlOcr", FakeOcr)
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
        "wait_for_advance_marker",
        lambda *args, **kwargs: (
            image,
            next(marker_results),
        ),
    )
    monkeypatch.setattr(
        commentary_module,
        "_recognize_capture",
        lambda *args, **kwargs: next(captured_texts),
    )
    monkeypatch.setattr(
        commentary_module,
        "press_enter",
        lambda hwnd: events.append("enter"),
    )

    result = commentary_module.main(
        [
            "--press-enter",
            "--max-turns",
            "2",
            "--narration-only",
            "--no-playback",
            "--no-obs-window",
            "--after-enter-delay",
            "0",
            "--output",
            str(tmp_path),
            "--memory-dir",
            str(tmp_path / "memory"),
        ]
    )

    assert result == 0
    assert events == [
        "startup",
        "narration",
        "enter",
        "enter",
        "narration",
        "enter",
    ]
    recovery = json.loads(
        (
            tmp_path
            / "turn_002"
            / "unchanged_screen_recovery_001.json"
        ).read_text(encoding="utf-8")
    )
    assert recovery["enter_pressed"] is True
    assert recovery["advance_marker"]["fallback_reason"] == "unchanged_ocr"
    second_summary = json.loads(
        (tmp_path / "turn_002" / "result.json").read_text(encoding="utf-8")
    )
    assert second_summary["source_text"] == "次の本文。"
    assert second_summary["unchanged_screen_retries"] == 1
    assert second_summary["enter_pressed"] is True


def test_main_stops_automatic_input_after_unchanged_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []
    captured_texts = iter(["最初の本文。", "最初の本文。"])
    image = Image.new("RGB", (960, 540), "black")

    class FakeObsWindow:
        error = None
        is_open = False

        def __init__(self, *, enabled: bool) -> None:
            assert enabled is False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

    class FakeOcr:
        initialization_seconds = 0.0

    class FakeRealtimeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def speak(self, **kwargs) -> SpeechResult:
            phase = kwargs["phase"]
            events.append(phase)
            return SpeechResult(
                phase=phase,
                transcript=(
                    "最初の本文。"
                    if phase == "narration"
                    else "開始します。"
                ),
                audio_bytes=100,
                response_id=f"{phase}-response",
            )

    planner_models: list[str] = []

    class FakeSummaryPlanner:
        def __init__(self, **kwargs) -> None:
            planner_models.append(kwargs["model"])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(commentary_module, "ObsCaptureWindow", FakeObsWindow)
    monkeypatch.setattr(commentary_module, "PersistentNdlOcr", FakeOcr)
    monkeypatch.setattr(
        commentary_module,
        "RealtimeSpeechClient",
        FakeRealtimeClient,
    )
    monkeypatch.setattr(
        commentary_module,
        "ResponsesCommentaryPlanner",
        FakeSummaryPlanner,
    )
    monkeypatch.setattr(
        commentary_module,
        "_finalize_timed_session",
        lambda **kwargs: events.append("finalize"),
    )
    monkeypatch.setattr(
        commentary_module,
        "find_window",
        lambda title: commentary_module.WindowInfo(123, title, 960, 540),
    )
    monkeypatch.setattr(
        commentary_module,
        "wait_for_advance_marker",
        lambda *args, **kwargs: (
            image,
            commentary_module.AdvanceMarker(
                "book",
                0.90,
                (377, 258),
                candidate_kind="book",
            ),
        ),
    )
    monkeypatch.setattr(
        commentary_module,
        "_recognize_capture",
        lambda *args, **kwargs: next(captured_texts),
    )
    monkeypatch.setattr(
        commentary_module,
        "press_enter",
        lambda hwnd: events.append("enter"),
    )

    result = commentary_module.main(
        [
            "--press-enter",
            "--max-turns",
            "2",
            "--narration-only",
            "--no-playback",
            "--no-obs-window",
            "--after-enter-delay",
            "0",
            "--unchanged-screen-retries",
            "0",
            "--output",
            str(tmp_path),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--summary-model",
            "gpt-5.6-luna",
            "--memory-model",
            "gpt-5.6-sol",
        ]
    )

    assert result == 0
    assert events == ["startup", "narration", "enter", "finalize"]
    assert planner_models == ["gpt-5.6-luna", "gpt-5.6-sol"]
    stopped_summary = json.loads(
        (tmp_path / "turn_002" / "result.json").read_text(encoding="utf-8")
    )
    assert stopped_summary["enter_pressed"] is False
    assert (
        stopped_summary["enter_blocked_reason"]
        == "unchanged_screen_recovery_exhausted"
    )
    assert len(commentary_module._collect_session_records(tmp_path)) == 1


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


def test_extract_incremental_text_tolerates_dropped_ellipsis_in_old_text() -> None:
    previous = (
        "「すいません。一緒に写ってもらえませんか?」"
        "三人組の一人、メガネをかけた女の子がデジタル"
        "カメラを差し出しながら、ぼく達に話しかけて来た。"
        "……来たな。"
    )
    current = (
        "「すいません。一緒に写ってもらえませんか?」"
        "三人組の一人、メガネをかけた女の子がデジタル"
        "カメラを差し出しながら、ぼく達に話しかけて来た。"
        "来たな。"
        "これだから、男前はこまるよ……。"
    )

    assert extract_incremental_text(previous, current) == (
        "これだから、男前はこまるよ……。"
    )


def test_extract_incremental_text_tolerates_inserted_ocr_noise() -> None:
    previous = (
        "「ああ、別に構いませんよ。さて、どんなボーズで…"
        "ところが、メガネの子は慌ててぼくを押しとどめる。"
    )
    current = (
        "「ああ、別に構いませんよ。さて、どんなボーズで…"
        "000J"
        "ところが、メガネの子は慌ててぼくを押しとどめる。"
        "「あ、マネージャーの方はけっこうですから」"
    )

    assert extract_incremental_text(previous, current) == (
        "「あ、マネージャーの方はけっこうですから」"
    )


def test_extract_incremental_text_tolerates_changed_ocr_noise() -> None:
    previous = (
        "「ああ、別に構いませんよ。さて、どんなボーズで…"
        "000J"
        "ところが、メガネの子は慌ててぼくを押しとどめる。"
        "「あ、マネージャーの方はけっこうですから」"
    )
    current = (
        "「ああ、別に構いませんよ。さて、どんなボーズで…"
        "COOJ"
        "ところが、メガネの子は慌ててぼくを押しとどめる。"
        "「あ、マネージャーの方はけっこうですから」"
        "……マネー……ジャー?"
    )

    assert extract_incremental_text(previous, current) == (
        "……マネー……ジャー?"
    )


def test_extract_incremental_text_returns_empty_for_ocr_only_change() -> None:
    assert extract_incremental_text(
        "長めの文章の途中に画像由来の000Jという誤認識が入った。",
        "長めの文章の途中に画像由来のCOOJという誤認識が入った。",
    ) == ""


def test_extract_incremental_text_keeps_replaced_screen_text() -> None:
    assert extract_incremental_text("前のページ。", "新しいページ。") == (
        "新しいページ。"
    )


def test_extract_incremental_text_tolerates_reordered_ocr_lines() -> None:
    """OCRが行順を入れ替えても、追加された行だけを返す"""
    previous = (
        "突然、小林さんががばっと立ち上がった。\n"
        "「どうかしましたか?」\n"
        "ぼくはたずねた。\n"
        "外を調べて来る」\n"
        "「外を?どうしてですか?」"
    )
    current = (
        "突然、小林さんががばっと立ち上がった。\n"
        "「どうかしましたか?」\n"
        "外を調べて来る」\n"
        "「外を?どうしてですか?」\n"
        "「何か、痕跡があるかもしれないじゃないか。\n"
        "ぼくはたずねた。"
    )

    assert extract_incremental_text(previous, current) == (
        "「何か、痕跡があるかもしれないじゃないか。"
    )


def test_extract_incremental_text_tolerates_reordered_and_rewritten_lines() -> None:
    """行順の入れ替えとOCR揺れが同時に起きても、追加された行だけを返す"""
    previous = (
        "突然、小林さんががばっと立ち上がった。\n"
        "「どうかしましたか?」\n"
        "外を調べて来る」\n"
        "「外を?どうしてですか?」\n"
        "「何か、痕跡があるかもしれないじゃないか。\n"
        "ぼくはたずねた。"
    )
    current = (
        "突然、 小林さんががばっと立ち上がった。\n"
        "「どうかしましたか?」\n"
        "ぼくはたずねた。\n"
        "外を調べて来る」\n"
        "「外を?どうしてですか?」\n"
        "「…… 痕跡があるかもしれないじゃないか。\n"
        "足跡とかタイヤの跡とか」"
    )

    assert extract_incremental_text(previous, current) == (
        "足跡とかタイヤの跡とか」"
    )


def test_extract_incremental_text_keeps_replaced_page_with_reordered_lines() -> None:
    """行が総入れ替えになったページ切り替えでは、従来どおり全文を返す"""
    previous = (
        "真理とは、今年の四月に大学で知り合った。\n"
        "果敢かつ執ようなアタックを繰り返してきた。"
    )
    current = (
        "季節はいつのまにか冬になっていた。\n"
        "山荘には誰も近づかなくなっていた。"
    )

    assert extract_incremental_text(previous, current) == (
        "季節はいつのまにか冬になっていた。山荘には誰も近づかなくなっていた。"
    )


def test_extract_incremental_text_drops_lines_already_spoken_on_page() -> None:
    """差分が特定できない画面でも、このページで朗読済みの行は読み直さない"""
    previous = "まったく無関係な直前の画面テキスト。"
    current = (
        "突然、小林さんががばっと立ち上がった。\n"
        "「どうかしましたか?」\n"
        "「外を調べて来る」"
    )
    spoken = "突然、小林さんががばっと立ち上がった。「どうかしましたか?」"

    assert extract_incremental_text(previous, current, spoken_text=spoken) == (
        "「外を調べて来る」"
    )


def test_extract_incremental_text_keeps_full_text_when_all_lines_spoken() -> None:
    """朗読済み行を除くと何も残らない場合は、全文を返して進行を止めない"""
    previous = "まったく無関係な直前の画面テキスト。"
    current = "突然、小林さんががばっと立ち上がった。\n「どうかしましたか?」"
    spoken = "突然、小林さんががばっと立ち上がった。「どうかしましたか?」"

    assert extract_incremental_text(previous, current, spoken_text=spoken) == (
        "突然、小林さんががばっと立ち上がった。「どうかしましたか?」"
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


def test_parse_commentary_plan_removes_startup_only_greeting() -> None:
    plan = parse_commentary_plan(
        '{"mode":"quick","comment":"ごきげんようこれは怪しいね。",'
        '"emotion":"tense","intensity":0.75,"pace":"slow"}'
    )
    assert plan.comment == "これは怪しいね。"


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


def test_commentary_plan_issue_enforces_reaction_length_boundary() -> None:
    allowed_plan = CommentaryPlan(
        comment="えっ、ちょっと待って今のはいったい何なの",
        mode="reaction",
        emotion="surprised",
        intensity=0.8,
        pace="fast",
    )
    long_plan = CommentaryPlan(
        comment="えっ、ちょっと待って今のはいったい何なの!?",
        mode="reaction",
        emotion="surprised",
        intensity=0.8,
        pace="fast",
    )
    assert len(allowed_plan.comment) == 20
    assert commentary_plan_issue(allowed_plan) is None
    assert "上限20文字" in (commentary_plan_issue(long_plan) or "")


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


def test_commentary_policy_notes_quick_at_triangle_marker() -> None:
    """通常の文字送りでのquickは作り直さず、文体メモにだけ残す"""
    quick = CommentaryPlan(
        comment="へぇ、そうなんだ。",
        mode="quick",
        emotion="calm",
        intensity=0.3,
        pace="normal",
    )
    assert commentary_plan_issue(quick, advance_marker="triangle") is None
    notes = commentary_plan_style_notes(quick, advance_marker="triangle")
    assert any("quick" in note for note in notes)


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


def test_commentary_policy_notes_short_page_comment() -> None:
    """ページ終端の感想が短くても作り直さず、文体メモにだけ残す"""
    short_comment = CommentaryPlan(
        comment="頼りになりそうだよね。",
        mode="quick",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    assert commentary_plan_issue(
        short_comment,
        must_speak=True,
        advance_marker="book",
    ) is None
    notes = commentary_plan_style_notes(short_comment, advance_marker="book")
    assert any("短すぎます" in note for note in notes)


def test_commentary_policy_notes_reaction_only_at_page_end() -> None:
    """ページ終端のreactionは作り直さず、文体メモにだけ残す"""
    reaction = CommentaryPlan(
        comment="うわっ！",
        mode="reaction",
        emotion="surprised",
        intensity=0.9,
        pace="fast",
    )
    assert commentary_plan_issue(
        reaction,
        must_speak=True,
        advance_marker="book",
    ) is None
    notes = commentary_plan_style_notes(reaction, advance_marker="book")
    assert any("2～3文" in note for note in notes)


def test_commentary_policy_notes_missing_soft_ending_at_page_end() -> None:
    """柔らかい語尾がなくても作り直さず、文体メモにだけ残す"""
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
    assert commentary_plan_issue(
        flat,
        must_speak=True,
        advance_marker="book",
    ) is None
    notes = commentary_plan_style_notes(flat, advance_marker="book")
    assert any("柔らかい語尾" in note for note in notes)


def test_commentary_policy_notes_blunt_sentence_ending() -> None:
    """ぶっきらぼうな語尾は作り直さず、文体メモにだけ残す"""
    blunt = CommentaryPlan(
        comment="この子、落ち着いてて頼りになりそうだな。",
        mode="quick",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    assert commentary_plan_issue(blunt) is None
    assert any(
        "ぶっきらぼうな語尾" in note
        for note in commentary_plan_style_notes(blunt)
    )


def test_commentary_policy_notes_two_sentence_reaction() -> None:
    """reactionが2文でも作り直さず、文体メモにだけ残す"""
    reaction = CommentaryPlan(
        comment="えっ、待って。今の何？",
        mode="reaction",
        emotion="surprised",
        intensity=0.9,
        pace="fast",
    )
    assert commentary_plan_issue(reaction, advance_marker="triangle") is None
    notes = commentary_plan_style_notes(reaction, advance_marker="triangle")
    assert any("2文以上" in note for note in notes)


def test_commentary_policy_notes_narrator_phrase() -> None:
    """解説口調の禁止表現は作り直さず、文体メモにだけ残す"""
    narrator = CommentaryPlan(
        comment="本文では静かな夜が続いているみたいだね。",
        mode="quick",
        emotion="calm",
        intensity=0.4,
        pace="normal",
    )
    assert commentary_plan_issue(narrator) is None
    assert any(
        "本文では" in note
        for note in commentary_plan_style_notes(narrator)
    )


def test_commentary_plan_style_notes_collects_every_deviation() -> None:
    """文体メモは最初の1件で打ち切らず、外れた点をすべて集める"""
    plan = CommentaryPlan(
        comment="この子は頼りになりそうだな。",
        mode="quick",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    notes = commentary_plan_style_notes(plan, advance_marker="book")
    assert len(notes) >= 2


def test_commentary_plan_style_notes_stay_empty_for_clean_plan() -> None:
    """狙い通りの感想には文体メモが付かない"""
    plan = CommentaryPlan(
        comment=(
            "この子、落ち着いてて頼りになりそうだよね。"
            "眼鏡も似合ってるし、まとめ役っぽいかも。"
        ),
        mode="quick",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    assert commentary_plan_style_notes(plan, advance_marker="book") == []


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


def test_commentary_plan_issue_notes_nominal_slogan_ending() -> None:
    """体言止めは作り直さず、文体メモにだけ残す"""
    plan = CommentaryPlan(
        comment="ゴーグルの向こう、可愛い予感。",
        mode="quick",
        emotion="amused",
        intensity=0.3,
        pace="normal",
    )
    assert commentary_plan_issue(plan) is None
    assert any(
        "体言止め" in note for note in commentary_plan_style_notes(plan)
    )


def test_commentary_plan_issue_rejects_long_page_comment() -> None:
    """ページ終端の感想が90文字を超えたら作り直す"""
    plan = CommentaryPlan(
        comment="この子は落ち着いていて頼りになりそうだよね。" * 5,
        mode="extended",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )
    assert "90文字" in (
        commentary_plan_issue(plan, must_speak=True, advance_marker="book")
        or ""
    )


def test_commentary_plan_issue_rejects_empty_comment() -> None:
    """喋る内容がなければ作り直す"""
    plan = CommentaryPlan(
        comment="……。",
        mode="quick",
        emotion="calm",
        intensity=0.4,
        pace="normal",
    )
    assert "有効なcomment" in (commentary_plan_issue(plan) or "")


def test_commentary_plan_fallback_speaks_at_unspoken_page_end() -> None:
    """作り直しても直らなかったsilentは、停止せず喋れる案に差し替える"""
    silent = CommentaryPlan(
        comment="",
        mode="silent",
        emotion="calm",
        intensity=0.0,
        pace="normal",
    )
    fallback = commentary_plan_fallback(
        silent,
        must_speak=True,
        advance_marker="book",
    )
    assert fallback.mode != "silent"
    assert fallback.comment
    assert commentary_plan_issue(
        fallback,
        must_speak=True,
        advance_marker="book",
    ) is None


def test_commentary_plan_fallback_trims_over_limit_comment() -> None:
    """上限を超えた感想は文の区切りで切り詰めて使う"""
    plan = CommentaryPlan(
        comment="えっ、今の何？ 本当に見間違いじゃないの？ ちょっと怖いかも。",
        mode="reaction",
        emotion="surprised",
        intensity=0.9,
        pace="fast",
    )
    fallback = commentary_plan_fallback(plan, advance_marker="triangle")
    assert len(fallback.comment) <= 20
    assert fallback.comment.startswith("えっ、今の何？")
    assert commentary_plan_issue(fallback, advance_marker="triangle") is None


def test_commentary_plan_fallback_keeps_valid_plan_untouched() -> None:
    """問題のない案はフォールバックでも書き換えない"""
    plan = CommentaryPlan(
        comment="いや、そういうものなの？",
        mode="quick",
        emotion="amused",
        intensity=0.3,
        pace="normal",
    )
    assert commentary_plan_fallback(plan) == plan


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
        ),
        persona="人間を知りたいAI。20代くらいの女性の声で話す。",
    )
    assert '"response_text": "えっ、今の何!?"' in prompt
    assert "思わず声が跳ねる大きな驚き" in prompt
    assert "20代くらいの女性" in prompt
    assert "相手へ話しかける柔らかい抑揚" in prompt
    assert "普段の会話よりリアクションを一段大きく" in prompt
    assert "少し速め" in prompt
    assert "0.80" in prompt
    assert "今回は開始挨拶ではない" in prompt


def test_commentary_speech_prompt_allows_greeting_only_for_startup() -> None:
    plan = CommentaryPlan(
        comment="ごきげんよう。スカイナです。",
        mode="extended",
        emotion="calm",
        intensity=0.6,
        pace="normal",
    )
    commentary_prompt = build_commentary_speech_prompt(plan)
    startup_prompt = build_commentary_speech_prompt(plan, startup=True)
    assert '"response_text": "スカイナです。"' in commentary_prompt
    assert "これは実況セッション開始時の最初の挨拶です" in startup_prompt
    assert "今回は開始挨拶なので" in startup_prompt
    assert "開始句を1回だけ" in startup_prompt
    assert "前置きを別途追加しない" in startup_prompt
    assert "今回は開始挨拶ではない" not in startup_prompt


def test_narration_match_ignores_spaces_and_punctuation() -> None:
    assert narration_matches(
        "ぼくの名前は、透。\n東京の大学に通う学生だ。",
        "ぼくの名前は透 東京の大学に通う学生だ",
    )
    assert not narration_matches("透です。", "真理です。")


def test_narration_match_rejects_preface_when_full_source_is_spoken() -> None:
    source = (
        "「なかなかいいですよ。リフトもそんなに混んでないし」"
        "可奈子ちゃんも真理に負けず劣らずスキー好きなのか。"
    )
    transcript = (
        "じゃあ、そのままの内容を自然に読み上げます。"
        + source
    )

    assert not narration_matches(source, transcript)


def test_narration_match_rejects_observed_choice_preface() -> None:
    source = (
        "A:「きれいだ」外国映画の男優のように、スマートに決めた。"
        "B:「君の瞳に乾杯」ハンフリー・ボガートを気取った。"
        "C:「セクシーだよ」007のように甘く危険"
    )
    transcript = "了解です。そのまま読み上げますね。" + source

    assert not narration_matches(source, transcript)


def test_narration_match_accepts_good_orthographic_variation() -> None:
    assert narration_matches(
        "このノリ、いい味出てる。",
        "このノリ、良い味出てる。",
    )


def test_session_stop_reason_waits_for_breakpoint_then_uses_grace() -> None:
    kwargs = {
        "session_started_at": 100.0,
        "duration_seconds": 1_200.0,
        "grace_seconds": 300.0,
    }

    assert (
        commentary_module._session_stop_reason(
            now=1_299.0,
            marker_kind="book",
            **kwargs,
        )
        is None
    )
    assert commentary_module._session_stop_reason(
        now=1_300.0,
        marker_kind="book",
        **kwargs,
    ) == "duration_breakpoint"
    assert commentary_module._session_stop_reason(
        now=1_350.0,
        marker_kind="choices",
        **kwargs,
    ) == "duration_breakpoint"
    assert (
        commentary_module._session_stop_reason(
            now=1_499.0,
            marker_kind="triangle",
            **kwargs,
        )
        is None
    )
    assert commentary_module._session_stop_reason(
        now=1_600.0,
        marker_kind="triangle",
        **kwargs,
    ) == "duration_grace_elapsed"


def test_capture_and_ocr_stops_at_session_hard_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image = Image.new("RGB", (960, 540), "black")
    clock = 0.0

    def fake_monotonic() -> float:
        nonlocal clock
        clock += 0.2
        return clock

    monkeypatch.setattr(commentary_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(
        commentary_module,
        "wait_for_advance_marker",
        lambda *args, **kwargs: (
            image,
            commentary_module.AdvanceMarker("none", 0.4, None),
        ),
    )
    monkeypatch.setattr(
        commentary_module,
        "capture_client",
        lambda *args, **kwargs: image,
    )
    monkeypatch.setattr(
        commentary_module,
        "_recognize_capture",
        lambda *args, **kwargs: "まだ表示中の本文",
    )
    args = commentary_module._build_parser().parse_args(
        ["--stable-ocr-samples", "99"]
    )

    text, marker = commentary_module._capture_and_ocr(
        object(),
        tmp_path,
        args,
        object(),
        hard_deadline=1.0,
    )

    assert text == "まだ表示中の本文"
    assert marker.kind == "session_end"
    assert marker.fallback_reason == "session_grace_elapsed"


def test_marker_poll_sleep_does_not_cross_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (960, 540), "black")
    clock = 0.0
    slept: list[float] = []

    monkeypatch.setattr(commentary_module.time, "monotonic", lambda: clock)

    def fake_sleep(seconds: float) -> None:
        nonlocal clock
        slept.append(seconds)
        clock += seconds

    monkeypatch.setattr(commentary_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(
        commentary_module,
        "capture_client",
        lambda *args, **kwargs: image,
    )
    monkeypatch.setattr(
        commentary_module,
        "detect_advance_marker",
        lambda *args, **kwargs: commentary_module.AdvanceMarker(
            "none",
            0.0,
            None,
        ),
    )

    _image, marker = commentary_module.wait_for_advance_marker(
        object(),
        activate=False,
        timeout=0.5,
        poll_interval=10.0,
    )

    assert marker.kind == "none"
    assert slept == [0.5]


def test_capture_and_ocr_takes_final_capture_when_deadline_already_elapsed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image = Image.new("RGB", (960, 540), "black")
    monkeypatch.setattr(commentary_module.time, "monotonic", lambda: 2.0)
    monkeypatch.setattr(
        commentary_module,
        "capture_client",
        lambda *args, **kwargs: image,
    )
    monkeypatch.setattr(
        commentary_module,
        "_recognize_capture",
        lambda *args, **kwargs: "最後に取得した本文",
    )
    monkeypatch.setattr(
        commentary_module,
        "wait_for_advance_marker",
        lambda *args, **kwargs: pytest.fail("マーク待機は行わない"),
    )
    args = commentary_module._build_parser().parse_args([])

    text, marker = commentary_module._capture_and_ocr(
        object(),
        tmp_path,
        args,
        object(),
        hard_deadline=1.0,
        deadline_fallback_reason="capture_deadline_elapsed",
    )

    assert text == "最後に取得した本文"
    assert marker.kind == "session_end"
    assert marker.fallback_reason == "capture_deadline_elapsed"


def test_title_memory_directory_is_safe_and_title_specific(tmp_path) -> None:
    first = commentary_module._safe_title_memory_dir(
        tmp_path,
        "ゲーム:第一章",
    )
    second = commentary_module._safe_title_memory_dir(
        tmp_path,
        "ゲーム?第一章",
    )

    assert first.parent == tmp_path
    assert ":" not in first.name
    assert "?" not in second.name
    assert first != second


def test_prior_commentary_memory_is_loaded_unless_initializing(
    tmp_path,
) -> None:
    memory_dir = tmp_path / "memory"
    title = "かまいたちの夜"
    overall_path = (
        commentary_module._safe_title_memory_dir(memory_dir, title)
        / "overall.json"
    )
    overall_path.parent.mkdir(parents=True)
    expected = {"schema_version": 1, "story_summary": "雪山の宿へ向かった。"}
    overall_path.write_text(
        json.dumps(expected, ensure_ascii=False),
        encoding="utf-8",
    )

    assert commentary_module._load_prior_commentary_memory(
        memory_dir,
        title,
        initialize_memory=False,
    ) == expected
    assert (
        commentary_module._load_prior_commentary_memory(
            memory_dir,
            title,
            initialize_memory=True,
        )
        is None
    )


def test_resume_startup_summarizes_memory_and_saves_speech_artifacts(
    tmp_path,
) -> None:
    planner_calls: list[dict] = []
    speech_calls: list[dict] = []

    class FakePlanner:
        def generate_text(self, **kwargs) -> TextResult:
            planner_calls.append(kwargs)
            return TextResult(
                text=json.dumps(
                    {
                        "recap": (
                            "透たちは雪山の宿へ到着し、"
                            "不穏な出来事の手がかりを探している。"
                        )
                    },
                    ensure_ascii=False,
                ),
                response_id="startup-resume-response",
            )

    class FakeRealtime:
        def speak(self, **kwargs) -> SpeechResult:
            speech_calls.append(kwargs)
            return SpeechResult(
                phase="startup",
                transcript="再開挨拶の転写",
                audio_bytes=100,
                response_id="startup-speech-response",
                started_at_seconds=1.0,
                ended_at_seconds=3.0,
            )

    root = tmp_path / "session"
    root.mkdir()
    subtitle_writer = commentary_module.SrtSubtitleWriter(
        root / "subtitles" / "commentary.srt"
    )
    commentary_module._create_startup_message(
        planner=FakePlanner(),
        prior_memory={
            "story_summary": "透たちは雪山の宿へ到着した。",
            "current_state": "宿の中を調べている。",
        },
        initial_intro_file=tmp_path / "unused.txt",
        title="かまいたちの夜",
        realtime=FakeRealtime(),
        subtitle_writer=subtitle_writer,
        root=root,
        persona="人間を知りたいAI。",
        playback=False,
        replacements=[("透", "トオル")],
    )

    record = json.loads(
        (root / "startup_message.json").read_text(encoding="utf-8")
    )
    subtitles = (
        root / "subtitles" / "commentary.srt"
    ).read_text(encoding="utf-8-sig")
    assert planner_calls[0]["phase"] == "startup_resume"
    assert planner_calls[0]["use_conversation_history"] is True
    assert record["mode"] == "resume"
    assert record["message"].startswith("ごきげんよう。")
    assert "スカイナ" in record["message"]
    assert "前回の続き" in record["message"]
    assert "雪山の宿へ到着" in record["message"]
    assert "トオル" in record["message"]
    assert "透" in record["raw_response"]
    assert speech_calls[0]["phase"] == "startup"
    assert "トオル" in speech_calls[0]["instructions"]
    assert (root / "startup_transcript.txt").read_text(
        encoding="utf-8"
    ) == "再開挨拶の転写\n"
    cue_lines = _subtitle_cue_lines(subtitles)
    assert len(cue_lines) == 3
    assert all(1 <= len(lines) <= 2 for lines in cue_lines)
    subtitle_message = "".join(
        line for lines in cue_lines for line in lines
    )
    assert subtitle_message == record["message"]


def test_startup_rejects_preface_and_retries_before_playback(tmp_path) -> None:
    message = (
        "ごきげんよう。人間を学ぶ実況AI、スカイナです。"
        "今回も前回の続きからやっていきましょう。"
    )
    invalid_transcript = (
        "ごきげんよう。それでは、続きを始めますね。" + message
    )
    calls: list[dict] = []

    class FakeRealtime:
        def speak(self, **kwargs) -> SpeechResult:
            calls.append(kwargs)
            if len(calls) == 1:
                return SpeechResult(
                    phase="startup",
                    transcript=invalid_transcript,
                    audio_bytes=100,
                    response_id="rejected",
                    playback_suppressed=True,
                )
            return SpeechResult(
                phase="startup_retry",
                transcript=message,
                audio_bytes=100,
                response_id="accepted",
                started_at_seconds=1.0,
                ended_at_seconds=3.0,
            )

    root = tmp_path / "session"
    root.mkdir()
    subtitle_path = root / "subtitles" / "commentary.srt"
    commentary_module._deliver_startup_message(
        message=message,
        mode="resume",
        realtime=FakeRealtime(),
        subtitle_writer=commentary_module.SrtSubtitleWriter(subtitle_path),
        root=root,
        persona="人間を知りたいAI。",
        playback=True,
        generation_errors=[],
        response=None,
    )

    assert [call["phase"] for call in calls] == [
        "startup",
        "startup_retry",
    ]
    assert calls[0]["playback_prefix"] == (
        "ごきげんよう。人間を学ぶ実況AI、スカイナです。"
    )
    assert "Mandatory retry correction" in calls[1]["instructions"]
    assert (root / "startup_rejected_001_transcript.txt").read_text(
        encoding="utf-8"
    ).strip() == invalid_transcript
    assert (root / "startup_transcript.txt").read_text(
        encoding="utf-8"
    ).strip() == message
    subtitles = subtitle_path.read_text(encoding="utf-8-sig")
    assert subtitles.count(" --> ") == 2
    assert all(
        1 <= len(lines) <= 2
        for lines in _subtitle_cue_lines(subtitles)
    )


def test_startup_continues_when_both_guarded_attempts_are_rejected(
    tmp_path,
) -> None:
    calls: list[dict] = []

    class FakeRealtime:
        def speak(self, **kwargs) -> SpeechResult:
            calls.append(kwargs)
            return SpeechResult(
                phase=str(kwargs["phase"]),
                transcript="それでは、始めますね。ごきげんよう。",
                audio_bytes=100,
                response_id="rejected",
                playback_suppressed=True,
            )

    root = tmp_path / "session"
    root.mkdir()
    subtitle_path = root / "subtitles" / "commentary.srt"
    commentary_module._deliver_startup_message(
        message="ごきげんよう。スカイナです。",
        mode="resume",
        realtime=FakeRealtime(),
        subtitle_writer=commentary_module.SrtSubtitleWriter(subtitle_path),
        root=root,
        persona="",
        playback=True,
        generation_errors=[],
        response=None,
    )

    assert len(calls) == 2
    assert " --> " not in subtitle_path.read_text(encoding="utf-8-sig")
    assert (root / "startup_transcript.txt").exists()


def test_initial_startup_uses_fixed_file_without_planner(tmp_path) -> None:
    intro_path = tmp_path / "intro.txt"
    intro_path.write_text(
        "ごきげんよう。人間を学ぶAI、スカイナです。"
        "今回は『{title}』を遊びます。",
        encoding="utf-8",
    )
    spoken_phases: list[str] = []

    class FakeRealtime:
        def speak(self, **kwargs) -> SpeechResult:
            spoken_phases.append(kwargs["phase"])
            return SpeechResult(
                phase="startup",
                transcript="固定挨拶",
                audio_bytes=100,
                response_id="startup-speech-response",
            )

    root = tmp_path / "session"
    root.mkdir()
    commentary_module._create_startup_message(
        planner=None,
        prior_memory=None,
        initial_intro_file=intro_path,
        title="かまいたちの夜",
        realtime=FakeRealtime(),
        subtitle_writer=commentary_module.SrtSubtitleWriter(
            root / "subtitles" / "commentary.srt"
        ),
        root=root,
        persona="",
        playback=False,
    )

    record = json.loads(
        (root / "startup_message.json").read_text(encoding="utf-8")
    )
    assert record["mode"] == "initial"
    assert record["fallback_used"] is False
    assert record["message"].startswith("ごきげんよう。")
    assert "かまいたちの夜" in record["message"]
    assert spoken_phases == ["startup"]


def test_closing_subtitle_is_split_into_two_line_timed_cues(tmp_path) -> None:
    plan = commentary_module.ClosingPlan(
        ending_line="あ。い。",
        session_impression="う。",
        call_to_action="え。",
        emotion="calm",
        intensity=0.5,
        pace="normal",
    )

    class FakeRealtime:
        def speak(self, **kwargs) -> SpeechResult:
            return SpeechResult(
                phase=str(kwargs["phase"]),
                transcript=plan.message,
                audio_bytes=100,
                response_id=None,
                started_at_seconds=10.0,
                ended_at_seconds=18.0,
            )

    root = tmp_path / "session"
    root.mkdir()
    subtitle_path = root / "subtitles" / "commentary.srt"
    commentary_module._deliver_closing_message(
        plan=plan,
        realtime=FakeRealtime(),
        subtitle_writer=commentary_module.SrtSubtitleWriter(subtitle_path),
        root=root,
        persona="",
        playback=False,
        generation_errors=[],
        response=None,
    )

    subtitles = subtitle_path.read_text(encoding="utf-8-sig")
    assert _subtitle_cue_lines(subtitles) == [
        ["あ。", "い。"],
        ["う。", "え。"],
    ]
    assert "00:00:10,000 --> 00:00:14,000" in subtitles
    assert "00:00:14,000 --> 00:00:18,000" in subtitles


def test_session_and_overall_memories_are_created_and_updated(tmp_path) -> None:
    root = tmp_path / "session"
    root.mkdir()
    memory_dir = tmp_path / "memory"
    phases: list[str] = []

    class FakePlanner:
        def generate_text(self, **kwargs) -> TextResult:
            phase = kwargs["phase"]
            phases.append(phase)
            if phase == "session_memory":
                payload = {
                    "summary": "雪山の宿へ向かい、気になる人物と出会った。",
                    "key_events": ["雪山の宿へ到着した"],
                    "characters": ["透: 主人公"],
                    "important_choices": [],
                    "unresolved_threads": ["宿で何が起きるのか"],
                    "commentator_impression": "穏やかな導入の裏が気になる。",
                    "next_start_point": "宿へ入る直前から再開する。",
                }
            else:
                assert phase == "overall_memory"
                payload = {
                    "story_summary": "透は雪山の宿へ向かった。",
                    "characters": ["透: 主人公"],
                    "important_choices": [],
                    "unresolved_threads": ["宿で何が起きるのか"],
                    "current_state": "宿へ入る直前。",
                    "commentator_perspective": "穏やかな導入の裏を疑っている。",
                    "next_start_point": "宿へ入る場面から再開する。",
                }
            return TextResult(
                text=json.dumps(payload, ensure_ascii=False),
                response_id=f"{phase}-response",
            )

    records = [{"turn": "turn_001", "game_text": "雪山の宿が見えた。"}]
    for _ in range(2):
        commentary_module._create_session_memories(
            planner=FakePlanner(),
            root=root,
            memory_dir=memory_dir,
            title="かまいたちの夜",
            memory_model="gpt-5.6-sol",
            termination_reason="duration_breakpoint",
            elapsed_seconds=1_205.0,
            records=records,
        )

    session_memory = json.loads(
        (root / "session_memory.json").read_text(encoding="utf-8")
    )
    overall_path = (
        commentary_module._safe_title_memory_dir(
            memory_dir,
            "かまいたちの夜",
        )
        / "overall.json"
    )
    overall_memory = json.loads(overall_path.read_text(encoding="utf-8"))

    assert phases == [
        "session_memory",
        "overall_memory",
        "session_memory",
        "overall_memory",
    ]
    assert session_memory["turn_count"] == 1
    assert overall_memory["session_count"] == 2
    assert not list(tmp_path.rglob("*.tmp"))


def test_session_memories_record_the_memory_model(tmp_path) -> None:
    """記憶ファイルには記憶生成に使ったモデル名を残す"""
    root = tmp_path / "session"
    root.mkdir()
    memory_dir = tmp_path / "memory"

    class FakePlanner:
        def generate_text(self, **kwargs) -> TextResult:
            if kwargs["phase"] == "session_memory":
                payload = {
                    "summary": "雪山の宿へ着いた。",
                    "key_events": [],
                    "characters": [],
                    "important_choices": [],
                    "unresolved_threads": [],
                    "commentator_impression": "続きが気になる。",
                    "next_start_point": "宿へ入る場面から再開する。",
                }
            else:
                payload = {
                    "story_summary": "透は雪山の宿へ向かった。",
                    "characters": [],
                    "important_choices": [],
                    "unresolved_threads": [],
                    "current_state": "宿へ入る直前。",
                    "commentator_perspective": "静かな導入を疑っている。",
                    "next_start_point": "宿へ入る場面から再開する。",
                }
            return TextResult(
                text=json.dumps(payload, ensure_ascii=False),
                response_id=f'{kwargs["phase"]}-response',
            )

    commentary_module._create_session_memories(
        planner=FakePlanner(),
        root=root,
        memory_dir=memory_dir,
        title="かまいたちの夜",
        memory_model="gpt-5.6-sol",
        termination_reason="duration_breakpoint",
        elapsed_seconds=1_205.0,
        records=[{"turn": "turn_001", "game_text": "雪山の宿が見えた。"}],
    )

    session_memory = json.loads(
        (root / "session_memory.json").read_text(encoding="utf-8")
    )
    overall_memory = json.loads(
        (
            commentary_module._safe_title_memory_dir(
                memory_dir,
                "かまいたちの夜",
            )
            / "overall.json"
        ).read_text(encoding="utf-8")
    )

    assert session_memory["memory_model"] == "gpt-5.6-sol"
    assert overall_memory["memory_model"] == "gpt-5.6-sol"


def test_finalize_uses_memory_planner_only_for_memory_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """締めはsummary_planner、記憶生成はmemory_plannerを使う"""
    root = tmp_path / "session"
    root.mkdir()
    summary_planner = object()
    memory_planner = object()
    used: dict[str, object] = {}

    def fake_closing(**kwargs) -> None:
        used["closing_planner"] = kwargs["planner"]

    def fake_memories(**kwargs) -> None:
        used["memory_planner"] = kwargs["planner"]
        used["memory_model"] = kwargs["memory_model"]

    monkeypatch.setattr(
        commentary_module,
        "_create_closing_message",
        fake_closing,
    )
    monkeypatch.setattr(
        commentary_module,
        "_create_session_memories",
        fake_memories,
    )

    commentary_module._finalize_timed_session(
        summary_planner=summary_planner,
        memory_planner=memory_planner,
        realtime=None,
        subtitle_writer=None,
        root=root,
        args=argparse.Namespace(
            no_playback=True,
            ocr_replacements={},
            memory_dir=tmp_path / "memory",
            title="かまいたちの夜",
            summary_model="gpt-5.6-luna",
            memory_model="gpt-5.6-sol",
            initialize_memory=False,
        ),
        persona="実況者の人格",
        termination_reason="duration_breakpoint",
        session_started_at=time.monotonic(),
    )

    assert used["closing_planner"] is summary_planner
    assert used["memory_planner"] is memory_planner
    assert used["memory_model"] == "gpt-5.6-sol"


def test_initialize_memory_starts_fresh_and_backs_up_existing_overall(
    tmp_path,
) -> None:
    root = tmp_path / "session"
    root.mkdir()
    memory_dir = tmp_path / "memory"
    title = "かまいたちの夜"
    overall_path = (
        commentary_module._safe_title_memory_dir(memory_dir, title)
        / "overall.json"
    )
    overall_path.parent.mkdir(parents=True)
    previous = {
        "schema_version": 1,
        "session_count": 7,
        "story_summary": "読み込ませない過去の展開。",
    }
    overall_path.write_text(
        json.dumps(previous, ensure_ascii=False),
        encoding="utf-8",
    )
    overall_prompts: list[str] = []

    class FakePlanner:
        def generate_text(self, **kwargs) -> TextResult:
            if kwargs["phase"] == "session_memory":
                payload = {
                    "summary": "新しく実況を始めた。",
                    "key_events": [],
                    "characters": [],
                    "important_choices": [],
                    "unresolved_threads": [],
                    "commentator_impression": "新鮮な気持ちだ。",
                    "next_start_point": "現在の画面から再開する。",
                }
            else:
                overall_prompts.append(kwargs["instructions"])
                payload = {
                    "story_summary": "新しい実況の展開。",
                    "characters": [],
                    "important_choices": [],
                    "unresolved_threads": [],
                    "current_state": "現在の画面。",
                    "commentator_perspective": "初見として見ている。",
                    "next_start_point": "現在の画面から再開する。",
                }
            return TextResult(
                text=json.dumps(payload, ensure_ascii=False),
                response_id=f'{kwargs["phase"]}-response',
            )

    commentary_module._create_session_memories(
        planner=FakePlanner(),
        root=root,
        memory_dir=memory_dir,
        title=title,
        memory_model="gpt-5.6-sol",
        termination_reason="duration_breakpoint",
        elapsed_seconds=1_205.0,
        records=[{"turn": "turn_001", "game_text": "新しい本文"}],
        initialize_memory=True,
    )

    updated = json.loads(overall_path.read_text(encoding="utf-8"))
    backups = list(
        overall_path.parent.glob("overall.before_initialize_*.json")
    )
    assert len(overall_prompts) == 1
    assert "読み込ませない過去の展開" not in overall_prompts[0]
    assert updated["session_count"] == 1
    assert updated["initialized"] is True
    assert updated["previous_memory_backup"] == str(backups[0])
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == previous


def test_initialize_memory_preserves_existing_overall_if_backup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root = tmp_path / "session"
    root.mkdir()
    memory_dir = tmp_path / "memory"
    title = "かまいたちの夜"
    overall_path = (
        commentary_module._safe_title_memory_dir(memory_dir, title)
        / "overall.json"
    )
    overall_path.parent.mkdir(parents=True)
    previous_text = '{"schema_version":1,"session_count":7}'
    overall_path.write_text(previous_text, encoding="utf-8")

    class FakePlanner:
        def generate_text(self, **kwargs) -> TextResult:
            if kwargs["phase"] == "session_memory":
                payload = {
                    "summary": "新しく実況を始めた。",
                    "key_events": [],
                    "characters": [],
                    "important_choices": [],
                    "unresolved_threads": [],
                    "commentator_impression": "新鮮な気持ちだ。",
                    "next_start_point": "現在の画面から再開する。",
                }
            else:
                payload = {
                    "story_summary": "新しい実況の展開。",
                    "characters": [],
                    "important_choices": [],
                    "unresolved_threads": [],
                    "current_state": "現在の画面。",
                    "commentator_perspective": "初見として見ている。",
                    "next_start_point": "現在の画面から再開する。",
                }
            return TextResult(
                text=json.dumps(payload, ensure_ascii=False),
                response_id=f'{kwargs["phase"]}-response',
            )

    monkeypatch.setattr(
        commentary_module.shutil,
        "copy2",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("バックアップ先へ書き込めない")
        ),
    )

    commentary_module._create_session_memories(
        planner=FakePlanner(),
        root=root,
        memory_dir=memory_dir,
        title=title,
        memory_model="gpt-5.6-sol",
        termination_reason="duration_breakpoint",
        elapsed_seconds=1_205.0,
        records=[{"turn": "turn_001", "game_text": "新しい本文"}],
        initialize_memory=True,
    )

    assert overall_path.read_text(encoding="utf-8") == previous_text
    pending = json.loads(
        (root / "overall_memory_pending.json").read_text(encoding="utf-8")
    )
    assert "バックアップできません" in pending["generation_errors"][0]
    assert pending["generated_overall_memory"]["session_count"] == 1


def test_memory_failure_saves_pending_source_without_raising(tmp_path) -> None:
    root = tmp_path / "session"
    root.mkdir()

    class FailingPlanner:
        def generate_text(self, **kwargs) -> TextResult:
            raise RuntimeError("一時的なAPI障害")

    commentary_module._create_session_memories(
        planner=FailingPlanner(),
        root=root,
        memory_dir=tmp_path / "memory",
        title="かまいたちの夜",
        memory_model="gpt-5.6-sol",
        termination_reason="duration_breakpoint",
        elapsed_seconds=1_205.0,
        records=[{"turn": "turn_001", "game_text": "雪が降っていた。"}],
    )

    pending = json.loads(
        (root / "session_memory_pending.json").read_text(encoding="utf-8")
    )
    assert pending["session_records"][0]["game_text"] == "雪が降っていた。"
    assert len(pending["generation_errors"]) == 2
    assert not (root / "session_memory.json").exists()


def test_invalid_existing_overall_memory_is_preserved(
    tmp_path,
) -> None:
    root = tmp_path / "session"
    root.mkdir()
    memory_dir = tmp_path / "memory"
    overall_path = (
        commentary_module._safe_title_memory_dir(
            memory_dir,
            "かまいたちの夜",
        )
        / "overall.json"
    )
    overall_path.parent.mkdir(parents=True)
    overall_path.write_text("{壊れた記憶", encoding="utf-8")
    phases: list[str] = []

    class FakePlanner:
        def generate_text(self, **kwargs) -> TextResult:
            phase = kwargs["phase"]
            phases.append(phase)
            assert phase == "session_memory"
            return TextResult(
                text=json.dumps(
                    {
                        "summary": "今回のまとめ。",
                        "key_events": [],
                        "characters": [],
                        "important_choices": [],
                        "unresolved_threads": [],
                        "commentator_impression": "続きが気になる。",
                        "next_start_point": "現在の画面から再開する。",
                    },
                    ensure_ascii=False,
                ),
                response_id="session-memory-response",
            )

    commentary_module._create_session_memories(
        planner=FakePlanner(),
        root=root,
        memory_dir=memory_dir,
        title="かまいたちの夜",
        memory_model="gpt-5.6-sol",
        termination_reason="duration_breakpoint",
        elapsed_seconds=1_205.0,
        records=[{"turn": "turn_001", "game_text": "本文"}],
    )

    assert phases == ["session_memory"]
    assert overall_path.read_text(encoding="utf-8") == "{壊れた記憶"
    assert (root / "overall_memory_pending.json").exists()


def test_sleep_with_deadline_caps_long_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = 10.0
    slept: list[float] = []

    def fake_monotonic() -> float:
        return clock

    def fake_sleep(seconds: float) -> None:
        nonlocal clock
        slept.append(seconds)
        clock += seconds

    monkeypatch.setattr(commentary_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(commentary_module.time, "sleep", fake_sleep)

    assert commentary_module._sleep_with_deadline(30.0, 12.5) is True
    assert slept == [2.5]


def test_choice_speech_retry_is_bounded_by_session_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeRealtime:
        def speak(self, **kwargs) -> SpeechResult:
            return SpeechResult(
                phase="choice",
                transcript="まったく違う発話です。",
                audio_bytes=100,
                response_id="choice-response",
            )

    monkeypatch.setattr(commentary_module.time, "monotonic", lambda: 10.0)
    plan = ChoicePlan(
        selected_label="B",
        opinion="こちらの展開を見てみたいよね。",
        emotion="thoughtful",
        intensity=0.6,
        pace="normal",
    )

    with pytest.raises(commentary_module.SessionEndingRequested) as raised:
        speak_choice_with_retries(
            FakeRealtime(),
            plan,
            turn_dir=tmp_path,
            playback=False,
            retries=0,
            retry_delay=0.0,
            hard_deadline=5.0,
        )

    assert raised.value.retry_count == 0
    assert raised.value.verification.matches is False


def test_timed_book_ending_does_not_press_enter_and_creates_memories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events: list[str] = []
    clock = 0.0
    session_start_times: list[float] = []
    timeline_origins: list[float | None] = []
    source_text = "ここで物語が大きく動いた。"

    class FakeObsWindow:
        error = None
        is_open = True

        def __init__(self, *, enabled: bool) -> None:
            assert enabled is True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def mark_ready(self) -> None:
            events.append("ready")

        def wait_for_start(self) -> float:
            nonlocal clock
            events.append("start_click")
            clock = 10.0
            return clock

    class FakeOcr:
        initialization_seconds = 0.0

    class FakeRealtimeClient:
        def __init__(self, **kwargs) -> None:
            self.timeline_origin = kwargs["timeline_origin"]
            timeline_origins.append(self.timeline_origin)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def speak(self, **kwargs) -> SpeechResult:
            nonlocal clock
            phase = kwargs["phase"]
            events.append(phase)
            if phase == "startup":
                assert self.timeline_origin == 10.0
                clock = 30.0
            transcript = (
                source_text
                if phase == "narration"
                else "締めの発話"
                if phase == "closing"
                else "ここまで一気に話が動いて驚いたよね。"
            )
            return SpeechResult(
                phase=phase,
                transcript=transcript,
                audio_bytes=100,
                response_id=f"{phase}-response",
                started_at_seconds=1.0,
                ended_at_seconds=2.0,
            )

    class FakePlanner:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def generate_text(self, **kwargs) -> TextResult:
            phase = kwargs["phase"]
            if phase == "commentary_plan":
                payload = {
                    "mode": "quick",
                    "comment": (
                        "ここまで一気に話が動いて、かなり驚いたよね。"
                        "まだ先に何が待っているのか、すごく気になるかも。"
                    ),
                    "emotion": "surprised",
                    "intensity": 0.7,
                    "pace": "normal",
                }
            elif phase == "closing_plan":
                payload = {
                    "ending_line": (
                        "ちょうど区切りがいいので、今日はこの辺で"
                        "終わっておきましょうか。"
                    ),
                    "session_impression": (
                        "思いがけない展開が続いて、この先がますます"
                        "気になってきました。"
                    ),
                    "call_to_action": (
                        "動画を気に入ってくれたら、チャンネル登録と高評価を"
                        "お願いします。それでは、また次回！"
                    ),
                    "emotion": "thoughtful",
                    "intensity": 0.7,
                    "pace": "normal",
                }
            elif phase == "session_memory":
                payload = {
                    "summary": "物語が大きく動いた。",
                    "key_events": ["物語が大きく動いた"],
                    "characters": [],
                    "important_choices": [],
                    "unresolved_threads": ["次の展開"],
                    "commentator_impression": "続きが気になる。",
                    "next_start_point": "現在の画面から再開する。",
                }
            else:
                assert phase == "overall_memory"
                payload = {
                    "story_summary": "物語が大きく動いた。",
                    "characters": [],
                    "important_choices": [],
                    "unresolved_threads": ["次の展開"],
                    "current_state": "区切りの画面。",
                    "commentator_perspective": "続きが気になる。",
                    "next_start_point": "現在の画面から再開する。",
                }
            return TextResult(
                text=json.dumps(payload, ensure_ascii=False),
                response_id=f"{phase}-response",
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        commentary_module.time,
        "monotonic",
        lambda: clock,
    )
    monkeypatch.setattr(commentary_module, "ObsCaptureWindow", FakeObsWindow)
    monkeypatch.setattr(commentary_module, "PersistentNdlOcr", FakeOcr)
    monkeypatch.setattr(
        commentary_module,
        "RealtimeSpeechClient",
        FakeRealtimeClient,
    )
    monkeypatch.setattr(
        commentary_module,
        "ResponsesCommentaryPlanner",
        FakePlanner,
    )
    monkeypatch.setattr(
        commentary_module,
        "find_window",
        lambda title: commentary_module.WindowInfo(123, title, 960, 540),
    )
    monkeypatch.setattr(
        commentary_module,
        "_capture_and_ocr",
        lambda *args, **kwargs: (
            source_text,
            commentary_module.AdvanceMarker("book", 1.0, None),
        ),
    )

    def fake_stop_reason(**kwargs) -> str:
        session_start_times.append(kwargs["session_started_at"])
        return "duration_breakpoint"

    monkeypatch.setattr(
        commentary_module,
        "_session_stop_reason",
        fake_stop_reason,
    )
    monkeypatch.setattr(
        commentary_module,
        "press_enter",
        lambda hwnd: events.append("enter"),
    )

    result = commentary_module.main(
        [
            "--press-enter",
            "--output",
            str(tmp_path / "session"),
            "--memory-dir",
            str(tmp_path / "memory"),
        ]
    )

    assert result == 0
    assert "enter" not in events
    assert events == [
        "ready",
        "start_click",
        "startup",
        "narration",
        "commentary",
        "closing",
    ]
    assert timeline_origins == [10.0]
    assert session_start_times == [10.0]
    turn_result = json.loads(
        (tmp_path / "session" / "turn_001" / "result.json").read_text(
            encoding="utf-8"
        )
    )
    assert turn_result["enter_pressed"] is False
    assert turn_result["enter_blocked_reason"] == "session_ending"
    assert (tmp_path / "session" / "closing_plan.json").exists()
    assert (tmp_path / "session" / "session_memory.json").exists()


def test_timed_choice_ending_does_not_plan_or_perform_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source_text = "どちらへ行く？\nA: 食堂\nB: 客室"
    phases: list[str] = []
    selections: list[int] = []

    class FakeObsWindow:
        error = None
        is_open = False

        def __init__(self, *, enabled: bool) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

    class FakeOcr:
        initialization_seconds = 0.0

    class FakeRealtimeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def speak(self, **kwargs) -> SpeechResult:
            phase = kwargs["phase"]
            phases.append(phase)
            return SpeechResult(
                phase=phase,
                transcript=source_text if phase == "narration" else "締め",
                audio_bytes=100,
                response_id=f"{phase}-response",
                started_at_seconds=1.0,
                ended_at_seconds=2.0,
            )

    class FakePlanner:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def generate_text(self, **kwargs) -> TextResult:
            phase = kwargs["phase"]
            assert not phase.startswith("choice_plan")
            if phase == "closing_plan":
                payload = {
                    "ending_line": "選択肢が出たところで、今日はこの辺にしましょうか。",
                    "session_impression": (
                        "どちらへ進むか、次回までゆっくり考えるのも楽しそうですね。"
                    ),
                    "call_to_action": (
                        "動画を気に入ってくれたら、チャンネル登録と高評価を"
                        "お願いします。それでは、また次回！"
                    ),
                    "emotion": "thoughtful",
                    "intensity": 0.65,
                    "pace": "normal",
                }
            elif phase == "session_memory":
                payload = {
                    "summary": "食堂か客室かを選ぶ場面まで進んだ。",
                    "key_events": ["選択肢が表示された"],
                    "characters": [],
                    "important_choices": ["食堂か客室かは未選択"],
                    "unresolved_threads": ["どちらへ進むか"],
                    "commentator_impression": "次回まで考えたい。",
                    "next_start_point": "未選択の画面から再開する。",
                }
            else:
                assert phase == "overall_memory"
                payload = {
                    "story_summary": "移動先を選ぶところまで進んだ。",
                    "characters": [],
                    "important_choices": ["食堂か客室かは未選択"],
                    "unresolved_threads": ["どちらへ進むか"],
                    "current_state": "選択肢を表示中。",
                    "commentator_perspective": "次回まで考えたい。",
                    "next_start_point": "未選択の画面から再開する。",
                }
            return TextResult(
                text=json.dumps(payload, ensure_ascii=False),
                response_id=f"{phase}-response",
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(commentary_module, "ObsCaptureWindow", FakeObsWindow)
    monkeypatch.setattr(commentary_module, "PersistentNdlOcr", FakeOcr)
    monkeypatch.setattr(
        commentary_module,
        "RealtimeSpeechClient",
        FakeRealtimeClient,
    )
    monkeypatch.setattr(
        commentary_module,
        "ResponsesCommentaryPlanner",
        FakePlanner,
    )
    monkeypatch.setattr(
        commentary_module,
        "find_window",
        lambda title: commentary_module.WindowInfo(123, title, 960, 540),
    )
    monkeypatch.setattr(
        commentary_module,
        "_capture_and_ocr",
        lambda *args, **kwargs: (
            source_text,
            commentary_module.AdvanceMarker("choices", 1.0, None),
        ),
    )
    monkeypatch.setattr(
        commentary_module,
        "_session_stop_reason",
        lambda **kwargs: "duration_breakpoint",
    )
    monkeypatch.setattr(
        commentary_module,
        "select_choice",
        lambda hwnd, index: selections.append(index),
    )

    result = commentary_module.main(
        [
            "--press-enter",
            "--no-playback",
            "--no-obs-window",
            "--output",
            str(tmp_path / "session"),
            "--memory-dir",
            str(tmp_path / "memory"),
        ]
    )

    assert result == 0
    assert selections == []
    assert phases == ["startup", "narration", "closing"]
    turn_result = json.loads(
        (tmp_path / "session" / "turn_001" / "result.json").read_text(
            encoding="utf-8"
        )
    )
    assert turn_result["choice_plan"] is None
    assert turn_result["selection_performed"] is False


def test_summary_client_start_failure_still_speaks_fallback_closing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source_text = "ここで一区切り。"
    phases: list[str] = []

    class FakeObsWindow:
        error = None
        is_open = False

        def __init__(self, *, enabled: bool) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

    class FakeOcr:
        initialization_seconds = 0.0

    class FakeRealtimeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            pass

        def speak(self, **kwargs) -> SpeechResult:
            phase = kwargs["phase"]
            phases.append(phase)
            return SpeechResult(
                phase=phase,
                transcript=(
                    source_text if phase == "narration" else "定型の締め"
                ),
                audio_bytes=100,
                response_id=f"{phase}-response",
                started_at_seconds=1.0,
                ended_at_seconds=2.0,
            )

    class FailingPlanner:
        def __init__(self, **kwargs) -> None:
            raise OSError("クライアント初期化失敗")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(commentary_module, "ObsCaptureWindow", FakeObsWindow)
    monkeypatch.setattr(commentary_module, "PersistentNdlOcr", FakeOcr)
    monkeypatch.setattr(
        commentary_module,
        "RealtimeSpeechClient",
        FakeRealtimeClient,
    )
    monkeypatch.setattr(
        commentary_module,
        "ResponsesCommentaryPlanner",
        FailingPlanner,
    )
    monkeypatch.setattr(
        commentary_module,
        "find_window",
        lambda title: commentary_module.WindowInfo(123, title, 960, 540),
    )
    monkeypatch.setattr(
        commentary_module,
        "_capture_and_ocr",
        lambda *args, **kwargs: (
            source_text,
            commentary_module.AdvanceMarker("book", 1.0, None),
        ),
    )
    monkeypatch.setattr(
        commentary_module,
        "_session_stop_reason",
        lambda **kwargs: "duration_breakpoint",
    )

    root = tmp_path / "session"
    result = commentary_module.main(
        [
            "--press-enter",
            "--narration-only",
            "--no-playback",
            "--no-obs-window",
            "--output",
            str(root),
            "--memory-dir",
            str(tmp_path / "memory"),
        ]
    )

    assert result == 0
    assert phases == ["startup", "narration", "closing"]
    closing_plan = json.loads(
        (root / "closing_plan.json").read_text(encoding="utf-8")
    )
    assert closing_plan["fallback_used"] is True
    assert (root / "session_memory_pending.json").exists()


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
