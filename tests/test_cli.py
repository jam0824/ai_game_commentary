import json

import pytest
from PIL import Image

from game_window_ocr.cli import clean_ocr_text, parse_crop, prepare_ocr_image
from game_window_ocr.commentary import (
    CommentaryPlan,
    build_commentary_prompt,
    build_commentary_speech_prompt,
    build_narration_prompt,
    collapse_visual_line_breaks,
    commentary_plan_issue,
    extract_incremental_text,
    narration_matches,
    parse_commentary_plan,
)
from game_window_ocr.windows import normalize_title


def test_title_normalization_matches_ascii_and_full_width() -> None:
    assert normalize_title("かまいたちの夜x3") == normalize_title(
        "かまいたちの夜×３"
    )


def test_parse_crop() -> None:
    assert parse_crop("10, 20, 300, 200") == (10, 20, 300, 200)


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
    prompt = build_commentary_prompt("雪が降っていた。")
    assert '"new_game_text": "雪が降っていた。"' in prompt
    assert "8～35文字" in prompt
    assert "最大2文・合計90文字" in prompt
    assert "基本はquick" in prompt
    assert "自然な実況口調" in prompt
    assert "もう一度朗読しない" in prompt
    assert '"length_mode":"quick"' in prompt


def test_extract_incremental_text_returns_only_appended_screen_text() -> None:
    previous = "ぼくは真理を見つめた。"
    current = "ぼくは真理を見つめた。\n彼女は注目の的だった。"
    assert extract_incremental_text(previous, current) == "彼女は注目の的だった。"


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
        length_mode="quick",
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
        length_mode="quick",
        emotion="thoughtful",
        intensity=1.0,
        pace="normal",
    )


def test_commentary_plan_issue_rejects_long_quick_comment() -> None:
    plan = CommentaryPlan(
        comment="これは一息では言えないくらい長い通常場面の説明的な感想になっていて短く直す必要があります。",
        length_mode="quick",
        emotion="calm",
        intensity=0.3,
        pace="normal",
    )
    assert "上限35文字" in (commentary_plan_issue(plan) or "")


def test_commentary_plan_issue_accepts_natural_quick_comment() -> None:
    plan = CommentaryPlan(
        comment="いや、そういうものなの？",
        length_mode="quick",
        emotion="amused",
        intensity=0.3,
        pace="normal",
    )
    assert commentary_plan_issue(plan) is None


def test_commentary_plan_issue_rejects_nominal_slogan_ending() -> None:
    plan = CommentaryPlan(
        comment="ゴーグルの向こう、可愛い予感。",
        length_mode="quick",
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
        length_mode="quick",
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
        length_mode="quick",
        emotion="tense",
        intensity=0.65,
        pace="slow",
    )


def test_commentary_speech_prompt_uses_selected_delivery() -> None:
    prompt = build_commentary_speech_prompt(
        CommentaryPlan(
            comment="えっ、今の何!?",
            length_mode="quick",
            emotion="surprised",
            intensity=0.8,
            pace="fast",
        )
    )
    assert '"response_text": "えっ、今の何!?"' in prompt
    assert "本当に意外だったような驚き" in prompt
    assert "少し速め" in prompt
    assert "0.80" in prompt


def test_narration_match_ignores_spaces_and_punctuation() -> None:
    assert narration_matches(
        "ぼくの名前は、透。\n東京の大学に通う学生だ。",
        "ぼくの名前は透 東京の大学に通う学生だ",
    )
    assert not narration_matches("透です。", "真理です。")


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
