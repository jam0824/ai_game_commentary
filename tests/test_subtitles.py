import pytest

from game_window_ocr.subtitles import SrtSubtitleWriter, format_srt_timestamp


def test_format_srt_timestamp_supports_long_recordings() -> None:
    assert format_srt_timestamp(90_061.007) == "25:01:01,007"


def test_srt_writer_uses_utf8_bom_crlf_and_sequential_cues(tmp_path) -> None:
    path = tmp_path / "subtitles" / "commentary.srt"
    writer = SrtSubtitleWriter(path)

    writer.add_cue(
        "驚いたね。\r\n次も気になる！",
        start_seconds=1.2344,
        end_seconds=2.3456,
    )
    writer.add_cue(
        "ここはBを選ぶね。",
        start_seconds=3.0,
        end_seconds=4.25,
    )

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8-sig") == (
        "1\r\n"
        "00:00:01,234 --> 00:00:02,346\r\n"
        "驚いたね。\r\n"
        "次も気になる！\r\n"
        "\r\n"
        "2\r\n"
        "00:00:03,000 --> 00:00:04,250\r\n"
        "ここはBを選ぶね。\r\n"
        "\r\n"
    )


def test_empty_srt_is_created_immediately(tmp_path) -> None:
    path = tmp_path / "subtitles" / "commentary.srt"

    SrtSubtitleWriter(path)

    assert path.read_bytes() == b"\xef\xbb\xbf"


def test_srt_writer_breaks_lines_after_sentence_end_marks(tmp_path) -> None:
    path = tmp_path / "commentary.srt"
    writer = SrtSubtitleWriter(path)

    writer.add_cue(
        (
            "トオル、かなり限界まで来てるじゃん……！"
            "このまま二人きりは危ないかもね。\n"
            "本当に大丈夫！？「まだ平気!」 次へ行こう?"
        ),
        start_seconds=0.0,
        end_seconds=1.0,
    )

    assert path.read_text(encoding="utf-8-sig") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "トオル、かなり限界まで来てるじゃん……！\n"
        "このまま二人きりは危ないかもね。\n"
        "本当に大丈夫！？\n"
        "「まだ平気!」\n"
        "次へ行こう?\n"
        "\n"
    )


def test_srt_writer_groups_lines_and_distributes_time_by_text_length(
    tmp_path,
) -> None:
    path = tmp_path / "commentary.srt"
    writer = SrtSubtitleWriter(path)

    writer.add_grouped_cues(
        "あ。い。う。え。お。か。き。",
        start_seconds=1.0,
        end_seconds=15.0,
        max_lines_per_cue=2,
    )

    assert path.read_bytes().decode("utf-8-sig") == (
        "1\r\n"
        "00:00:01,000 --> 00:00:05,000\r\n"
        "あ。\r\n"
        "い。\r\n"
        "\r\n"
        "2\r\n"
        "00:00:05,000 --> 00:00:09,000\r\n"
        "う。\r\n"
        "え。\r\n"
        "\r\n"
        "3\r\n"
        "00:00:09,000 --> 00:00:13,000\r\n"
        "お。\r\n"
        "か。\r\n"
        "\r\n"
        "4\r\n"
        "00:00:13,000 --> 00:00:15,000\r\n"
        "き。\r\n"
        "\r\n"
    )


def test_srt_writer_keeps_one_or_two_lines_in_one_group(tmp_path) -> None:
    path = tmp_path / "commentary.srt"
    writer = SrtSubtitleWriter(path)

    writer.add_grouped_cues(
        "最初の文。次の文。",
        start_seconds=2.0,
        end_seconds=4.0,
        max_lines_per_cue=2,
    )

    subtitles = path.read_text(encoding="utf-8-sig")
    assert subtitles.count(" --> ") == 1
    assert "最初の文。\n次の文。" in subtitles


def test_srt_writer_falls_back_to_one_cue_when_duration_is_too_short(
    tmp_path,
) -> None:
    path = tmp_path / "commentary.srt"
    writer = SrtSubtitleWriter(path)

    writer.add_grouped_cues(
        "あ。い。う。",
        start_seconds=3.0,
        end_seconds=3.0,
        max_lines_per_cue=2,
    )

    assert path.read_text(encoding="utf-8-sig") == (
        "1\n"
        "00:00:03,000 --> 00:00:03,001\n"
        "あ。\n"
        "い。\n"
        "う。\n"
        "\n"
    )


def test_srt_writer_grouped_cues_ignore_empty_text(tmp_path) -> None:
    path = tmp_path / "commentary.srt"
    writer = SrtSubtitleWriter(path)

    writer.add_grouped_cues(
        " \r\n ",
        start_seconds=0.0,
        end_seconds=1.0,
        max_lines_per_cue=2,
    )

    assert path.read_bytes() == b"\xef\xbb\xbf"


def test_srt_writer_grouped_cues_validate_line_limit_and_time(tmp_path) -> None:
    writer = SrtSubtitleWriter(tmp_path / "commentary.srt")

    with pytest.raises(ValueError, match="最大行数"):
        writer.add_grouped_cues(
            "字幕。",
            start_seconds=0.0,
            end_seconds=1.0,
            max_lines_per_cue=0,
        )
    with pytest.raises(ValueError, match="終了時刻"):
        writer.add_grouped_cues(
            "字幕。",
            start_seconds=2.0,
            end_seconds=1.0,
            max_lines_per_cue=2,
        )
