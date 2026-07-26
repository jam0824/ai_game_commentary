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
