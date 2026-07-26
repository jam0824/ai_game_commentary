from __future__ import annotations

import math
import re
from pathlib import Path


_SENTENCE_END = re.compile(
    r"([。！？!?]+)([」』】）》〉〕〗〙〛]*)(?:[ \t]*)(?=\S)"
)


def _break_lines_after_sentence_end(text: str) -> str:
    return _SENTENCE_END.sub(r"\1\2\n", text)


def _format_srt_milliseconds(total_milliseconds: int) -> str:
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1_000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},"
        f"{milliseconds:03d}"
    )


def format_srt_timestamp(seconds: float) -> str:
    if not math.isfinite(seconds):
        raise ValueError("字幕時刻は有限の数値で指定してください。")

    total_milliseconds = max(0, int(seconds * 1000 + 0.5))
    return _format_srt_milliseconds(total_milliseconds)


class SrtSubtitleWriter:
    """Append completed speech cues to a UTF-8 SRT file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as output:
            output.write("\ufeff")
        self._next_index = 1

    def add_cue(
        self,
        text: str,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        normalized_text = (
            text.replace("\r\n", "\n").replace("\r", "\n").strip()
        )
        normalized_text = _break_lines_after_sentence_end(normalized_text)
        if not normalized_text:
            return
        if end_seconds < start_seconds:
            raise ValueError("字幕の終了時刻は開始時刻以降にしてください。")

        start_milliseconds = max(0, int(start_seconds * 1000 + 0.5))
        end_milliseconds = max(0, int(end_seconds * 1000 + 0.5))
        if end_milliseconds <= start_milliseconds:
            end_milliseconds = start_milliseconds + 1

        srt_text = normalized_text.replace("\n", "\r\n")
        cue = (
            f"{self._next_index}\r\n"
            f"{_format_srt_milliseconds(start_milliseconds)} --> "
            f"{_format_srt_milliseconds(end_milliseconds)}\r\n"
            f"{srt_text}\r\n\r\n"
        )
        with self.path.open("a", encoding="utf-8", newline="") as output:
            output.write(cue)
            output.flush()
        self._next_index += 1
