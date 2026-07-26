from __future__ import annotations

import math
import re
from pathlib import Path


_SENTENCE_END = re.compile(
    r"([。！？!?]+)([」』】）》〉〕〗〙〛]*)(?:[ \t]*)(?=\S)"
)


def _break_lines_after_sentence_end(text: str) -> str:
    return _SENTENCE_END.sub(r"\1\2\n", text)


def _normalize_subtitle_text(text: str) -> str:
    normalized_text = (
        text.replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    return _break_lines_after_sentence_end(normalized_text)


def _non_whitespace_length(text: str) -> int:
    return sum(not char.isspace() for char in text)


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
        normalized_text = _normalize_subtitle_text(text)
        if not normalized_text:
            return
        if end_seconds < start_seconds:
            raise ValueError("字幕の終了時刻は開始時刻以降にしてください。")

        start_milliseconds = max(0, int(start_seconds * 1000 + 0.5))
        end_milliseconds = max(0, int(end_seconds * 1000 + 0.5))
        if end_milliseconds <= start_milliseconds:
            end_milliseconds = start_milliseconds + 1

        self._append_cue(
            normalized_text,
            start_milliseconds=start_milliseconds,
            end_milliseconds=end_milliseconds,
        )

    def add_grouped_cues(
        self,
        text: str,
        *,
        start_seconds: float,
        end_seconds: float,
        max_lines_per_cue: int,
    ) -> None:
        if max_lines_per_cue < 1:
            raise ValueError("字幕の最大行数は1以上にしてください。")

        normalized_text = _normalize_subtitle_text(text)
        if not normalized_text:
            return
        if end_seconds < start_seconds:
            raise ValueError("字幕の終了時刻は開始時刻以降にしてください。")

        lines = [
            line.strip()
            for line in normalized_text.splitlines()
            if line.strip()
        ]
        groups = [
            "\n".join(lines[index : index + max_lines_per_cue])
            for index in range(0, len(lines), max_lines_per_cue)
        ]
        start_milliseconds = max(0, int(start_seconds * 1000 + 0.5))
        end_milliseconds = max(0, int(end_seconds * 1000 + 0.5))
        if end_milliseconds <= start_milliseconds:
            end_milliseconds = start_milliseconds + 1

        duration_milliseconds = end_milliseconds - start_milliseconds
        if len(groups) == 1 or duration_milliseconds < len(groups):
            self._append_cue(
                normalized_text,
                start_milliseconds=start_milliseconds,
                end_milliseconds=end_milliseconds,
            )
            return

        weights = [
            max(1, _non_whitespace_length(group))
            for group in groups
        ]
        total_weight = sum(weights)
        cumulative_weight = 0
        cue_start = start_milliseconds
        for group_index, (group, weight) in enumerate(
            zip(groups, weights, strict=True)
        ):
            cumulative_weight += weight
            remaining_groups = len(groups) - group_index - 1
            if remaining_groups == 0:
                cue_end = end_milliseconds
            else:
                proportional_end = start_milliseconds + int(
                    duration_milliseconds
                    * cumulative_weight
                    / total_weight
                    + 0.5
                )
                cue_end = min(
                    max(proportional_end, cue_start + 1),
                    end_milliseconds - remaining_groups,
                )
            self._append_cue(
                group,
                start_milliseconds=cue_start,
                end_milliseconds=cue_end,
            )
            cue_start = cue_end

    def _append_cue(
        self,
        text: str,
        *,
        start_milliseconds: int,
        end_milliseconds: int,
    ) -> None:
        srt_text = text.replace("\n", "\r\n")
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
