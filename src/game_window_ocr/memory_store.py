"""ゲームタイトル別の全体記憶ファイルの保存・退避・ロールバックを扱う。

撮影に失敗したときへの備えとして、全体記憶を上書きする直前の内容を
1世代だけ ``overall.previous.json`` に残す。ロールバックは現行と
1回前を入れ替えるだけなので、誤って実行しても再実行で元へ戻せる。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OVERALL_FILENAME = "overall.json"
PREVIOUS_FILENAME = "overall.previous.json"


class RollbackError(RuntimeError):
    """ロールバックできない状態を表す。"""


@dataclass(frozen=True)
class OverallMemoryPaths:
    directory: Path
    overall: Path
    previous: Path


@dataclass(frozen=True)
class RollbackResult:
    paths: OverallMemoryPaths
    restored: dict[str, Any]
    replaced: dict[str, Any] | None


def safe_title_memory_dir(memory_dir: Path, title: str) -> Path:
    normalized = unicodedata.normalize("NFKC", title).strip()
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", normalized)
    safe = re.sub(r"\s+", "_", safe).strip(" ._")[:80] or "game"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return memory_dir / f"{safe}_{digest}"


def overall_memory_paths(memory_dir: Path, title: str) -> OverallMemoryPaths:
    directory = safe_title_memory_dir(memory_dir, title)
    return OverallMemoryPaths(
        directory=directory,
        overall=directory / OVERALL_FILENAME,
        previous=directory / PREVIOUS_FILENAME,
    )


def read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(
            f"既存の全体記憶を読み込めません: {path} ({exc})"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"既存の全体記憶がJSONオブジェクトではありません: {path}"
        )
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def backup_overall_memory(overall_path: Path) -> Path | None:
    """上書き直前の全体記憶を1世代だけ退避する。古い退避は上書きする。"""
    if not overall_path.exists():
        return None
    previous_path = overall_path.with_name(PREVIOUS_FILENAME)
    temporary_path = previous_path.with_suffix(previous_path.suffix + ".tmp")
    temporary_path.write_bytes(overall_path.read_bytes())
    temporary_path.replace(previous_path)
    return previous_path


def rollback_overall_memory(memory_dir: Path, title: str) -> RollbackResult:
    """現行の全体記憶と1回前の全体記憶を入れ替える。"""
    paths = overall_memory_paths(memory_dir, title)
    if not paths.previous.exists():
        raise RollbackError(
            f"1回前の全体記憶がありません: {paths.previous}"
        )
    try:
        restored = read_json_object(paths.previous)
        replaced = read_json_object(paths.overall)
    except ValueError as exc:
        raise RollbackError(str(exc)) from exc
    if restored is None:
        raise RollbackError(
            f"1回前の全体記憶がありません: {paths.previous}"
        )

    swap_path = paths.overall.with_suffix(paths.overall.suffix + ".swap")
    if replaced is None:
        paths.previous.replace(paths.overall)
    else:
        paths.overall.replace(swap_path)
        paths.previous.replace(paths.overall)
        swap_path.replace(paths.previous)
    return RollbackResult(paths=paths, restored=restored, replaced=replaced)
