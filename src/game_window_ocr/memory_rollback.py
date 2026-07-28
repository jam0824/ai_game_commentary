"""全体記憶を1回前の状態へ戻す単体スクリプト。

撮影に失敗した回の記憶を取り消すために使う。実況側は全体記憶を
上書きする直前に ``overall.previous.json`` を残しているので、この
スクリプトは現行とその1世代前を入れ替える。保持しているのは1世代
だけなので、それより前へは戻せない。誤って実行した場合はもう一度
実行すれば元の状態へ戻る。
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

from .memory_store import (
    OverallMemoryPaths,
    RollbackError,
    overall_memory_paths,
    read_json_object,
    rollback_overall_memory,
)

DEFAULT_CONFIG_PATH = Path("game-commentary.toml")
DEFAULT_TITLE = "ゲーム"
DEFAULT_MEMORY_DIR = "output/memory"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        payload = tomllib.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("設定ファイルのトップレベルはテーブルにしてください。")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "全体記憶を1回前の状態へ戻します。"
            "保持しているのは1世代だけで、それより前へは戻せません。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "設定ファイル。titleとmemory_dirの既定値をここから読みます。"
            f"既定は {DEFAULT_CONFIG_PATH}。"
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="対象のゲームタイトル。既定は設定ファイルのtitle。",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=None,
        help=(
            "全体記憶のフォルダ。既定は設定ファイルのmemory_dirで、"
            "相対パスは設定ファイルの場所を基準にします。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="入れ替え内容を表示するだけで、ファイルは書き換えません。",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="確認プロンプトを表示せずに実行します。",
    )
    return parser


def _resolve_args(argv: list[str] | None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)

    config_values: dict[str, Any] = {}
    if args.config.exists():
        config_values = _load_config(args.config)
    elif args.title is None or args.memory_dir is None:
        raise ValueError(
            f"設定ファイルが見つかりません: {args.config}"
            "（--title と --memory-dir を指定すれば設定ファイルなしでも実行できます）"
        )

    if args.title is None:
        args.title = str(config_values.get("title", DEFAULT_TITLE))
    if args.memory_dir is None:
        args.memory_dir = Path(
            str(config_values.get("memory_dir", DEFAULT_MEMORY_DIR))
        )
    if not args.memory_dir.is_absolute():
        args.memory_dir = (args.config.parent / args.memory_dir).resolve()
    return args


def _describe(label: str, memory: dict[str, Any] | None) -> str:
    if memory is None:
        return f"{label}: （なし）"
    updated_at = memory.get("updated_at", "不明")
    session_count = memory.get("session_count", "不明")
    summary = str(memory.get("last_session_summary", "")).replace("\n", " ")
    if len(summary) > 60:
        summary = summary[:60] + "…"
    return (
        f"{label}: {session_count}回目 / 更新 {updated_at}\n"
        f"  直近の実況: {summary or '（記録なし）'}"
    )


def _preview(paths: OverallMemoryPaths) -> tuple[
    dict[str, Any] | None, dict[str, Any] | None
]:
    current = read_json_object(paths.overall)
    previous = read_json_object(paths.previous)
    return current, previous


def _confirm() -> bool:
    try:
        answer = input("入れ替えを実行しますか？ [y/N]: ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    try:
        args = _resolve_args(argv)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    paths = overall_memory_paths(args.memory_dir, args.title)
    print(f"対象タイトル: {args.title}")
    print(f"記憶フォルダ: {paths.directory}")

    try:
        current, previous = _preview(paths)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if previous is None:
        print(
            "エラー: 1回前の全体記憶がありません。"
            f"戻せる記憶が保存されていません: {paths.previous}",
            file=sys.stderr,
        )
        return 1

    print(_describe("現在の全体記憶", current))
    print(_describe("戻す先の全体記憶", previous))

    if args.dry_run:
        print("dry-run のため書き換えていません。")
        return 0

    if not args.yes and not _confirm():
        print("中止しました。", file=sys.stderr)
        return 1

    try:
        result = rollback_overall_memory(args.memory_dir, args.title)
    except RollbackError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    print(f"1回前の全体記憶へ戻しました: {result.paths.overall.resolve()}")
    if result.replaced is not None:
        print(
            "入れ替え前の記憶は次の場所に残っています"
            f"（もう一度実行すると元に戻せます）: {result.paths.previous.resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
