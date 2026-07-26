from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

from PIL import Image

from .windows import (
    capture_client,
    enable_dpi_awareness,
    find_window,
    list_windows,
)


DEFAULT_TITLE = "かまいたちの夜"


def _is_halfwidth_character(character: str) -> bool:
    return unicodedata.east_asian_width(character) in {"H", "Na"}


def parse_crop(value: str) -> tuple[int, int, int, int]:
    try:
        crop = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--crop は left,top,right,bottom の4整数で指定してください。"
        ) from exc
    if len(crop) != 4:
        raise argparse.ArgumentTypeError(
            "--crop は left,top,right,bottom の4整数で指定してください。"
        )
    left, top, right, bottom = crop
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("--crop の座標またはサイズが不正です。")
    return left, top, right, bottom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Windowsのゲーム画面をキャプチャし、NDLOCR-Liteで本文を抽出します。"
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help=f"対象ウィンドウのタイトル（部分一致、既定: {DEFAULT_TITLE!r}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="保存先（省略時: output/YYYYMMDD_HHMMSS）",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="OCR前の拡大率（既定: 2.0）",
    )
    parser.add_argument(
        "--crop",
        type=parse_crop,
        help="クライアント領域内の切り抜き: left,top,right,bottom",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="キャプチャだけを保存し、OCRを実行しません。",
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="キャプチャ前に対象ウィンドウを前面化しません。",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="NDLOCR-Liteの認識枠画像も保存します。",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="本文に採用する検出信頼度の下限（0.0～1.0、既定: 0.5）",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="キャプチャ可能なウィンドウ一覧を表示して終了します。",
    )
    return parser


def _validate_crop(
    image: Image.Image, crop: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    left, top, right, bottom = crop
    if right > image.width or bottom > image.height:
        raise ValueError(
            f"--crop {crop} は画像サイズ {image.width}x{image.height} の外側です。"
        )
    return crop


def prepare_ocr_image(
    image: Image.Image,
    *,
    crop: tuple[int, int, int, int] | None,
    scale: float,
) -> Image.Image:
    if scale <= 0:
        raise ValueError("--scale は0より大きい値にしてください。")
    prepared = image.crop(_validate_crop(image, crop)) if crop else image.copy()
    if scale != 1:
        size = (
            max(1, round(prepared.width * scale)),
            max(1, round(prepared.height * scale)),
        )
        prepared = prepared.resize(size, Image.Resampling.LANCZOS)
    return prepared


def _ndlocr_command(image_path: Path, output_dir: Path, viz: bool) -> list[str]:
    script_name = "ndlocr-lite.exe" if sys.platform == "win32" else "ndlocr-lite"
    executable = Path(sys.executable).with_name(script_name)
    resolved = executable if executable.exists() else shutil.which("ndlocr-lite")
    if not resolved:
        raise FileNotFoundError(
            "NDLOCR-Liteが見つかりません。先に `uv sync` を実行してください。"
        )
    command = [
        str(resolved),
        "--sourceimg",
        str(image_path.resolve()),
        "--output",
        str(output_dir.resolve()),
    ]
    if viz:
        command.extend(["--viz", "True"])
    return command


def run_ocr(image_path: Path, output_dir: Path, *, viz: bool) -> Path:
    command = _ndlocr_command(image_path, output_dir, viz)
    print("NDLOCR-Liteで文字認識中です（初回は少し時間がかかります）...")
    subprocess.run(command, check=True)
    text_path = output_dir / f"{image_path.stem}.txt"
    if not text_path.exists():
        raise RuntimeError("OCRは終了しましたが、本文テキストが生成されませんでした。")
    return text_path


def clean_ocr_text(json_path: Path, *, min_confidence: float) -> str:
    if not 0 <= min_confidence <= 1:
        raise ValueError("--min-confidence は0.0～1.0で指定してください。")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    contents = payload.get("contents", [])
    lines: list[str] = []
    for page in contents:
        for item in page:
            text = str(item.get("text", "")).strip()
            confidence = float(item.get("confidence", 0))
            if text and item.get("isTextline") != "false" and confidence >= min_confidence:
                lines.append(text)
    if lines:
        # The game's page-advance book icon is sometimes recognized as one or two
        # trailing characters, either on its own line or after sentence punctuation.
        if (
            len(lines) >= 2
            and len(lines[-1]) <= 2
            and re.search(r"[。！？!?」』）)]$", lines[-2])
        ):
            lines.pop()
        elif re.search(r"[。！？!?」』）)][^。！？!?」』）)]{1,2}$", lines[-1]):
            lines[-1] = re.sub(
                r"([。！？!?」』）)一-龯ぁ-んァ-ヶー])"
                r"([。！？!?」』）)])[^。！？!?」』）)]{1,2}$",
                r"\1\2",
                lines[-1],
            )
    text = "\n".join(lines)
    if len(text) >= 2 and all(
        _is_halfwidth_character(character) for character in text[-2:]
    ):
        text = text[:-2].rstrip("\n")
    return text


def _print_windows() -> None:
    windows = list_windows()
    if not windows:
        print("キャプチャ可能なウィンドウはありません。")
        return
    for window in windows:
        print(f"{window.width:>5}x{window.height:<5}  {window.title}")


def main(argv: list[str] | None = None) -> int:
    enable_dpi_awareness()
    args = build_parser().parse_args(argv)

    if args.list_windows:
        _print_windows()
        return 0

    try:
        window = find_window(args.title)
        print(
            f"対象: {window.title} "
            f"({window.width}x{window.height}, HWND=0x{window.hwnd:X})"
        )
        output_dir = args.output or Path("output") / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_image = capture_client(
            window,
            activate=not args.no_activate,
        )
        raw_path = output_dir / "capture_raw.png"
        raw_image.save(raw_path)

        prepared = prepare_ocr_image(
            raw_image,
            crop=args.crop,
            scale=args.scale,
        )
        input_path = output_dir / "capture.png"
        prepared.save(input_path)
        print(f"キャプチャ保存: {raw_path.resolve()}")
        print(f"OCR入力画像: {input_path.resolve()} ({prepared.width}x{prepared.height})")

        if args.capture_only:
            return 0

        raw_text_path = run_ocr(input_path, output_dir, viz=args.viz)
        json_path = output_dir / f"{input_path.stem}.json"
        text = clean_ocr_text(
            json_path,
            min_confidence=args.min_confidence,
        )
        text_path = output_dir / "text.txt"
        text_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        print(f"本文保存: {text_path.resolve()}")
        print(f"NDLOCR-Lite生出力: {raw_text_path.resolve()}")
        print("\n--- 抽出本文 ---")
        print(text if text else "（文字を検出できませんでした）")
        return 0
    except (LookupError, OSError, RuntimeError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(
            f"エラー: NDLOCR-Liteが終了コード {exc.returncode} で失敗しました。",
            file=sys.stderr,
        )
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
