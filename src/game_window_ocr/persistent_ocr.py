from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image


class PersistentNdlOcr:
    """Load NDLOCR-Lite's ONNX models once and reuse them for every frame."""

    def __init__(self, *, device: str = "cpu") -> None:
        # NDLOCR-Lite exposes its CLI implementation as the top-level `ocr`
        # module. The dependency is revision-pinned in pyproject.toml.
        import ocr as ndlocr

        base_dir = Path(ndlocr.__file__).resolve().parent
        args = SimpleNamespace(
            det_weights=str(base_dir / "model" / "deim-s-1024x1024.onnx"),
            det_classes=str(base_dir / "config" / "ndl.yaml"),
            det_score_threshold=0.2,
            det_conf_threshold=0.25,
            det_iou_threshold=0.2,
            rec_weights30=str(
                base_dir
                / "model"
                / "parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx"
            ),
            rec_weights50=str(
                base_dir
                / "model"
                / "parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx"
            ),
            rec_weights=str(
                base_dir
                / "model"
                / "parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx"
            ),
            rec_classes=str(base_dir / "config" / "NDLmoji.yaml"),
            device=device,
            enable_tcy=False,
        )

        started = time.perf_counter()
        self._ndlocr = ndlocr
        self._detector = ndlocr.get_detector(args)
        self._recognizer100 = ndlocr.get_recognizer(args=args)
        self._recognizer30 = ndlocr.get_recognizer(
            args=args,
            weights_path=args.rec_weights30,
        )
        self._recognizer50 = ndlocr.get_recognizer(
            args=args,
            weights_path=args.rec_weights50,
        )
        self.initialization_seconds = time.perf_counter() - started

    def recognize(
        self,
        image: Image.Image,
        *,
        input_path: Path,
        output_dir: Path,
        viz: bool,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result: dict[str, Any] = self._ndlocr._run_ocr_on_image_array(
            detector=self._detector,
            recognizer30=self._recognizer30,
            recognizer50=self._recognizer50,
            recognizer100=self._recognizer100,
            inputname=input_path.name,
            img=np.asarray(image.convert("RGB")),
            outputpath=str(output_dir.resolve()),
            save_viz=viz,
        )

        stem = input_path.stem
        text_path = output_dir / f"{stem}.txt"
        text_path.write_text(result["text"], encoding="utf-8")
        (output_dir / f"{stem}.xml").write_text(
            "<OCRDATASET>\n" + result["page_xml"] + "\n</OCRDATASET>\n",
            encoding="utf-8",
        )
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(
                {
                    "contents": [result["json_lines"]],
                    "imginfo": {
                        "img_width": result["img_width"],
                        "img_height": result["img_height"],
                        "img_path": str(input_path.resolve()),
                        "img_name": input_path.name,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        elapsed = time.perf_counter() - started
        print(f"NDLOCR認識: {elapsed:.3f}秒（モデル再利用）")
        return json_path
