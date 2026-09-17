"""Adapter for the external unified DINOv3 flag/logo classifier."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any, Dict


class DinoV3Predictor:
    """Load the external Predictor once and expose in-memory PIL inference."""

    def __init__(self, prediction_root: Path, device: str):
        self.prediction_root = Path(prediction_root).expanduser().resolve()
        predict_file = self.prediction_root / "predict.py"
        if not predict_file.is_file():
            raise FileNotFoundError(f"DINOv3 predict.py not found: {predict_file}")

        root_text = str(self.prediction_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        spec = importlib.util.spec_from_file_location(
            "external_dinov3_prediction", predict_file
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import DINOv3 predictor from {predict_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._predictor = module.Predictor(device=device)
        self._lock = threading.Lock()

    def predict_image(self, image, top_k: int = 5) -> Dict[str, Any]:
        """Classify one PIL image across all configured feature banks."""
        image = image.convert("RGB")
        with self._lock:
            feature = self._predictor.embedder.encode_images([image])[0]
            ranked = self._predictor.index.predict(
                feature,
                top_k=top_k,
                category=None,
            )

        if not ranked:
            raise RuntimeError("DINOv3 classifier returned no candidates")

        top_k_results = [dict(item) for item in ranked]
        result = dict(top_k_results[0])
        result["confidence_type"] = "normalized_cosine_not_calibrated"
        result["margin"] = (
            float(top_k_results[0]["confidence"])
            - float(top_k_results[1]["confidence"])
            if len(top_k_results) > 1
            else None
        )
        result["top_k"] = top_k_results
        return result
