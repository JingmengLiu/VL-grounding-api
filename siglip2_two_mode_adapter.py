"""Adapter for the packaged SigLIP2 flag/logo retrieval model.

The upstream package exposes one ``Predictor`` per mode, and each Predictor
loads its own copy of SigLIP2.  The grounding service needs both modes, so this
adapter shares one image encoder and only switches the retrieval bank.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, SiglipProcessor


class Siglip2TwoModeRecognizer:
    """Shared SigLIP2 encoder with independently selectable flag/logo banks."""

    VALID_MODES = ("flag", "logo")

    def __init__(
        self,
        root: str | Path,
        device: str | torch.device = "auto",
        enabled_modes: Iterable[str] = VALID_MODES,
        fp16: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.device = self._choose_device(device)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self._inference_lock = Lock()

        model_path = self.root / "model"
        if not model_path.is_dir():
            raise FileNotFoundError(f"SigLIP2 model directory not found: {model_path}")

        modes = tuple(dict.fromkeys(str(mode).lower() for mode in enabled_modes))
        invalid_modes = set(modes) - set(self.VALID_MODES)
        if invalid_modes:
            raise ValueError(f"Unsupported SigLIP2 modes: {sorted(invalid_modes)}")
        if not modes:
            raise ValueError("At least one SigLIP2 mode must be enabled")

        self.processor = SiglipProcessor.from_pretrained(str(model_path))
        self.model = AutoModel.from_pretrained(str(model_path)).eval().to(self.device)
        self.banks = {mode: self._load_bank(mode) for mode in modes}

    @staticmethod
    def _choose_device(value: str | torch.device) -> torch.device:
        requested = str(value or "auto").lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({requested}) but is unavailable")
        return torch.device(requested)

    def _load_bank(self, mode: str) -> dict:
        bank_path = self.root / "banks" / f"{mode}.pt"
        if not bank_path.is_file():
            raise FileNotFoundError(f"SigLIP2 {mode} bank not found: {bank_path}")
        try:
            bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        except TypeError:
            bank = torch.load(bank_path, map_location="cpu")

        classes = bank.get("classes")
        vectors = bank.get("vectors")
        if not isinstance(classes, list) or not isinstance(vectors, torch.Tensor):
            raise ValueError(f"Invalid SigLIP2 bank format: {bank_path}")
        if vectors.ndim != 2 or len(classes) != vectors.shape[0]:
            raise ValueError(f"Mismatched classes/vectors in SigLIP2 bank: {bank_path}")

        expected_dim = int(self.model.config.vision_config.hidden_size)
        if vectors.shape[1] != expected_dim:
            raise ValueError(
                f"SigLIP2 bank dimension {vectors.shape[1]} does not match model {expected_dim}: {bank_path}"
            )
        return {
            "classes": classes,
            "vectors": F.normalize(vectors.float(), dim=-1),
        }

    @torch.inference_mode()
    def predict(self, image: Image.Image, mode: str) -> dict:
        """Return the top-1 class and its raw cosine similarity."""
        mode = str(mode).lower()
        if mode not in self.banks:
            raise ValueError(f"SigLIP2 mode is not loaded: {mode}")

        rgb_image = image.convert("RGB")
        inputs = self.processor(images=[rgb_image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)

        # logo and flag tasks run in a ThreadPoolExecutor. Serialize access to
        # the shared GPU encoder to avoid concurrent forward passes on it.
        with self._inference_lock:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.fp16,
            ):
                features = self.model.get_image_features(pixel_values=pixel_values)

        if not isinstance(features, torch.Tensor):
            features = getattr(features, "pooler_output", None)
            if features is None:
                features = getattr(features, "last_hidden_state", None)
                if features is not None and features.ndim == 3:
                    features = features[:, 0]
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise RuntimeError("SigLIP2 image features are not a 2D tensor")

        query = F.normalize(features[0].float(), dim=0).cpu()
        bank = self.banks[mode]
        similarities = bank["vectors"] @ query
        similarity, position = similarities.max(dim=0)
        class_info = bank["classes"][int(position.item())]

        result = {
            "id": class_info["id"],
            "class_key": class_info["class_key"],
            "source_category": class_info["source_category"],
            "class_name": class_info["class_name"],
            "category": class_info["category"],
            "similarity": float(similarity.item()),
            "score_type": "cosine_similarity",
        }
        if class_info.get("qid"):
            result["qid"] = class_info["qid"]
        return result
