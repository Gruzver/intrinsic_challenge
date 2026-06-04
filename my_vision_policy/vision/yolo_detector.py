"""
YOLO-Pose port detector.

Loads a trained yolov8-pose model (best.pt) and returns the highest-confidence
detection of the requested port class, using the same Detection dataclass as
the original contour-based detector.

Classes trained:
  0  sfp_port  (13.5 × 8.8 mm)
  1  sc_port   (25.8 × 10.8 mm)

Keypoints (8, stored in det.keypoints after detection):
  KP0-3  entrance face  TL TR BR BL  ← use for solvePnP
  KP4-7  back face      TL TR BR BL  ← visible at oblique angles
"""

from pathlib import Path
from typing import Optional

import numpy as np

from my_vision_policy.vision.detector import Detection

_CLASS_ID = {"sfp": 0, "sc": 1}

_DEFAULT_MODEL = Path(__file__).parent.parent / "weights" / "best.pt"


class YoloPoseDetector:
    """Detects a target port using a trained YOLOv8-pose model."""

    def __init__(self, plug_type: str, model_path: Optional[str] = None):
        """
        plug_type : 'sfp' or 'sc'
        model_path: path to .pt weights; defaults to bundled best.pt
        """
        if plug_type not in _CLASS_ID:
            raise ValueError(f"Unknown plug_type '{plug_type}'. Expected: {list(_CLASS_ID)}")
        self._class_id = _CLASS_ID[plug_type]

        path = Path(model_path) if model_path else _DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {path}")

        from ultralytics import YOLO
        self._model = YOLO(str(path))
        self._model.fuse()  # fuse Conv+BN for faster inference

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """Run inference on image_rgb (H×W×3, uint8, RGB).

        Returns ALL target-class detections sorted by confidence descending.
        Each Detection carries .keypoints (8×2) and .keypoints_vis (8,) for solvePnP.
        Returns empty list if none found.
        """
        results = self._model(image_rgb[..., ::-1], verbose=False)[0]  # RGB→BGR

        boxes = results.boxes
        kpts  = results.keypoints

        if boxes is None or len(boxes) == 0:
            _inv = {v: k for k, v in _CLASS_ID.items()}
            print(f"[YOLO] target={_inv.get(self._class_id,'?')} — no boxes detected", flush=True)
            return []

        # Log when wrong-class boxes appear or target not detected
        _CLASS_NAME = {v: k for k, v in _CLASS_ID.items()}
        target_name = _CLASS_NAME.get(self._class_id, "?")
        all_preds = [
            f"{_CLASS_NAME.get(int(b.cls[0].item()), '?')}:{float(b.conf[0].item()):.2f}"
            for b in boxes
        ]
        has_wrong_class = any(int(b.cls[0].item()) != self._class_id for b in boxes)
        has_target = any(int(b.cls[0].item()) == self._class_id for b in boxes)
        if has_wrong_class or not has_target:
            print(f"[YOLO] target={target_name} | all_boxes=[{', '.join(all_preds)}]", flush=True)

        detections: list[Detection] = []
        for i, box in enumerate(boxes):
            if int(box.cls[0].item()) != self._class_id:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det = Detection(
                u=int(round((x1 + x2) / 2.0)),
                v=int(round((y1 + y2) / 2.0)),
                w=int(round(x2 - x1)),
                h=int(round(y2 - y1)),
                score=float(box.conf[0].item()),
            )
            if kpts is not None and i < len(kpts):
                det.keypoints     = kpts[i].xy[0].cpu().numpy()   # (8, 2)
                det.keypoints_vis = kpts[i].conf[0].cpu().numpy()  # (8,)
            else:
                det.keypoints     = None
                det.keypoints_vis = None
            detections.append(det)

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def set_plug_type(self, plug_type: str) -> None:
        """Switch class filter (sfp/sc) without reloading the model."""
        if plug_type not in _CLASS_ID:
            raise ValueError(f"Unknown plug_type '{plug_type}'. Expected: {list(_CLASS_ID)}")
        self._class_id = _CLASS_ID[plug_type]
