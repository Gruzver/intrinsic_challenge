"""
Port detector using contour analysis on grayscale images.

Approach:
  1. Enhance contrast (CLAHE) — compensates for variable illumination.
  2. Threshold dark regions — port openings are darker than their surroundings
     (dark rectangular hole in the NIC card / SC housing).
  3. Find contours and filter by:
       - Aspect ratio matching the known port shape.
       - Rectangularity (contour area vs bounding box area).
       - Proximity to image center (robot starts near the port).
  4. Return the highest-scoring candidate.

Port physical dimensions (from SDF / standard specs):
  SFP cage opening : ~13.5 mm wide × ~8.8 mm tall  → aspect ≈ 1.53
  SC  port housing : ~25.8 mm wide × ~10.8 mm tall  → aspect ≈ 2.39
  (aspect = width / height as seen from the front face)
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ── Port profiles ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortProfile:
    real_width_m: float    # physical width of port opening [m] — used for monocular depth
    real_height_m: float   # physical height [m]
    aspect_min: float      # accepted width/height range (accounts for ±10° yaw)
    aspect_max: float

PORT_PROFILES: dict[str, PortProfile] = {
    "sfp": PortProfile(
        real_width_m=0.0135,
        real_height_m=0.0088,
        aspect_min=0.8,   # 1.53 ± view-angle tolerance
        aspect_max=2.5,
    ),
    "sc": PortProfile(
        real_width_m=0.0258,
        real_height_m=0.0108,
        aspect_min=1.5,   # 2.39 ± view-angle tolerance
        aspect_max=4.0,
    ),
}


# ── Detection result ───────────────────────────────────────────────────────────

@dataclass
class Detection:
    u: int          # pixel x of port center
    v: int          # pixel y of port center
    w: int          # bounding box width  [px]
    h: int          # bounding box height [px]
    score: float    # 0–1 confidence score


# ── Detector ──────────────────────────────────────────────────────────────────

class PortDetector:
    """Detects the target port in a single camera image using contour analysis."""

    def __init__(self, plug_type: str):
        """plug_type: 'sfp' or 'sc'."""
        if plug_type not in PORT_PROFILES:
            raise ValueError(f"Unknown plug_type '{plug_type}'. Expected: {list(PORT_PROFILES)}")
        self.plug_type = plug_type
        self.profile = PORT_PROFILES[plug_type]

        # Minimum area in pixels — rejects tiny noise contours.
        # At ~15 cm distance, SFP port is ~90×58 px (rough estimate with fx≈1000).
        # We allow much smaller to handle larger distances.
        self._min_area_px = 200
        self._min_score = 0.20   # candidates below this are discarded

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, image_rgb: np.ndarray) -> Optional[Detection]:
        """Detect the port in image_rgb. Returns best Detection or None.

        image_rgb: H×W×3 uint8, RGB order (as decoded from ROS Image msg).
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h_img, w_img = gray.shape

        # 1. Enhance local contrast so port edges are clear regardless of lighting.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Detect dark regions (port openings are dark holes).
        #    Adaptive threshold binarises the image so that pixels darker than
        #    their local neighbourhood become white in the mask.
        dark_mask = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,   # neighbourhood size in px
            C=8,            # threshold offset
        )

        # 3. Morphological cleanup — close small gaps, remove pepper noise.
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k, iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,  k, iterations=1)

        # 4. Find external contours of dark blobs.
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Score each contour.
        best: Optional[Detection] = None
        img_cx, img_cy = w_img / 2.0, h_img / 2.0
        img_diag = np.sqrt(w_img ** 2 + h_img ** 2)

        for cnt in contours:
            det = self._score_contour(cnt, img_cx, img_cy, img_diag)
            if det is None:
                continue
            if best is None or det.score > best.score:
                best = det

        if best is None or best.score < self._min_score:
            return None
        return best

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _score_contour(
        self,
        cnt: np.ndarray,
        img_cx: float,
        img_cy: float,
        img_diag: float,
    ) -> Optional[Detection]:
        area = cv2.contourArea(cnt)
        if area < self._min_area_px:
            return None

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0:
            return None

        # ── Aspect ratio filter ────────────────────────────────────────────────
        aspect = w / h
        if not (self.profile.aspect_min <= aspect <= self.profile.aspect_max):
            return None

        # ── Rectangularity: contour area vs bbox area ──────────────────────────
        # A perfect rectangle = 1.0; irregular blobs are much lower.
        rect_ratio = area / (w * h)
        if rect_ratio < 0.40:
            return None

        # ── Proximity to image centre ──────────────────────────────────────────
        # Robot starts near the port, so it should be roughly centred.
        cx = x + w / 2.0
        cy = y + h / 2.0
        norm_dist = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2) / (img_diag / 2.0)
        center_score = max(0.0, 1.0 - norm_dist)

        # ── Aspect ratio closeness to ideal ────────────────────────────────────
        ideal_aspect = self.profile.real_width_m / self.profile.real_height_m
        aspect_score = 1.0 / (1.0 + abs(aspect - ideal_aspect))

        score = 0.5 * rect_ratio + 0.3 * center_score + 0.2 * aspect_score

        return Detection(
            u=int(round(cx)),
            v=int(round(cy)),
            w=w,
            h=h,
            score=score,
        )
