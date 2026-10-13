"""YOLO26 defect detector with optional SAHI-style sliced inference.

- Model is loaded ONCE at startup (get_model caches; warm_model preloads).
- detect(image_path, conf, use_sahi) returns (detections, method, seconds).
- SAHI mode: split image into 4 slices (2x2 grid) with small overlap,
  detect on each slice, map boxes back to full image, merge with NMS.
- Normal mode: single YOLO predict call.
"""

import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.config import (
    CLASS_COLORS,
    CONF_THRESHOLD,
    DEFECT_CLASSES,
    DEVICE,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    LEGACY_MODEL_PATH,
    MODEL_PATH,
    SAHI_OVERLAP_RATIO,
    SAHI_SLICE_COLS,
    SAHI_SLICE_ROWS,
)

Detection = dict[str, Any]

_model: Optional[Any] = None  # cached YOLO instance


def category_mapping() -> dict[str, str]:
    """SAHI category mapping: str id -> class name (fixed order)."""
    return {str(i): name for i, name in enumerate(DEFECT_CLASSES)}


def resolve_weights() -> Path:
    """Resolve weights path (models/best.pt, fallback legacy file).

    Raises FileNotFoundError if neither exists.
    """
    if MODEL_PATH.exists():
        return MODEL_PATH
    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    raise FileNotFoundError(
        f"No model weights found at {MODEL_PATH}. "
        "Place trained YOLO26 weights at models/best.pt "
        "(see README for download steps)."
    )


def get_model() -> Any:
    """Load (and cache) the YOLO model. Called once at startup."""
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO  # lazy: app boots even without torch

    _model = YOLO(str(resolve_weights()))
    return _model


def warm_model() -> None:
    """Preload model at startup; tolerate missing weights (lazy error)."""
    try:
        get_model()
    except Exception:
        _clear_model()


def _clear_model() -> None:
    global _model
    _model = None


def set_conf_threshold(value: float) -> float:
    """Clamp a user confidence value into a sane range."""
    try:
        v = float(value)
    except Exception:
        return CONF_THRESHOLD
    return max(0.05, min(0.95, v))


def is_large_image(width: int, height: int, threshold: int = 1280) -> bool:
    """Compat helper: True if max side exceeds threshold (SAHI candidate)."""
    return max(int(width), int(height)) > int(threshold)


# ---------------- Box helpers ----------------

def _boxes_to_detections(xyxy: Any, confs: Any,
                         clss: Any) -> list[Detection]:
    """Convert raw Nx4 boxes + conf + cls arrays to detection dicts."""
    out: list[Detection] = []
    for box, conf, cls in zip(xyxy, confs, clss):
        cid = int(cls)
        name = DEFECT_CLASSES[cid] if 0 <= cid < len(DEFECT_CLASSES) else str(cid)
        out.append({
            "x1": float(box[0]), "y1": float(box[1]),
            "x2": float(box[2]), "y2": float(box[3]),
            "class_id": cid, "class_name": name,
            "confidence": float(conf),
        })
    return out


def _iou(a: Detection, b: Detection) -> float:
    """IoU of two boxes (same class only should be compared)."""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_merge(detections: list[Detection],
              iou_thresh: float = IOU_THRESHOLD) -> list[Detection]:
    """Class-wise greedy NMS merge (used to fuse SAHI slice results)."""
    merged: list[Detection] = []
    # Group by class so different defects never suppress each other.
    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        by_class.setdefault(str(d.get("class_name")), []).append(d)
    for group in by_class.values():
        group = sorted(group, key=lambda d: float(d.get("confidence", 0)),
                       reverse=True)
        kept: list[Detection] = []
        for cand in group:
            if all(_iou(cand, k) < iou_thresh for k in kept):
                kept.append(cand)
        merged.extend(kept)
    # Highest confidence first for stable tables/reports.
    merged.sort(key=lambda d: float(d.get("confidence", 0)), reverse=True)
    return merged


# ---------------- Inference paths ----------------

def run_inference(image_path: Path,
                  conf: float = CONF_THRESHOLD) -> list[Detection]:
    """Single full-image YOLO inference."""
    model = get_model()
    results = model.predict(
        source=str(image_path),
        conf=float(conf),
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )
    detections: list[Detection] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        data = r.boxes.data.cpu().numpy()  # x1,y1,x2,y2,conf,cls
        detections.extend(
            _boxes_to_detections(data[:, :4], data[:, 4], data[:, 5]))
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def run_sliced_inference(image_path: Path,
                         conf: float = CONF_THRESHOLD) -> list[Detection]:
    """SAHI-mode: 4 slices (2x2) with overlap, detect, remap, NMS-merge.

    First tries the `sahi` library path; falls back to manual slicing
    (crop -> YOLO predict -> offset boxes) so high-res mode always works.
    """
    # Try native SAHI library for best merging behaviour.
    try:
        return _run_sahi_library(image_path, conf)
    except Exception:
        pass  # fall through to manual slicing
    return _run_manual_slices(image_path, conf)


def _run_sahi_library(image_path: Path,
                      conf: float) -> list[Detection]:
    """Sliced prediction via sahi library (GREEDYNMM merge)."""
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from sahi.predict import get_sliced_prediction

    from app.config import (
        SAHI_OVERLAP_HEIGHT_RATIO,
        SAHI_OVERLAP_WIDTH_RATIO,
        SAHI_POSTPROCESS_MATCH_METRIC,
        SAHI_POSTPROCESS_MATCH_THRESHOLD,
        SAHI_POSTPROCESS_TYPE,
    )

    weights = resolve_weights()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image for sliced inference.")
    h, w = img.shape[:2]
    slice_h = max(h // SAHI_SLICE_ROWS, 256)
    slice_w = max(w // SAHI_SLICE_COLS, 256)
    det_model = UltralyticsDetectionModel(
        model_path=str(weights),
        device=DEVICE,
        confidence_threshold=float(conf),
        category_mapping=category_mapping(),
        image_size=IMAGE_SIZE,
    )
    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=det_model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=SAHI_OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=SAHI_OVERLAP_WIDTH_RATIO,
        postprocess_type=SAHI_POSTPROCESS_TYPE,
        postprocess_match_metric=SAHI_POSTPROCESS_MATCH_METRIC,
        postprocess_match_threshold=SAHI_POSTPROCESS_MATCH_THRESHOLD,
        verbose=0,
    )
    detections: list[Detection] = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        name = str(pred.category.name)
        cid = (DEFECT_CLASSES.index(name) if name in DEFECT_CLASSES
               else int(pred.category.id))
        detections.append({
            "x1": float(bbox.minx), "y1": float(bbox.miny),
            "x2": float(bbox.maxx), "y2": float(bbox.maxy),
            "class_id": cid, "class_name": name,
            "confidence": float(pred.score.value),
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def _run_manual_slices(image_path: Path,
                       conf: float) -> list[Detection]:
    """Manual 2x2 overlapping slices (no sahi dependency at runtime)."""
    import tempfile

    model = get_model()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image for sliced inference.")
    h, w = img.shape[:2]
    rows, cols = SAHI_SLICE_ROWS, SAHI_SLICE_COLS
    ov = SAHI_OVERLAP_RATIO
    # Overlap in pixels so defects on cut lines appear in both slices.
    ov_h = int(h * ov)
    ov_w = int(w * ov)
    base_h, base_w = h // rows, w // cols

    all_dets: list[Detection] = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in range(rows):
            for c in range(cols):
                y1 = max(0, r * base_h - (ov_h if r > 0 else 0))
                y2 = min(h, (r + 1) * base_h + (ov_h if r < rows - 1 else 0))
                x1 = max(0, c * base_w - (ov_w if c > 0 else 0))
                x2 = min(w, (c + 1) * base_w + (ov_w if c < cols - 1 else 0))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_path = str(Path(tmp) / f"slice_{r}_{c}.jpg")
                cv2.imwrite(crop_path, crop)
                results = model.predict(
                    source=crop_path, conf=float(conf), iou=IOU_THRESHOLD,
                    imgsz=IMAGE_SIZE, device=DEVICE, verbose=False)
                for res in results:
                    if res.boxes is None or len(res.boxes) == 0:
                        continue
                    data = res.boxes.data.cpu().numpy()
                    for box, cf, cl in zip(data[:, :4], data[:, 4],
                                           data[:, 5]):
                        # Map slice coords back to full-image coords.
                        cid = int(cl)
                        name = (DEFECT_CLASSES[cid]
                                if 0 <= cid < len(DEFECT_CLASSES) else str(cid))
                        all_dets.append({
                            "x1": float(box[0] + x1),
                            "y1": float(box[1] + y1),
                            "x2": float(box[2] + x1),
                            "y2": float(box[3] + y1),
                            "class_id": cid, "class_name": name,
                            "confidence": float(cf),
                        })
    return nms_merge(all_dets)


# ---------------- Public API ----------------

def detect(image_path: Path, conf: float = CONF_THRESHOLD,
           use_sahi: bool = False) -> tuple[list[Detection], str, float]:
    """Detect defects.

    Args:
        image_path: image file.
        conf: confidence threshold (from slider).
        use_sahi: True -> 4-slice high-res mode.

    Returns:
        (detections, method, seconds) with method "sahi" or "yolo".
    """
    conf = set_conf_threshold(conf)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image for detection.")
    t0 = time.perf_counter()
    if use_sahi:
        dets = run_sliced_inference(image_path, conf)
        method = "sahi"
    else:
        dets = run_inference(image_path, conf)
        method = "yolo"
    dt = time.perf_counter() - t0
    return dets, method, dt


def decide_pass_fail(detections: list[Detection]) -> str:
    """PASS if no defects, else FAIL."""
    return "FAIL" if len(detections) > 0 else "PASS"


def draw_boxes(image_path: Path, detections: list[Detection],
               output_path: Path) -> Path:
    """Draw coloured boxes + label + confidence, save annotated image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image for annotation.")
    for d in detections:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        color = CLASS_COLORS.get(str(d["class_name"]), (0, 255, 0))
        label = f"{d['class_name']} {float(d['confidence']):.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return output_path


def count_per_class(detections: list[Detection]) -> dict[str, int]:
    """Count detections per defect class (all 6 keys always present)."""
    counts = {c: 0 for c in DEFECT_CLASSES}
    for d in detections:
        name = str(d.get("class_name", ""))
        if name in counts:
            counts[name] += 1
    return counts
