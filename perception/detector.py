"""
NexusPilot: YOLO-based Object Detector
Optimized for PyTorch, ONNX, and OpenVINO with robust metadata handling.

When use_onnx=True, uses onnxruntime directly (bypasses ultralytics wrapper)
to eliminate memory leaks and reduce CPU/RAM overhead on Raspberry Pi.
"""
import numpy as np
import time
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# Try to import onnxruntime for direct inference
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


@dataclass
class DetectedObject:
    """Represents a single detected object in the scene."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    category: str  # "vehicle", "pedestrian", "cyclist", "traffic_sign", "other"
    track_id: int = -1
    distance: float = -1.0
    world_position: Optional[Tuple[float, float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_stale: bool = False

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2,
                (self.bbox[1] + self.bbox[3]) // 2)

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class SimpleIOUTracker:
    """
    Lightweight tracker to maintain object identity across frames.
    """
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections: List[DetectedObject], dt: float) -> List[DetectedObject]:
        matched_indices = set()

        for det in detections:
            best_iou = -1
            best_id = -1
            for tid, track in self.tracks.items():
                iou = self._calculate_iou(det.bbox, track['last_bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_id = tid

            if best_id != -1:
                det.track_id = best_id
                matched_indices.add(best_id)
                self.tracks[best_id].update({
                    'last_bbox': det.bbox,
                    'age': 0,
                })
            else:
                det.track_id = self.next_id
                self.tracks[self.next_id] = {'last_bbox': det.bbox, 'age': 0}
                self.next_id += 1

        to_remove = []
        for tid in self.tracks:
            if tid not in matched_indices:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age: to_remove.append(tid)
        for tid in to_remove: del self.tracks[tid]

        # Prevent unbounded track growth
        if len(self.tracks) > 50:
            oldest = sorted(self.tracks.items(), key=lambda x: x[1]['age'], reverse=True)
            for tid, _ in oldest[len(self.tracks) - 50:]:
                del self.tracks[tid]

        return detections


# ---- COCO80 class names (fallback) ----
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


class ONNXRuntimeDetector:
    """
    Direct ONNX Runtime YOLO detector — bypasses ultralytics wrapper entirely.
    Eliminates memory leaks and reduces RAM overhead on Raspberry Pi.
    """
    def __init__(self, config: dict):
        self.imgsz = config.get("imgsz", 320)
        self.conf_threshold = config.get("confidence_threshold", 0.3)
        self.nms_threshold = config.get("nms_threshold", 0.45)

        # Category mapping
        self._vehicle_ids = set(config.get("vehicle_classes", [2, 3, 5, 7]))
        self._pedestrian_ids = set(config.get("pedestrian_classes", [0]))
        self._cyclist_ids = set(config.get("cyclist_classes", [1]))
        self._class_names = COCO_NAMES

        # Load ONNX model
        model_path = config.get("onnx_model_path", "model/yolo11n.onnx")
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, model_path)

        print(f"[ONNXDetector] Loading: {model_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Use single thread to avoid CPU contention on RPi
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path, sess_opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"[ONNXDetector] Ready. Input={self.input_name}, Providers={self.session.get_providers()}")

        self.tracker = SimpleIOUTracker()
        self._inference_times = []
        self._last_time = time.perf_counter()

    def _classify_category(self, class_id: int) -> str:
        if class_id in self._vehicle_ids: return "vehicle"
        if class_id in self._pedestrian_ids: return "pedestrian"
        if class_id in self._cyclist_ids: return "cyclist"
        return "other"

    @staticmethod
    def _preprocess(image: np.ndarray, imgsz: int):
        """
        Letterbox resize + normalize + HWC→CHW.
        Returns: preprocessed (1,3,H,W), letterbox scale, padding (top, left).
        """
        orig_h, orig_w = image.shape[:2]
        ratio = min(imgsz / orig_h, imgsz / orig_w)
        new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)

        resized = np.zeros((imgsz, imgsz, 3), dtype=np.float32)
        resized[:new_h, :new_w] = (
            cv2_resize_safe(image, (new_w, new_h)).astype(np.float32) / 255.0
        )

        # HWC → CHW → batch
        img = np.ascontiguousarray(resized.transpose(2, 0, 1)[np.newaxis, ...])
        return img, ratio, 0, 0

    def _postprocess(self, output: np.ndarray, ratio: float,
                     pad_top: int, pad_left: int, orig_w: int, orig_h: int):
        """
        Parse YOLO output (1, 84, N), apply confidence filter + NMS,
        return list of DetectedObject.
        """
        # output shape: (1, 84, N) → transpose to (N, 84)
        pred = output[0].T  # (N, 84)

        # Box coords: cx, cy, w, h
        boxes = pred[:, :4]  # (N, 4)
        # Class scores: columns 4-83
        class_scores = pred[:, 4:]  # (N, 80)

        # Best class per anchor
        best_cls_ids = np.argmax(class_scores, axis=1)
        best_confs = class_scores[np.arange(len(best_cls_ids)), best_cls_ids]

        # Confidence filter
        mask = best_confs >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confs = best_confs[mask]
        cls_ids = best_cls_ids[mask]

        # Convert cx,cy,w,h → x1,y1,x2,y2 (center to corners)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        indices = nms_numpy(boxes_xyxy, confs, self.nms_threshold)
        if len(indices) == 0:
            return []

        detections = []
        for idx in indices:
            box = boxes_xyxy[idx]
            # Scale back to original image
            x1 = int((box[0] - pad_left) / ratio)
            y1 = int((box[1] - pad_top) / ratio)
            x2 = int((box[2] - pad_left) / ratio)
            y2 = int((box[3] - pad_top) / ratio)

            # Clamp to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            cls_id = int(cls_ids[idx])
            det = DetectedObject(
                class_id=cls_id,
                class_name=self._class_names.get(cls_id, f"class_{cls_id}"),
                confidence=float(confs[idx]),
                bbox=(x1, y1, x2, y2),
                category=self._classify_category(cls_id),
            )
            detections.append(det)

        return detections

    def detect(self, image: np.ndarray) -> List[DetectedObject]:
        current_time = time.perf_counter()
        dt = current_time - self._last_time
        self._last_time = current_time

        orig_h, orig_w = image.shape[:2]

        t0 = time.perf_counter()
        preprocessed, ratio, pad_top, pad_left = self._preprocess(image, self.imgsz)
        output = self.session.run(None, {self.input_name: preprocessed})
        self._inference_times.append((time.perf_counter() - t0) * 1000)

        detections = self._postprocess(output[0], ratio, pad_top, pad_left, orig_w, orig_h)

        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

        return self.tracker.update(detections, dt)

    def get_obstacles(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        return [d for d in detections if d.category in {"vehicle", "pedestrian", "cyclist"}]

    @property
    def avg_inference_ms(self) -> float:
        if not self._inference_times: return 0.0
        return sum(self._inference_times) / len(self._inference_times)


# ---- Helpers ----

def cv2_resize_safe(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize with a safe fallback if cv2 is unavailable."""
    try:
        import cv2
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    except ImportError:
        # Pure numpy fallback (slower but works)
        from numpy import array
        from PIL import Image as PILImage
        return array(PILImage.fromarray(image).resize(target_size))


def nms_numpy(boxes: np.ndarray, scores: np.ndarray,
              iou_threshold: float) -> np.ndarray:
    """Numpy-only NMS (no torch dependency)."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


# ---- Backward Compatibility: YOLODetector ----

class YOLODetector:
    """
    High-performance YOLO detector with metadata fail-safes.
    When use_onnx=True, delegates to ONNXRuntimeDetector for memory-safe inference.
    """
    def __init__(self, config: dict):
        self.config = config
        self.imgsz = config.get("imgsz", 320)
        self.conf_threshold = config.get("confidence_threshold", 0.4)

        self._vehicle_ids = set(config.get("vehicle_classes", [2, 3, 5, 7]))
        self._pedestrian_ids = set(config.get("pedestrian_classes", [0]))
        self._cyclist_ids = set(config.get("cyclist_classes", [1]))
        self._default_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        self._inference_times = []
        self._last_time = time.perf_counter()

        use_onnx = config.get("use_onnx", False)

        if use_onnx and ORT_AVAILABLE:
            print("[YOLODetector] Using ONNX Runtime backend (memory-safe)")
            self._backend = ONNXRuntimeDetector(config)
            self._use_ort = True
        else:
            if use_onnx and not ORT_AVAILABLE:
                print("[WARNING] use_onnx=True but onnxruntime not installed. Falling back to ultralytics.")
            self._load_ultralytics()
            self._use_ort = False

    def _load_ultralytics(self):
        from ultralytics import YOLO
        model_path = self.config.get("onnx_model_path", "model/yolo11n.onnx")
        print(f"[Detector] Initializing Backend: {model_path}")
        self.model = YOLO(model_path, task='detect')
        raw_names = getattr(self.model, 'names', None)
        if raw_names and isinstance(raw_names, dict):
            self._class_names = raw_names
        else:
            print("[WARNING] Model metadata (names) missing or corrupted. Using fallback map.")
            self._class_names = self._default_names
        self.tracker = SimpleIOUTracker()

    def _classify_category(self, class_id: int) -> str:
        if class_id in self._vehicle_ids: return "vehicle"
        if class_id in self._pedestrian_ids: return "pedestrian"
        if class_id in self._cyclist_ids: return "cyclist"
        return "other"

    def detect(self, image: np.ndarray) -> List[DetectedObject]:
        if self._use_ort:
            return self._backend.detect(image)

        # Ultralytics path (fallback)
        current_time = time.perf_counter()
        dt = current_time - self._last_time
        self._last_time = current_time

        t0 = time.perf_counter()
        results = self.model(image, verbose=False, imgsz=self.imgsz, conf=self.conf_threshold)
        self._inference_times.append((time.perf_counter() - t0) * 1000)

        detections = []
        if not results or results[0].boxes is None:
            return []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det = DetectedObject(
                class_id=cls_id,
                class_name=self._class_names.get(cls_id, f"class_{cls_id}"),
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                category=self._classify_category(cls_id)
            )
            detections.append(det)

        if len(self._inference_times) > 100: self._inference_times.pop(0)
        return self.tracker.update(detections, dt)

    def get_obstacles(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        if self._use_ort:
            return self._backend.get_obstacles(detections)
        return [d for d in detections if d.category in {"vehicle", "pedestrian", "cyclist"}]

    @property
    def avg_inference_ms(self) -> float:
        if self._use_ort:
            return self._backend.avg_inference_ms
        if not self._inference_times: return 0.0
        return sum(self._inference_times) / len(self._inference_times)
