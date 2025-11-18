"""
Person Detector using YOLOv8

Detects people in video frames using Ultralytics YOLOv8.
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

from src.logger import log
from src.config import settings
from src.storage.models import Detection, BoundingBox


class PersonDetector:
    """
    YOLOv8-based person detector

    Detects people in images/video frames.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        """
        Initialize YOLOv8 detector

        Args:
            model_path: Path to YOLO model (default: yolov8n.pt)
            confidence_threshold: Detection confidence threshold
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

        self.model_path = model_path or settings.yolo_model_path
        self.confidence_threshold = confidence_threshold or settings.detection_confidence_threshold
        self.device = device or settings.device

        log.info(f"Loading YOLO model: {self.model_path}")
        log.info(f"  Device: {self.device}")
        log.info(f"  Confidence threshold: {self.confidence_threshold}")

        # Download model if not exists
        if not Path(self.model_path).exists():
            log.warning(f"Model not found at {self.model_path}, downloading...")
            self.model_path = "yolov8n.pt"  # Will auto-download

        self.model = YOLO(str(self.model_path))

        # Set device
        if self.device != 'cpu':
            self.model.to(self.device)

        log.success("YOLOv8 detector ready")

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> List[Detection]:
        """
        Detect people in a frame

        Args:
            frame: Input frame (BGR format)
            frame_number: Frame number for tracking
            timestamp: Timestamp in video (seconds)

        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=[0],  # 0 = person in COCO dataset
            verbose=False,
            device=self.device
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Convert to x, y, w, h format
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                # Validate bbox
                if w < settings.min_detection_size or h < settings.min_detection_size:
                    continue

                detection = Detection(
                    bbox=BoundingBox(x=x, y=y, w=w, h=h),
                    confidence=confidence,
                    class_id=class_id,
                    class_name="person",
                    frame_number=frame_number,
                    timestamp=timestamp
                )

                detections.append(detection)

        log.debug(f"Frame {frame_number}: {len(detections)} detections")

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        frame_numbers: Optional[List[int]] = None,
        timestamps: Optional[List[float]] = None
    ) -> List[List[Detection]]:
        """
        Detect people in multiple frames (batch processing)

        Args:
            frames: List of frames
            frame_numbers: List of frame numbers
            timestamps: List of timestamps

        Returns:
            List of detection lists (one per frame)
        """
        if frame_numbers is None:
            frame_numbers = list(range(len(frames)))

        if timestamps is None:
            timestamps = [0.0] * len(frames)

        all_detections = []

        for i, frame in enumerate(frames):
            detections = self.detect(frame, frame_numbers[i], timestamps[i])
            all_detections.append(detections)

        return all_detections


def download_yolo_model(model_name: str = "yolov8n.pt") -> str:
    """
    Download YOLO model if not exists

    Args:
        model_name: Model name (yolov8n, yolov8s, yolov8m, etc.)

    Returns:
        Path to downloaded model
    """
    from ultralytics import YOLO

    log.info(f"Downloading {model_name}...")
    model = YOLO(model_name)

    # Model is automatically downloaded to ultralytics cache
    log.success(f"Model {model_name} ready")

    return model_name
