"""
Frame Processor Module

Handles frame preprocessing and quality assessment.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from src.logger import log


class FrameProcessor:
    """
    Frame preprocessing and quality assessment

    Provides utilities for:
    - Frame resizing
    - Quality assessment (blur detection)
    - Normalization
    """

    @staticmethod
    def resize_frame(
        frame: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        max_dimension: Optional[int] = None
    ) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio

        Args:
            frame: Input frame
            target_size: Exact (width, height) to resize to
            max_dimension: Maximum dimension (width or height)

        Returns:
            Resized frame
        """
        if target_size is not None:
            return cv2.resize(frame, target_size)

        if max_dimension is not None:
            h, w = frame.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                return cv2.resize(frame, (new_w, new_h))

        return frame

    @staticmethod
    def assess_blur(frame: np.ndarray, threshold: float = 100.0) -> Tuple[float, bool]:
        """
        Assess frame blur using Laplacian variance

        Args:
            frame: Input frame
            threshold: Blur threshold (lower = more blurry)

        Returns:
            Tuple of (variance, is_sharp)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = laplacian_var >= threshold

        return laplacian_var, is_sharp

    @staticmethod
    def normalize_frame(frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to [0, 1] range

        Args:
            frame: Input frame

        Returns:
            Normalized frame
        """
        return frame.astype(np.float32) / 255.0

    @staticmethod
    def crop_bbox(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 0
    ) -> np.ndarray:
        """
        Crop frame to bounding box with optional padding

        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            padding: Padding pixels around bbox

        Returns:
            Cropped frame
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_frame, x + w + padding)
        y2 = min(h_frame, y + h + padding)

        return frame[y1:y2, x1:x2]

    @staticmethod
    def calculate_brightness(frame: np.ndarray) -> float:
        """
        Calculate average brightness of frame

        Args:
            frame: Input frame

        Returns:
            Average brightness [0, 255]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    @staticmethod
    def enhance_contrast(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Enhance frame contrast using CLAHE

        Args:
            frame: Input frame
            clip_limit: Contrast limiting threshold

        Returns:
            Contrast-enhanced frame
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def is_valid_detection(
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        min_size: int = 64
    ) -> bool:
        """
        Check if detection bounding box is valid

        Args:
            bbox: Bounding box (x, y, w, h)
            frame_shape: Frame (height, width)
            min_size: Minimum bbox width/height

        Returns:
            True if valid detection
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]

        # Check minimum size
        if w < min_size or h < min_size:
            return False

        # Check if bbox is within frame bounds
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return False

        # Check aspect ratio (person should be taller than wide)
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 1.2 or aspect_ratio > 5.0:
            return False

        return True
