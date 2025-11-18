"""
Video Loader Module

Handles video file loading and frame extraction using OpenCV.
"""

import cv2
from pathlib import Path
from typing import Optional, Generator, Tuple
import numpy as np

from src.logger import log
from src.config import settings


class VideoLoader:
    """
    Video loader with frame extraction capabilities

    Usage:
        loader = VideoLoader("video.mp4")
        for frame_number, frame, timestamp in loader.frames():
            # Process frame
            pass
    """

    def __init__(self, video_path: str):
        """
        Initialize video loader

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Extract video metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        log.info(f"Video loaded: {self.video_path.name}")
        log.info(f"  Resolution: {self.width}x{self.height}")
        log.info(f"  FPS: {self.fps:.2f}")
        log.info(f"  Frames: {self.total_frames}")
        log.info(f"  Duration: {self.duration:.2f}s")

    def frames(
        self,
        sample_rate: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Generate frames from video

        Args:
            sample_rate: Process every Nth frame (default from config)
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all frames)

        Yields:
            Tuple of (frame_number, frame_image, timestamp_seconds)
        """
        sample_rate = sample_rate or settings.fps_sample_rate
        end_frame = end_frame or self.total_frames

        log.info(f"Processing frames {start_frame} to {end_frame}, sample rate: {sample_rate}")

        # Set starting position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = start_frame

        while frame_count < end_frame:
            ret, frame = self.cap.read()

            if not ret:
                log.warning(f"Failed to read frame {frame_count}")
                break

            # Only yield frames according to sample rate
            if (frame_count - start_frame) % sample_rate == 0:
                timestamp = frame_count / self.fps if self.fps > 0 else 0
                yield frame_count, frame, timestamp

            frame_count += 1

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number

        Args:
            frame_number: Frame index to retrieve

        Returns:
            Frame image or None if frame doesn't exist
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            log.warning(f"Frame {frame_number} out of range [0, {self.total_frames})")
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        return frame if ret else None

    def save_frame(
        self,
        frame: np.ndarray,
        output_path: str,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> bool:
        """
        Save frame to disk, optionally cropping to bbox

        Args:
            frame: Frame image
            output_path: Where to save the frame
            bbox: Optional bounding box (x, y, w, h) to crop

        Returns:
            True if successful
        """
        try:
            if bbox is not None:
                x, y, w, h = bbox
                # Ensure bbox is within frame bounds
                x = max(0, min(x, frame.shape[1]))
                y = max(0, min(y, frame.shape[0]))
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                frame = frame[y:y+h, x:x+w]

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, frame)
            return True

        except Exception as e:
            log.error(f"Failed to save frame: {e}")
            return False

    def release(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
            log.debug(f"Released video: {self.video_path.name}")

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.release()

    def __del__(self):
        """Cleanup on deletion"""
        self.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata without loading the full video

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        "path": video_path,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

    cap.release()

    return info
