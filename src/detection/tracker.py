"""
Multi-Object Tracker

Simple IOU-based tracker for person tracking across frames.
For MVP, using simple matching. Can upgrade to ByteTrack later.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from src.logger import log
from src.storage.models import Detection, TrackingResult, BoundingBox


def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        IoU score [0, 1]
    """
    # Calculate intersection
    x1 = max(bbox1.x, bbox2.x)
    y1 = max(bbox1.y, bbox2.y)
    x2 = min(bbox1.x + bbox1.w, bbox2.x + bbox2.w)
    y2 = min(bbox1.y + bbox1.h, bbox2.y + bbox2.h)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    area1 = bbox1.area
    area2 = bbox2.area
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class SimpleTracker:
    """
    Simple IOU-based multi-object tracker

    Tracks people across frames using bounding box overlap.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3
    ):
        """
        Initialize tracker

        Args:
            iou_threshold: Minimum IOU for matching
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum detections before track is confirmed
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: Dict[int, dict] = {}  # tracking_id -> track_data
        self.next_id = 1
        self.frame_count = 0

        log.info("SimpleTracker initialized")
        log.info(f"  IOU threshold: {iou_threshold}")
        log.info(f"  Max age: {max_age} frames")
        log.info(f"  Min hits: {min_hits}")

    def update(self, detections: List[Detection]) -> List[TrackingResult]:
        """
        Update tracker with new detections

        Args:
            detections: List of detections from current frame

        Returns:
            List of tracking results
        """
        self.frame_count += 1

        # Match detections to existing tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._match(detections)

        # Update matched tracks
        results = []
        for track_id, detection in matched_tracks:
            track = self.tracks[track_id]
            track['bbox'] = detection.bbox
            track['age'] = 0
            track['hits'] += 1
            track['confidence'] = detection.confidence

            # Only return confirmed tracks
            if track['hits'] >= self.min_hits:
                results.append(TrackingResult(
                    tracking_id=track_id,
                    detection=detection,
                    track_confidence=track['confidence']
                ))

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                'bbox': detection.bbox,
                'age': 0,
                'hits': 1,
                'confidence': detection.confidence
            }

        # Age unmatched tracks
        tracks_to_delete = []
        for track_id in unmatched_tracks:
            self.tracks[track_id]['age'] += 1

            if self.tracks[track_id]['age'] > self.max_age:
                tracks_to_delete.append(track_id)

        # Delete old tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]

        log.debug(f"Frame {self.frame_count}: {len(results)} active tracks, {len(self.tracks)} total")

        return results

    def _match(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, Detection]], List[Detection], List[int]]:
        """
        Match detections to existing tracks using IOU

        Returns:
            Tuple of (matched_tracks, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], detections, []

        if len(detections) == 0:
            return [], [], list(self.tracks.keys())

        # Calculate IOU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        track_ids = list(self.tracks.keys())

        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracks[track_id]['bbox']

            for j, detection in enumerate(detections):
                iou_matrix[i, j] = calculate_iou(track_bbox, detection.bbox)

        # Greedy matching (simple Hungarian algorithm alternative)
        matched_tracks = []
        matched_detection_indices = set()
        matched_track_indices = set()

        # Match in order of highest IOU
        while True:
            max_iou = np.max(iou_matrix)

            if max_iou < self.iou_threshold:
                break

            # Find indices of max IOU
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

            # Skip if already matched
            if i in matched_track_indices or j in matched_detection_indices:
                iou_matrix[i, j] = 0
                continue

            # Add match
            track_id = track_ids[i]
            matched_tracks.append((track_id, detections[j]))
            matched_detection_indices.add(j)
            matched_track_indices.add(i)

            # Zero out this match
            iou_matrix[i, j] = 0

        # Find unmatched detections
        unmatched_detections = [
            det for i, det in enumerate(detections)
            if i not in matched_detection_indices
        ]

        # Find unmatched tracks
        unmatched_tracks = [
            track_ids[i] for i in range(len(track_ids))
            if i not in matched_track_indices
        ]

        return matched_tracks, unmatched_detections, unmatched_tracks

    def reset(self):
        """Reset tracker state"""
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        log.info("Tracker reset")

    def get_active_tracks(self) -> List[int]:
        """Get list of active track IDs"""
        return [
            track_id for track_id, track in self.tracks.items()
            if track['hits'] >= self.min_hits
        ]

    def get_track_count(self) -> int:
        """Get total number of tracks created"""
        return self.next_id - 1
