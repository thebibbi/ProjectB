"""
Face Feature Extractor using InsightFace

Extracts face embeddings for identity matching.
"""

import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from src.logger import log
from src.config import settings
from src.storage.models import FaceEmbedding, BoundingBox


class FaceExtractor:
    """
    InsightFace-based face feature extractor

    Detects faces and extracts 512-dim embeddings for recognition.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        detection_threshold: Optional[float] = None
    ):
        """
        Initialize InsightFace extractor

        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_s, etc.)
            detection_threshold: Face detection confidence threshold
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface not installed. Run: pip install insightface onnxruntime"
            )

        self.model_name = model_name
        self.detection_threshold = detection_threshold or settings.face_detection_threshold

        log.info(f"Loading InsightFace model: {model_name}")
        log.info(f"  Detection threshold: {self.detection_threshold}")

        # Initialize face analysis
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CPUExecutionProvider']  # Can add GPU providers if available
        )

        self.app.prepare(
            ctx_id=0 if settings.device == 'cpu' else 0,
            det_thresh=self.detection_threshold
        )

        log.success("InsightFace extractor ready")

    def extract(
        self,
        image: np.ndarray,
        min_face_size: Optional[int] = None
    ) -> List[FaceEmbedding]:
        """
        Extract face embeddings from image

        Args:
            image: Input image (BGR format from OpenCV)
            min_face_size: Minimum face size (width or height)

        Returns:
            List of FaceEmbedding objects
        """
        min_face_size = min_face_size or settings.face_min_size

        # Detect and analyze faces
        # InsightFace expects RGB, so convert if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]  # BGR to RGB
        else:
            image_rgb = image

        faces = self.app.get(image_rgb)

        embeddings = []

        for face in faces:
            # Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1

            # Filter by minimum size
            if w < min_face_size or h < min_face_size:
                continue

            # Get embedding (normalized to unit vector)
            embedding = face.normed_embedding.flatten().tolist()

            # Get landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
            landmarks = face.kps.tolist() if hasattr(face, 'kps') else None

            # Calculate quality score based on detection confidence and size
            det_score = float(face.det_score) if hasattr(face, 'det_score') else 0.9
            size_score = min(1.0, (w * h) / (100 * 100))  # Normalize by 100x100
            quality_score = (det_score + size_score) / 2

            face_emb = FaceEmbedding(
                embedding=embedding,
                bbox=BoundingBox(x=x1, y=y1, w=w, h=h),
                quality_score=quality_score,
                landmarks=landmarks
            )

            embeddings.append(face_emb)

        log.debug(f"Extracted {len(embeddings)} face embeddings")

        return embeddings

    def extract_from_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 20
    ) -> Optional[FaceEmbedding]:
        """
        Extract face embedding from a specific region

        Args:
            image: Full image
            bbox: Bounding box (x, y, w, h) of person
            padding: Padding around bbox

        Returns:
            FaceEmbedding or None if no face found
        """
        x, y, w, h = bbox

        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        # Crop region
        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        # Extract faces from cropped region
        faces = self.extract(cropped)

        if len(faces) == 0:
            return None

        # Return face with highest quality score
        best_face = max(faces, key=lambda f: f.quality_score)

        # Adjust bbox coordinates to full image
        best_face.bbox.x += x1
        best_face.bbox.y += y1

        return best_face

    @staticmethod
    def calculate_similarity(
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two face embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score [0, 1] (higher = more similar)
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Cosine similarity (embeddings are already normalized)
        similarity = np.dot(emb1, emb2)

        # Clip to [0, 1] range
        return float(np.clip(similarity, 0.0, 1.0))


def get_face_model_path(model_name: str = "buffalo_l") -> Path:
    """
    Get path to InsightFace model

    Args:
        model_name: Model name

    Returns:
        Path to model directory
    """
    # InsightFace stores models in ~/.insightface/models/
    home = Path.home()
    model_path = home / ".insightface" / "models" / model_name

    return model_path
