"""
Data Models for Identity Tracking System

Pydantic models for type safety and validation.
"""

from datetime import datetime
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field
from uuid import uuid4


def generate_uuid() -> str:
    """Generate a new UUID string"""
    return str(uuid4())


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x: int = Field(..., description="Top-left x coordinate")
    y: int = Field(..., description="Top-left y coordinate")
    w: int = Field(..., description="Width")
    h: int = Field(..., description="Height")

    @property
    def area(self) -> int:
        """Calculate bounding box area"""
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point"""
        return (self.x + self.w / 2, self.y + self.h / 2)


class Detection(BaseModel):
    """Object detection result"""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int = Field(default=0, description="Class ID (0 for person)")
    class_name: str = Field(default="person")
    frame_number: int
    timestamp: float = Field(..., description="Timestamp in video (seconds)")


class TrackingResult(BaseModel):
    """Multi-object tracking result"""
    tracking_id: int = Field(..., description="Unique tracking ID within video")
    detection: Detection
    track_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FaceEmbedding(BaseModel):
    """Face embedding data"""
    embedding: List[float] = Field(..., description="512-dim face embedding")
    bbox: BoundingBox = Field(..., description="Face bounding box")
    quality_score: float = Field(..., ge=0.0, le=1.0)
    landmarks: Optional[List[Tuple[float, float]]] = None


class AppearanceCreate(BaseModel):
    """Schema for creating a new appearance"""
    timestamp: datetime
    camera_id: str
    tracking_id: int
    bbox: BoundingBox
    detection_confidence: float
    face_quality: Optional[float] = None
    reid_quality: float = 0.0
    blur_score: float = 0.0
    frame_path: str


class Appearance(BaseModel):
    """Complete appearance record"""
    id: str = Field(default_factory=generate_uuid)
    timestamp: datetime
    camera_id: str
    tracking_id: int

    # Bounding box
    bbox: BoundingBox

    # Embeddings (references to vector store)
    face_embedding_idx: int = Field(default=-1, description="-1 if no face detected")
    reid_embedding_idx: int = Field(default=-1)

    # Quality metrics
    detection_confidence: float
    face_quality: Optional[float] = None
    reid_quality: float = 0.0
    blur_score: float = 0.0

    # Metadata
    frame_path: str
    match_confidence: float = 1.0
    identity_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "app_123abc",
                "timestamp": "2025-11-18T10:30:00",
                "camera_id": "cam_entrance",
                "tracking_id": 42,
                "bbox": {"x": 100, "y": 200, "w": 150, "h": 400},
                "detection_confidence": 0.95,
                "frame_path": "/path/to/frame_001234.jpg"
            }
        }


class IdentityCreate(BaseModel):
    """Schema for creating a new identity"""
    cluster_method: str = Field(default="tracking_id")
    has_face: bool = False


class Identity(BaseModel):
    """Complete identity record"""
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Temporal info
    first_seen: datetime
    last_seen: datetime

    # Statistics
    num_appearances: int = 0
    num_cameras: int = 0

    # Quality metrics
    confidence_score: float = 0.0
    has_face: bool = False

    # Cluster info
    cluster_method: str = "tracking_id"
    merge_history: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "id_xyz789",
                "created_at": "2025-11-18T10:00:00",
                "first_seen": "2025-11-18T10:00:00",
                "last_seen": "2025-11-18T15:30:00",
                "num_appearances": 47,
                "confidence_score": 0.87
            }
        }


class Camera(BaseModel):
    """Camera configuration"""
    id: str
    name: str
    location: str
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    orientation: str = "north"
    fov: float = 90.0
    created_at: datetime = Field(default_factory=datetime.now)


class VideoProcessingJob(BaseModel):
    """Video processing job status"""
    id: str = Field(default_factory=generate_uuid)
    video_path: str
    camera_id: str
    status: str = Field(default="pending")  # pending, processing, completed, failed
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    total_frames: int = 0
    processed_frames: int = 0
    detections_count: int = 0
    identities_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class MatchScore(BaseModel):
    """Similarity match between two appearances"""
    appearance1_id: str
    appearance2_id: str
    face_similarity: Optional[float] = None
    reid_similarity: Optional[float] = None
    combined_similarity: float
    temporal_factor: float = 1.0
    spatial_factor: float = 1.0
    final_score: float
    is_match: bool
    method: str  # 'deterministic' or 'probabilistic'
