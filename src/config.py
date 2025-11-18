"""
Configuration Management

Centralized configuration using Pydantic settings.
Loads from environment variables and .env file.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Application
    app_name: str = Field(default="ProjectB")
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="projectb_password")
    neo4j_database: str = Field(default="neo4j")

    # Redis Configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)

    # FAISS Configuration
    faiss_index_dir: Path = Field(default=Path("./data/indexes"))
    faiss_face_dim: int = Field(default=512)
    faiss_reid_dim: int = Field(default=2048)

    # Model Paths
    models_dir: Path = Field(default=Path("./data/models"))
    yolo_model_path: Path = Field(default=Path("./data/models/yolov8n.pt"))
    insightface_model_path: Path = Field(default=Path("./data/models/buffalo_l"))
    fastreid_model_path: Path = Field(default=Path("./data/models/market_bot_R50.pth"))

    # Processing Configuration
    batch_size: int = Field(default=16)
    num_workers: int = Field(default=4)
    device: str = Field(default="cpu")
    fps_sample_rate: int = Field(default=5)  # Process every Nth frame

    # Detection Configuration
    detection_confidence_threshold: float = Field(default=0.5)
    detection_iou_threshold: float = Field(default=0.45)
    min_detection_size: int = Field(default=64)

    # Face Recognition Configuration
    face_detection_threshold: float = Field(default=0.8)
    face_min_size: int = Field(default=64)
    face_similarity_threshold: float = Field(default=0.6)

    # ReID Configuration
    reid_similarity_threshold: float = Field(default=0.5)
    reid_min_size: int = Field(default=128)

    # Identity Resolution Configuration
    hybrid_similarity_threshold: float = Field(default=0.55)
    face_weight: float = Field(default=0.6)
    reid_weight: float = Field(default=0.4)
    temporal_decay_constant: int = Field(default=3600)  # seconds
    max_temporal_gap: int = Field(default=86400)  # 24 hours in seconds

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    api_reload: bool = Field(default=True)

    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/0")
    celery_result_backend: str = Field(default="redis://localhost:6379/0")

    # Dashboard Configuration
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_port: int = Field(default=8501)

    # Data Paths
    data_dir: Path = Field(default=Path("./data"))
    output_dir: Path = Field(default=Path("./data/output"))
    test_videos_dir: Path = Field(default=Path("./data/test_videos"))

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.faiss_index_dir,
            self.models_dir,
            self.data_dir,
            self.output_dir,
            self.test_videos_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def redis_url(self) -> str:
        """Get Redis URL for connections"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.ensure_directories()
