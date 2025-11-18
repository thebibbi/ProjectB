"""
Configuration Management

Centralized configuration using Pydantic settings.
Loads from environment variables and .env file.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""

    # Application
    app_name: str = Field(default="ProjectB", env="APP_NAME")
    app_env: str = Field(default="development", env="APP_ENV")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="projectb_password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")

    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    # FAISS Configuration
    faiss_index_dir: Path = Field(default=Path("./data/indexes"), env="FAISS_INDEX_DIR")
    faiss_face_dim: int = Field(default=512, env="FAISS_FACE_DIM")
    faiss_reid_dim: int = Field(default=2048, env="FAISS_REID_DIM")

    # Model Paths
    models_dir: Path = Field(default=Path("./data/models"), env="MODELS_DIR")
    yolo_model_path: Path = Field(
        default=Path("./data/models/yolov8n.pt"), env="YOLO_MODEL_PATH"
    )
    insightface_model_path: Path = Field(
        default=Path("./data/models/buffalo_l"), env="INSIGHTFACE_MODEL_PATH"
    )
    fastreid_model_path: Path = Field(
        default=Path("./data/models/market_bot_R50.pth"), env="FASTREID_MODEL_PATH"
    )

    # Processing Configuration
    batch_size: int = Field(default=16, env="BATCH_SIZE")
    num_workers: int = Field(default=4, env="NUM_WORKERS")
    device: str = Field(default="cpu", env="DEVICE")
    fps_sample_rate: int = Field(
        default=5, env="FPS_SAMPLE_RATE"
    )  # Process every Nth frame

    # Detection Configuration
    detection_confidence_threshold: float = Field(
        default=0.5, env="DETECTION_CONFIDENCE_THRESHOLD"
    )
    detection_iou_threshold: float = Field(default=0.45, env="DETECTION_IOU_THRESHOLD")
    min_detection_size: int = Field(default=64, env="MIN_DETECTION_SIZE")

    # Face Recognition Configuration
    face_detection_threshold: float = Field(default=0.8, env="FACE_DETECTION_THRESHOLD")
    face_min_size: int = Field(default=64, env="FACE_MIN_SIZE")
    face_similarity_threshold: float = Field(
        default=0.6, env="FACE_SIMILARITY_THRESHOLD"
    )

    # ReID Configuration
    reid_similarity_threshold: float = Field(default=0.5, env="REID_SIMILARITY_THRESHOLD")
    reid_min_size: int = Field(default=128, env="REID_MIN_SIZE")

    # Identity Resolution Configuration
    hybrid_similarity_threshold: float = Field(
        default=0.55, env="HYBRID_SIMILARITY_THRESHOLD"
    )
    face_weight: float = Field(default=0.6, env="FACE_WEIGHT")
    reid_weight: float = Field(default=0.4, env="REID_WEIGHT")
    temporal_decay_constant: int = Field(
        default=3600, env="TEMPORAL_DECAY_CONSTANT"
    )  # seconds
    max_temporal_gap: int = Field(
        default=86400, env="MAX_TEMPORAL_GAP"
    )  # 24 hours in seconds

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_reload: bool = Field(default=True, env="API_RELOAD")

    # Celery Configuration
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0", env="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND"
    )

    # Dashboard Configuration
    dashboard_host: str = Field(default="0.0.0.0", env="DASHBOARD_HOST")
    dashboard_port: int = Field(default=8501, env="DASHBOARD_PORT")

    # Data Paths
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    output_dir: Path = Field(default=Path("./data/output"), env="OUTPUT_DIR")
    test_videos_dir: Path = Field(
        default=Path("./data/test_videos"), env="TEST_VIDEOS_DIR"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

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
