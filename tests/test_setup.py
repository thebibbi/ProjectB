"""
Test suite for verifying Phase 0 setup is complete
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
def test_imports():
    """Test that all core modules can be imported"""
    try:
        from src import config
        from src import logger
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


@pytest.mark.unit
def test_config_loading():
    """Test that configuration loads correctly"""
    from src.config import settings

    assert settings.app_name == "ProjectB"
    assert settings.neo4j_uri is not None
    assert settings.neo4j_user is not None


@pytest.mark.unit
def test_logger_initialization():
    """Test that logger initializes without errors"""
    from src.logger import log

    log.info("Test log message")
    assert True


@pytest.mark.unit
def test_directory_structure():
    """Test that required directories exist"""
    from src.config import settings

    required_dirs = [
        settings.data_dir,
        settings.output_dir,
        settings.test_videos_dir,
        settings.faiss_index_dir,
        settings.models_dir,
    ]

    for directory in required_dirs:
        assert directory.exists(), f"Directory {directory} does not exist"


@pytest.mark.unit
def test_env_file_exists():
    """Test that .env.example exists"""
    env_example = Path(".env.example")
    assert env_example.exists(), ".env.example file is missing"


@pytest.mark.integration
def test_neo4j_connection():
    """Test Neo4j connection (requires Docker services running)"""
    pytest.skip("Neo4j connection test - run manually after setup")
    # This is a placeholder for integration testing


@pytest.mark.integration
def test_redis_connection():
    """Test Redis connection (requires Docker services running)"""
    pytest.skip("Redis connection test - run manually after setup")
    # This is a placeholder for integration testing
