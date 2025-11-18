# ProjectB

**Graph-Based Multi-View Identity Tracking System**

A computer vision system for tracking and identifying individuals across multiple camera views using graph databases and deep learning.

---

## 🎯 Overview

ProjectB is a CV-based identity tracking system that:
- Tracks people across multiple camera feeds
- Maintains persistent identities using face and body features
- Uses hybrid matching (deterministic + probabilistic)
- Stores relationships in a graph database (Neo4j)
- Provides analytics and visualization dashboards

**Key Technologies**: YOLOv8, InsightFace, FastReID, Neo4j, FAISS, PyTorch

---

## 📋 Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose** (for Neo4j and Redis)
- **Git**
- **8GB+ RAM** (16GB recommended)
- **CUDA-capable GPU** (optional, but recommended for performance)

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/thebibbi/ProjectB.git
cd ProjectB
```

### 2. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set your passwords
nano .env  # or use your preferred editor
```

**Important**: Change `NEO4J_PASSWORD` in `.env` before starting services!

### 4. Start Services

```bash
# Start Neo4j and Redis with Docker Compose
docker-compose up -d

# Check services are running
docker-compose ps

# View logs if needed
docker-compose logs -f
```

### 5. Initialize Neo4j Schema

```bash
# Run schema setup script
python scripts/setup_neo4j.py
```

When prompted, you can create sample camera nodes for testing.

### 6. Verify Installation

```bash
# Check Neo4j is accessible
# Open browser: http://localhost:7474
# Login with credentials from .env

# Run tests
pytest tests/ -v
```

---

## 📁 Project Structure

```
ProjectB/
├── src/                    # Source code
│   ├── ingestion/         # Video processing
│   ├── detection/         # Object detection & tracking
│   ├── features/          # Feature extraction (face, ReID)
│   ├── storage/           # Neo4j & FAISS storage
│   ├── resolution/        # Identity resolution engine
│   ├── analytics/         # Analytics & metrics
│   ├── api/               # FastAPI REST API
│   ├── tasks/             # Celery background tasks
│   ├── dashboard/         # Streamlit dashboard
│   ├── config.py          # Configuration management
│   └── logger.py          # Logging setup
│
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── data/                  # Data directory
│   ├── test_videos/      # Input videos
│   ├── models/           # Model weights
│   ├── indexes/          # FAISS indexes
│   └── output/           # Processing outputs
│
├── docs/                  # Documentation
│   └── PRDs/             # Product requirement docs
├── notebooks/             # Jupyter notebooks
├── logs/                  # Application logs
│
├── docker-compose.yml     # Docker services
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

---

## 🔧 Configuration

All configuration is managed through environment variables in `.env`:

### Key Settings

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Device Selection
DEVICE=cuda  # or 'cpu' if no GPU

# Thresholds
FACE_SIMILARITY_THRESHOLD=0.6
REID_SIMILARITY_THRESHOLD=0.5
HYBRID_SIMILARITY_THRESHOLD=0.55
```

See `.env.example` for all available settings.

---

## 📚 Documentation

- **[claude.md](./claude.md)** - Complete architecture overview
- **[TODO.md](./TODO.md)** - Development roadmap
- **[PRD-IdentityResolution.md](./docs/PRDs/PRD-IdentityResolution.md)** - Identity resolution spec
- **[PRD-GraphVectorStore.md](./docs/PRDs/PRD-GraphVectorStore.md)** - Storage layer spec
- **[PRD-Analytics.md](./docs/PRDs/PRD-Analytics.md)** - Analytics layer spec

---

## 🎯 Usage

### Process a Video (Coming in Phase 1)

```bash
python scripts/process_video.py --input data/test_videos/video.mp4 --camera-id cam_001
```

### Start API Server (Coming in Phase 1)

```bash
uvicorn src.api.main:app --reload
```

### Launch Dashboard (Coming in Phase 3)

```bash
streamlit run src/dashboard/app.py
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# View coverage report
open htmlcov/index.html  # or xdg-open on Linux
```

---

## 🛠 Development

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Check with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## 📊 Development Phases

### ✅ Phase 0: Setup & Infrastructure (Current)
- Project structure
- Docker services (Neo4j, Redis)
- Configuration management
- Logging and testing framework

### 🔄 Phase 1: Core Pipeline (Next - MVP)
- Video processing
- Object detection (YOLOv8)
- Face recognition (InsightFace)
- Basic identity clustering
- Neo4j storage

### 📅 Phase 2: Multi-View Tracking
- Person ReID (FastReID)
- Cross-camera matching
- Confidence scoring
- Advanced identity resolution

### 📅 Phase 3: Analytics & Reporting
- Trajectory analysis
- Quality metrics
- Visualization dashboard
- Export functionality

### 📅 Phase 4: Advanced Features
- Real-time processing
- Active learning
- Performance optimization
- Production deployment

See [TODO.md](./TODO.md) for detailed roadmap.

---

## 🐛 Troubleshooting

### Neo4j Connection Issues

```bash
# Check if Neo4j is running
docker-compose ps

# View Neo4j logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j
```

### GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, set DEVICE=cpu in .env
```

### Memory Issues

- Reduce `BATCH_SIZE` in `.env`
- Process videos at lower FPS (`FPS_SAMPLE_RATE`)
- Use smaller models (YOLOv8n instead of YOLOv8x)

### Port Conflicts

If ports 7474, 7687, or 6379 are in use:

```bash
# Edit docker-compose.yml to use different ports
# Update .env to match new ports
```

---

## 🤝 Contributing

This is currently an internal project. See `CONTRIBUTING.md` (coming soon) for guidelines.

---

## 📄 License

MIT License (to be added)

---

## 🔗 Useful Links

- **Neo4j Browser**: http://localhost:7474
- **API Docs** (when running): http://localhost:8000/docs
- **Dashboard** (when running): http://localhost:8501

---

## 📞 Support

For issues or questions:
1. Check the documentation in `docs/`
2. Review the troubleshooting section above
3. Check existing issues in the repository

---

## 🙏 Acknowledgments

Built with state-of-the-art open-source technologies:
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
- [FastReID](https://github.com/JDAI-CV/fast-reid) - Person re-identification
- [Neo4j](https://neo4j.com/) - Graph database
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

---

**Status**: Phase 0 Complete ✅ | Next: Phase 1 MVP Development

Last Updated: 2025-11-18
