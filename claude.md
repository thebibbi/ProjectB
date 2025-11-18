# ProjectB: Graph-Based Multi-View Identity Tracking

## Project Overview

ProjectB is a computer vision (CV) identity tracking system that maintains persistent identity graphs across multiple video sources and camera views. The system uses hybrid identity resolution combining deterministic tracking with probabilistic feature matching to associate visual appearances across different viewpoints, time periods, and devices.

### Core Purpose

Track and maintain persistent identities of people across:
- Multiple camera feeds (multi-view tracking)
- Different time periods (temporal tracking)
- Various lighting conditions and angles
- Re-identification after occlusion or absence

### Key Use Cases

1. **Multi-Camera Surveillance**: Track individuals across multiple camera feeds in a facility
2. **Long-term Re-identification**: Re-identify people appearing in different sessions/days
3. **Trajectory Analysis**: Analyze movement patterns and spatial relationships
4. **Identity Clustering**: Group appearances by the same individual
5. **Forensic Search**: Query "all appearances of person X" across time and space

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input Layer                                  │
│  Video Streams → Frame Extraction → Detection & Tracking        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                Feature Extraction Layer                          │
│  Face Recognition │ Person ReID │ Appearance Features           │
│  (InsightFace)    │ (FastReID)  │ (Color, Pose, etc.)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Identity Resolution Engine (Hybrid)                 │
│  ┌──────────────────┐          ┌─────────────────────┐         │
│  │  Deterministic   │          │   Probabilistic     │         │
│  │  - Tracking IDs  │    +     │   - Embedding Sim   │         │
│  │  - Exact Matches │          │   - Temporal Prox   │         │
│  │  - Same Camera   │          │   - Spatial Context │         │
│  └──────────────────┘          └─────────────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Storage Layer                                  │
│  ┌──────────────────────┐    ┌────────────────────────┐        │
│  │  Vector Database     │    │  Graph Database        │        │
│  │  (Milvus/FAISS)     │    │  (Neo4j)               │        │
│  │  - Face embeddings   │    │  - Identity nodes      │        │
│  │  - ReID features     │    │  - Appearance edges    │        │
│  │  - Similarity search │    │  - Temporal relations  │        │
│  └──────────────────────┘    └────────────────────────┘        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Analytics & Query Layer                          │
│  Graph Queries │ Trajectory Analysis │ Confidence Scoring       │
│  Temporal Tracking │ Identity Merging │ Reporting                │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion**: Video frames → Object detection → Tracking
2. **Feature Extraction**: Extract face/body embeddings for each detection
3. **Identity Resolution**:
   - **Deterministic**: Link detections with same tracking ID in single camera
   - **Probabilistic**: Match embeddings across cameras/sessions using similarity
4. **Graph Update**: Create/update identity nodes and relationship edges
5. **Query & Analytics**: Search, analyze, and report on identity graph

---

## Technology Stack

### Core Technologies

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Graph Database** | Neo4j Community Edition | Largest community, excellent documentation, powerful Cypher query language, graph algorithms library |
| **Vector Database** | Milvus or FAISS | FAISS for simplicity (Facebook AI), Milvus for scalability with large embeddings |
| **Object Detection** | YOLOv8/YOLOv9 | SOTA detection speed and accuracy |
| **Object Tracking** | ByteTrack / Deep SORT | State-of-the-art multi-object tracking |
| **Face Recognition** | InsightFace | Best open-source face recognition, pre-trained models, active community |
| **Person ReID** | FastReID / torchreid | SOTA person re-identification across cameras |
| **Graph Processing** | NetworkX + Neo4j GDS | NetworkX for prototyping, Neo4j GDS for production graph algorithms |
| **API Framework** | FastAPI | Modern, fast, automatic API documentation |
| **Task Queue** | Celery + Redis | Batch processing of video files |
| **Analytics** | Neo4j Browser + Custom Dashboard | Graph visualization and custom metrics |

### Python Libraries

```python
# Computer Vision
opencv-python          # Video processing
ultralytics           # YOLOv8/v9 for detection
insightface           # Face recognition
fast-reid             # Person re-identification
torch, torchvision    # Deep learning framework

# Graph & Vector Storage
neo4j                 # Graph database driver
pymilvus              # Milvus vector database client (or faiss-cpu)
networkx              # Graph algorithms and analysis

# API & Services
fastapi               # REST API framework
celery                # Task queue for batch processing
redis                 # Message broker

# Data Processing
numpy                 # Numerical computing
pandas                # Data manipulation
scikit-learn          # ML utilities (clustering, metrics)

# Utilities
pydantic              # Data validation
loguru                # Logging
python-dotenv         # Configuration management
```

---

## Key Concepts

### Identity Resolution Methods

#### 1. Deterministic Matching
- **Same tracking ID** within a video sequence
- **Exact feature hash** matches (rare but high confidence)
- **Same camera + short time gap** (< 5 minutes)

#### 2. Probabilistic Matching
- **Face embedding similarity** (cosine similarity > 0.6)
- **ReID feature similarity** (for body/appearance matching)
- **Temporal proximity** (appeared in plausible time window)
- **Spatial context** (logical path between camera locations)
- **Appearance consistency** (clothing, height, etc.)

#### 3. Hybrid Approach
- Start with deterministic links (high confidence)
- Add probabilistic links with confidence scores
- Use graph algorithms to refine clusters
- Human-in-the-loop for ambiguous cases

### Graph Schema

#### Nodes
```
(:Identity {
  id: UUID,
  created_at: timestamp,
  confidence_score: float,
  num_appearances: int,
  first_seen: timestamp,
  last_seen: timestamp
})

(:Appearance {
  id: UUID,
  identity_id: UUID,
  timestamp: timestamp,
  camera_id: string,
  tracking_id: int,
  bbox: [x, y, w, h],
  face_embedding_id: string,  // reference to vector DB
  reid_embedding_id: string,   // reference to vector DB
  frame_path: string,
  confidence: float
})

(:Camera {
  id: string,
  location: string,
  position: [x, y, z]
})
```

#### Relationships
```
(:Identity)-[:HAS_APPEARANCE {confidence: float}]->(:Appearance)
(:Appearance)-[:SAME_TRACK {method: 'deterministic'}]->(:Appearance)
(:Appearance)-[:PROBABLE_MATCH {similarity: float, method: 'face'}]->(:Appearance)
(:Appearance)-[:CAPTURED_BY]->(:Camera)
(:Appearance)-[:NEXT {time_delta: int}]->(:Appearance)  // temporal sequence
```

### Confidence Scoring

Each identity has a confidence score based on:
- **Number of appearances**: More sightings = higher confidence
- **Feature quality**: High-quality face/body features
- **Matching consistency**: Consistent across multiple matchers
- **Temporal coverage**: Seen over longer time periods
- **Spatial diversity**: Seen across multiple cameras

---

## Features & Capabilities

### Phase 1: Core Identity Resolution (MVP)
- ✅ Single video ingestion and processing
- ✅ Object detection and tracking (YOLOv8 + ByteTrack)
- ✅ Face embedding extraction (InsightFace)
- ✅ Basic identity clustering using face similarity
- ✅ Neo4j graph storage with basic schema
- ✅ Simple query API

### Phase 2: Multi-View Tracking
- Multi-camera ingestion
- Person ReID integration (FastReID)
- Cross-camera identity resolution
- Temporal tracking improvements
- Vector database integration (Milvus/FAISS)

### Phase 3: Analytics & Reporting
- Confidence scoring system
- Trajectory analysis
- Identity merge/split operations
- Graph visualization dashboard
- Quality metrics and reporting

### Phase 4: Advanced Features
- Real-time processing mode
- Active learning for identity refinement
- Appearance attribute extraction (clothing, accessories)
- Scene context integration
- Export and integration APIs

---

## Development Approach

### Principles
1. **Incremental Development**: Build MVP first, iterate based on results
2. **Modular Design**: Loosely coupled components for flexibility
3. **Batch-First**: Optimize for batch processing, add real-time later
4. **Quality over Speed**: Accuracy more important than processing speed
5. **Experimentation-Friendly**: Easy to swap algorithms and parameters

### Technology Decisions

#### Why Neo4j?
- **Community**: Largest graph database community for support
- **Cypher**: Intuitive query language for complex graph patterns
- **Algorithms**: Built-in graph algorithms (community detection, centrality, etc.)
- **Visualization**: Excellent built-in browser for exploration
- **Python Integration**: Mature Python driver

#### Why InsightFace?
- **Accuracy**: SOTA face recognition models
- **Pre-trained Models**: Ready-to-use ArcFace, RetinaFace models
- **Speed**: Optimized for inference
- **Active Development**: Regular updates and improvements

#### Why FastReID?
- **Flexibility**: Supports multiple ReID architectures
- **Pre-trained Models**: Models trained on multiple datasets
- **Research-Backed**: Based on latest research papers
- **Extensible**: Easy to add custom features

#### Vector Database Choice: FAISS vs Milvus
- **Start with FAISS**: Simpler, local deployment, good for MVP
- **Migrate to Milvus**: If scaling beyond millions of embeddings
- **Both are open-source**: No vendor lock-in

---

## Deployment Architecture

### Local Deployment (MVP)

```
┌──────────────────────────────────────────┐
│         Docker Compose Stack              │
│                                           │
│  ┌────────────┐  ┌──────────────┐       │
│  │   Neo4j    │  │    Redis     │       │
│  │  (Graph)   │  │  (Broker)    │       │
│  └────────────┘  └──────────────┘       │
│                                           │
│  ┌────────────────────────────────────┐  │
│  │     FastAPI Application            │  │
│  │  - REST API                        │  │
│  │  - Query Interface                 │  │
│  └────────────────────────────────────┘  │
│                                           │
│  ┌────────────────────────────────────┐  │
│  │     Celery Workers                 │  │
│  │  - Video Processing                │  │
│  │  - Feature Extraction              │  │
│  │  - Identity Resolution             │  │
│  └────────────────────────────────────┘  │
│                                           │
│  ┌────────────────────────────────────┐  │
│  │     FAISS Index                    │  │
│  │  - Embedding Storage               │  │
│  │  - Similarity Search               │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

### Storage Requirements (Estimates)

For 1000 identities, 100,000 appearances:
- **Neo4j Graph Data**: ~500 MB - 1 GB
- **Face Embeddings** (512-dim × 100K): ~200 MB
- **ReID Embeddings** (2048-dim × 100K): ~800 MB
- **Video Frames** (if stored): ~10-100 GB (depends on retention)
- **Total**: ~2-3 GB (without video frames)

---

## Success Metrics

### Technical Metrics
- **Identity Resolution Accuracy**: % correctly matched identities
- **False Positive Rate**: % incorrectly merged identities
- **Processing Speed**: Frames per second throughput
- **Query Latency**: Response time for identity lookups
- **Confidence Distribution**: Distribution of confidence scores

### Business/Research Metrics
- **Coverage**: % of detections successfully assigned to identities
- **Re-identification Rate**: % successfully re-identified after absence
- **Temporal Persistence**: Average duration of tracked identities
- **Multi-view Coverage**: % identities seen across multiple cameras

---

## Next Steps

See [TODO.md](./TODO.md) for detailed development roadmap and [PRD documents](./docs/PRDs/) for component specifications.

### Immediate Actions
1. Set up development environment
2. Install and configure Neo4j
3. Implement basic video processing pipeline
4. Integrate YOLOv8 for detection
5. Extract face embeddings with InsightFace
6. Build initial graph schema in Neo4j

---

## References

### Research Papers
- "Deep SORT: Simple Online and Realtime Tracking with a Deep Association Metric" (2017)
- "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (2019)
- "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (2021)
- "FastReID: A Pytorch Toolbox for General Instance Re-identification" (2020)

### Key Technologies
- Neo4j: https://neo4j.com/
- InsightFace: https://github.com/deepinsight/insightface
- FastReID: https://github.com/JDAI-CV/fast-reid
- YOLOv8: https://github.com/ultralytics/ultralytics
- ByteTrack: https://github.com/ifzhang/ByteTrack
- FAISS: https://github.com/facebookresearch/faiss
- Milvus: https://milvus.io/

### Documentation
- Neo4j Python Driver: https://neo4j.com/docs/python-manual/current/
- Graph Data Science Library: https://neo4j.com/docs/graph-data-science/current/
- OpenCV Python: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
