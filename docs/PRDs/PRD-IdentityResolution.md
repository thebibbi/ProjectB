# PRD: Identity Resolution Engine

**Version**: 1.0
**Date**: 2025-11-18
**Status**: Draft
**Owner**: Core Team

---

## Executive Summary

The Identity Resolution Engine is the core component of ProjectB that performs multi-view identity tracking across video streams. It combines deterministic and probabilistic matching methods to associate visual appearances of individuals across time, space, and camera views, creating a unified identity graph.

---

## Problem Statement

In multi-camera computer vision systems, the same person generates multiple detections across different:
- **Cameras** (different viewpoints, lighting conditions)
- **Time periods** (continuous tracking, re-appearance after absence)
- **Occlusions** (temporary disappearances, partial views)
- **Appearance changes** (clothing adjustments, carried items)

**Challenge**: How do we reliably determine that two detections refer to the same individual?

### Current Limitations
- Single-camera tracking cannot follow people across camera boundaries
- Simple matching fails with viewpoint/lighting changes
- Manual review doesn't scale to thousands of detections
- Existing solutions are either too rigid (deterministic only) or too error-prone (probabilistic only)

---

## Goals & Objectives

### Primary Goals
1. **Accurate Identity Matching**: Achieve >80% accuracy in cross-camera identity resolution
2. **Robust to Variations**: Handle viewpoint, lighting, and appearance changes
3. **Scalable Processing**: Process batches of videos efficiently
4. **Transparent Confidence**: Provide confidence scores for all matches
5. **Hybrid Approach**: Combine deterministic and probabilistic methods

### Success Metrics
- **Identity Resolution Accuracy**: >80% correct matches on test dataset
- **False Positive Rate**: <10% incorrect merges
- **False Negative Rate**: <15% missed matches
- **Processing Throughput**: >5 identities per second
- **Confidence Calibration**: Confidence scores correlate with actual accuracy

### Non-Goals (Out of Scope)
- Real-time streaming (Phase 1-3, deferred to Phase 4)
- Privacy-preserving matching (not required for personal use)
- Multi-person pose estimation
- Action recognition or behavior analysis

---

## User Stories

### US-1: Single Video Identity Clustering
**As a** user
**I want** to process a single video file
**So that** I can identify all unique individuals appearing in the video

**Acceptance Criteria**:
- Video is processed frame by frame
- People are detected and tracked within the video
- Face embeddings are extracted for each detection
- Detections are clustered into unique identities
- Each identity has a unique ID and list of appearances

---

### US-2: Cross-Camera Identity Matching
**As a** user
**I want** to match identities across multiple camera feeds
**So that** I can track people as they move between camera coverage areas

**Acceptance Criteria**:
- Videos from different cameras are processed
- Face and ReID embeddings are compared across cameras
- High-similarity matches are linked with confidence scores
- Identities are merged when confidence exceeds threshold
- Manual review is available for ambiguous cases

---

### US-3: Re-identification After Absence
**As a** user
**I want** to re-identify people who reappear after being absent
**So that** I can maintain persistent identities over time

**Acceptance Criteria**:
- System searches for matches after temporal gaps
- Confidence degrades gradually with gap duration
- Re-appearance is detected and linked to original identity
- Long-term identity persistence is maintained

---

### US-4: Confidence-Based Filtering
**As a** user
**I want** to filter identities by confidence score
**So that** I can focus on high-quality identities and review uncertain ones

**Acceptance Criteria**:
- Each identity has a confidence score (0-1)
- Scores reflect quality and consistency of evidence
- API allows filtering by minimum confidence
- Low-confidence identities are flagged for review

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Identity Resolution Engine                 │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐    │
│  │   Feature    │  │   Matching   │  │  Clustering   │    │
│  │  Extraction  │→ │    Engine    │→ │    Engine     │    │
│  └──────────────┘  └──────────────┘  └───────────────┘    │
│         ↓                  ↓                   ↓            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐    │
│  │   Vector     │  │  Similarity  │  │  Confidence   │    │
│  │    Store     │  │  Calculator  │  │    Scorer     │    │
│  └──────────────┘  └──────────────┘  └───────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Feature Extraction Module

**Purpose**: Extract visual features from person detections

**Inputs**:
- Cropped person images (bounding boxes)
- Camera metadata
- Timestamp information

**Outputs**:
- Face embeddings (512-dim from InsightFace)
- ReID embeddings (2048-dim from FastReID)
- Appearance metadata (bbox, quality scores)

**Key Functions**:
```python
class FeatureExtractor:
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]
    def extract_reid_embedding(self, image: np.ndarray) -> np.ndarray
    def assess_quality(self, image: np.ndarray) -> float
    def extract_appearance_features(self, image: np.ndarray) -> dict
```

**Quality Criteria**:
- Face detection confidence > 0.8
- Face size > 64x64 pixels
- Face angle < 45 degrees
- Blur score < threshold
- Minimum bbox size for ReID

---

#### 2. Matching Engine

**Purpose**: Find potential matches between appearances

**Matching Strategies**:

##### A. Deterministic Matching
High-confidence, rule-based matches:
- **Same tracking ID**: Detections in same video sequence
- **Exact feature match**: Hash collision (rare but 100% confidence)
- **Same camera + short gap**: <5 minutes, same camera

##### B. Probabilistic Matching
ML-based similarity matching:
- **Face similarity**: Cosine similarity on face embeddings
- **ReID similarity**: Cosine similarity on body embeddings
- **Temporal proximity**: Gaussian decay with time
- **Spatial feasibility**: Camera transition possibility

**Matching Pipeline**:
```python
class MatchingEngine:
    def find_candidates(
        self,
        appearance: Appearance,
        search_scope: SearchScope
    ) -> List[MatchCandidate]

    def calculate_similarity(
        self,
        appearance1: Appearance,
        appearance2: Appearance
    ) -> MatchScore

    def apply_constraints(
        self,
        candidates: List[MatchCandidate]
    ) -> List[MatchCandidate]
```

**Similarity Scoring**:
```python
def hybrid_similarity(app1, app2):
    # Weighted combination
    face_sim = cosine_similarity(app1.face_emb, app2.face_emb)
    reid_sim = cosine_similarity(app1.reid_emb, app2.reid_emb)

    # Time decay
    time_delta = abs(app1.timestamp - app2.timestamp)
    temporal_factor = exp(-time_delta / DECAY_CONSTANT)

    # Spatial feasibility
    spatial_factor = is_feasible_transition(
        app1.camera, app2.camera, time_delta
    )

    # Combined score
    score = (
        FACE_WEIGHT * face_sim +
        REID_WEIGHT * reid_sim
    ) * temporal_factor * spatial_factor

    return score
```

**Default Weights**:
- `FACE_WEIGHT = 0.6` (face more reliable when available)
- `REID_WEIGHT = 0.4`
- `MIN_FACE_SIMILARITY = 0.60`
- `MIN_REID_SIMILARITY = 0.50`
- `MIN_HYBRID_SIMILARITY = 0.55`

---

#### 3. Clustering Engine

**Purpose**: Group appearances into identity clusters

**Algorithms**:

##### Phase 1: Simple Threshold Clustering
- Create identity for first appearance
- For each new appearance:
  - Find best matching existing identity
  - If similarity > threshold, add to that identity
  - Else, create new identity

##### Phase 2: Advanced Clustering
- **DBSCAN**: Density-based clustering in embedding space
- **Hierarchical Agglomerative Clustering**: Bottom-up merging
- **Community Detection**: On similarity graph

**Operations**:
```python
class ClusteringEngine:
    def create_identity(self, appearance: Appearance) -> Identity

    def add_to_identity(
        self,
        identity: Identity,
        appearance: Appearance,
        confidence: float
    ) -> None

    def merge_identities(
        self,
        identity1: Identity,
        identity2: Identity,
        reason: str
    ) -> Identity

    def split_identity(
        self,
        identity: Identity,
        split_point: Appearance
    ) -> Tuple[Identity, Identity]
```

---

#### 4. Confidence Scoring System

**Purpose**: Assign confidence scores to identities and matches

**Factors**:

1. **Observation Count** (30%)
   - More appearances = higher confidence
   - Formula: `min(1.0, num_appearances / 10)`

2. **Feature Quality** (25%)
   - Average quality of face/ReID features
   - Based on detection confidence, size, blur, etc.

3. **Matching Consistency** (25%)
   - How consistent are match scores within identity?
   - Low variance = high confidence

4. **Temporal Coverage** (10%)
   - Duration over which identity is observed
   - Formula: `min(1.0, duration_hours / 24)`

5. **Spatial Diversity** (10%)
   - Number of different cameras
   - Formula: `min(1.0, num_cameras / 5)`

**Calculation**:
```python
def calculate_confidence(identity: Identity) -> float:
    observation_score = min(1.0, len(identity.appearances) / 10)
    quality_score = mean([app.quality for app in identity.appearances])

    match_scores = [app.match_confidence for app in identity.appearances]
    consistency_score = 1.0 - std(match_scores)

    duration = identity.last_seen - identity.first_seen
    temporal_score = min(1.0, duration.total_seconds() / 86400)

    unique_cameras = len(set(app.camera_id for app in identity.appearances))
    spatial_score = min(1.0, unique_cameras / 5)

    confidence = (
        0.30 * observation_score +
        0.25 * quality_score +
        0.25 * consistency_score +
        0.10 * temporal_score +
        0.10 * spatial_score
    )

    return confidence
```

---

## Data Models

### Appearance Model
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Appearance(BaseModel):
    id: str                           # Unique appearance ID
    timestamp: datetime               # When detected
    camera_id: str                    # Which camera
    tracking_id: int                  # Within-video tracking ID

    # Bounding box
    bbox: tuple[int, int, int, int]   # (x, y, w, h)

    # Embeddings (references to vector store)
    face_embedding_id: Optional[str]
    reid_embedding_id: str

    # Quality metrics
    detection_confidence: float       # Detector confidence
    face_quality: Optional[float]     # Face quality score
    reid_quality: float               # ReID quality score
    blur_score: float                 # Image sharpness

    # Metadata
    frame_path: str                   # Path to extracted frame
    identity_id: Optional[str]        # Assigned identity (initially None)
    match_confidence: float = 1.0     # Confidence in identity assignment
```

### Identity Model
```python
class Identity(BaseModel):
    id: str                           # Unique identity ID
    created_at: datetime
    updated_at: datetime

    # Temporal info
    first_seen: datetime
    last_seen: datetime

    # Statistics
    num_appearances: int
    num_cameras: int

    # Quality metrics
    confidence_score: float           # Overall identity confidence
    has_face: bool                    # Has face embeddings

    # Cluster info
    cluster_method: str               # How was this identity created
    merge_history: list[str]          # IDs of merged identities
```

### Match Model
```python
class MatchScore(BaseModel):
    appearance1_id: str
    appearance2_id: str

    # Similarity scores
    face_similarity: Optional[float]
    reid_similarity: float
    combined_similarity: float

    # Context factors
    temporal_factor: float
    spatial_factor: float

    # Final score and decision
    final_score: float
    is_match: bool
    method: str                       # 'deterministic' or 'probabilistic'
```

---

## API Specification

### Core Endpoints

#### Process Detection
```python
POST /api/v1/process/detection

Request:
{
  "image": "base64_encoded_image",
  "timestamp": "2025-11-18T10:30:00Z",
  "camera_id": "cam_001",
  "tracking_id": 123,
  "bbox": [100, 200, 150, 400]
}

Response:
{
  "appearance_id": "app_abc123",
  "identity_id": "id_xyz789",
  "match_confidence": 0.87,
  "method": "probabilistic",
  "face_detected": true,
  "quality_score": 0.92
}
```

#### Find Matches
```python
GET /api/v1/matches/{appearance_id}?min_similarity=0.6

Response:
{
  "appearance_id": "app_abc123",
  "matches": [
    {
      "appearance_id": "app_def456",
      "similarity": 0.85,
      "method": "face",
      "temporal_gap_seconds": 120,
      "same_camera": false
    }
  ]
}
```

#### Merge Identities
```python
POST /api/v1/identities/merge

Request:
{
  "identity_id_1": "id_xyz789",
  "identity_id_2": "id_abc456",
  "reason": "high_face_similarity",
  "confidence": 0.92
}

Response:
{
  "merged_identity_id": "id_xyz789",
  "num_appearances": 47,
  "confidence_score": 0.88
}
```

---

## Configuration Parameters

### Similarity Thresholds
```yaml
matching:
  face:
    min_similarity: 0.60
    weight: 0.6
  reid:
    min_similarity: 0.50
    weight: 0.4
  hybrid:
    min_similarity: 0.55

temporal:
  decay_constant: 3600  # 1 hour in seconds
  max_gap: 86400        # 24 hours

spatial:
  max_transition_speed: 5.0  # meters/second

quality:
  min_face_size: 64
  max_blur_score: 0.3
  min_detection_confidence: 0.7
```

### Clustering Parameters
```yaml
clustering:
  algorithm: "threshold"  # or "dbscan", "hierarchical"
  initial_threshold: 0.65
  merge_threshold: 0.75
  split_threshold: 0.40

dbscan:
  eps: 0.3
  min_samples: 3
```

---

## Performance Requirements

### Latency
- Feature extraction: <100ms per detection
- Similarity search: <50ms per query
- Identity clustering: <500ms for 1000 appearances

### Throughput
- Process 5+ identities per second
- Handle batches of 1000+ detections efficiently

### Accuracy
- Cross-camera matching: >80% accuracy
- False positive rate: <10%
- False negative rate: <15%

### Scalability
- Support 10,000+ identities
- Support 100,000+ appearances
- Support 10+ cameras

---

## Testing Strategy

### Unit Tests
- Test individual similarity calculations
- Test confidence scoring
- Test clustering algorithms
- Test quality assessment

### Integration Tests
- Test end-to-end appearance processing
- Test identity creation and merging
- Test cross-camera matching

### Accuracy Tests
- Use labeled test dataset
- Calculate precision, recall, F1
- Measure false positive/negative rates
- Validate confidence calibration

### Performance Tests
- Benchmark feature extraction speed
- Benchmark similarity search latency
- Test scalability with large datasets

---

## Test Datasets

### Synthetic Test Set
- Create 5 test videos with known identities
- 3-5 cameras with overlapping coverage
- 10-20 unique individuals
- Ground truth labels for all appearances

### Public Datasets
- **MOT Challenge**: Multi-object tracking benchmarks
- **Market-1501**: Person re-identification dataset
- **DukeMTMC-reID**: Multi-camera tracking dataset
- Custom annotations for identity resolution

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Low face matching accuracy | High | Medium | Add ReID, tune thresholds, use ensemble |
| ReID fails on similar clothing | Medium | High | Add face, use temporal/spatial context |
| Computational cost too high | Medium | Medium | Optimize inference, use GPU, batch processing |
| Threshold tuning difficult | Medium | Medium | Implement adaptive thresholds, active learning |
| Identity drift over time | High | Medium | Periodic re-clustering, confidence decay |

---

## Dependencies

### External Libraries
- **InsightFace**: Face detection and recognition
- **FastReID** or **torchreid**: Person re-identification
- **FAISS**: Vector similarity search
- **NumPy/SciPy**: Numerical computing
- **scikit-learn**: Clustering algorithms

### Internal Components
- Detection & Tracking module
- Graph Store module
- Vector Store module

---

## Future Enhancements

### Phase 2+
- Ensemble matching (multiple face/ReID models)
- Appearance attribute extraction (clothing, accessories)
- Active learning for threshold tuning
- Incremental clustering for real-time

### Advanced Features
- Group/crowd detection and association
- Scene context integration
- Anomaly detection (unusual patterns)
- Privacy-preserving matching techniques

---

## Appendix

### Research References
1. "Deep SORT: Simple Online and Realtime Tracking" (Wojke et al., 2017)
2. "ArcFace: Additive Angular Margin Loss" (Deng et al., 2019)
3. "FastReID: A Pytorch Toolbox for General Instance Re-identification" (He et al., 2020)
4. "Bag of Tricks and A Strong Baseline for Deep Person Re-identification" (Luo et al., 2019)

### Glossary
- **Appearance**: A single detection/sighting of a person in a video frame
- **Identity**: A unique individual, represented by multiple appearances
- **Embedding**: A vector representation of visual features
- **ReID**: Person Re-Identification across cameras
- **Deterministic Match**: Rule-based, high-confidence match
- **Probabilistic Match**: Similarity-based match with uncertainty

---

**Document Version History**:
- v1.0 (2025-11-18): Initial draft
