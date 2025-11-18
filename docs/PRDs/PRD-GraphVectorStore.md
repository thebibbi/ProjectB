# PRD: Graph Database & Vector Store Layer

**Version**: 1.0
**Date**: 2025-11-18
**Status**: Draft
**Owner**: Core Team

---

## Executive Summary

The Graph Database & Vector Store Layer provides persistent storage and efficient querying for the identity tracking system. It combines Neo4j (graph database) for relationship modeling with FAISS (vector database) for embedding similarity search, creating a hybrid storage architecture optimized for multi-view identity tracking.

---

## Problem Statement

Identity tracking systems require two fundamentally different storage and query patterns:

1. **Relationship Queries**: "Show all appearances of identity X", "Find path between cameras A and B", "Which identities appeared together?"
2. **Similarity Queries**: "Find the 10 most similar faces to this embedding", "Search for people with similar appearance"

Traditional databases cannot efficiently handle both:
- **Relational databases** struggle with graph traversals and similarity search
- **Pure graph databases** lack efficient high-dimensional vector similarity
- **Vector databases** cannot model complex relationships

**Solution**: Hybrid architecture combining graph database (relationships) with vector database (similarity).

---

## Goals & Objectives

### Primary Goals
1. **Efficient Graph Operations**: Fast graph traversals and pattern matching
2. **Fast Similarity Search**: Sub-50ms k-NN search on millions of embeddings
3. **Consistent Data Model**: Single source of truth with strong consistency
4. **Scalable Storage**: Handle 100K+ appearances, 10K+ identities
5. **Developer-Friendly**: Intuitive APIs for common operations

### Success Metrics
- **Query Latency**:
  - Graph queries: <100ms for typical queries
  - Vector search: <50ms for k=10 nearest neighbors
  - Combined queries: <200ms
- **Write Throughput**: >100 appearances/second
- **Storage Efficiency**: <5KB per appearance (excluding image data)
- **Data Integrity**: Zero data loss, strong ACID guarantees

### Non-Goals
- Distributed/sharded deployment (single node sufficient for MVP)
- Multi-region replication
- Real-time streaming updates (batch processing focus)
- Built-in ML model serving

---

## Architecture Overview

### Hybrid Storage Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Application Layer                       │
└───────────────┬────────────────────┬────────────────────┘
                │                    │
                ▼                    ▼
┌───────────────────────┐  ┌──────────────────────────┐
│    Neo4j Graph DB     │  │   FAISS Vector Store     │
│                       │  │                          │
│  ┌─────────────────┐ │  │  ┌────────────────────┐  │
│  │   Identities    │ │  │  │  Face Embeddings   │  │
│  │   Appearances   │ │  │  │  (512-dim)         │  │
│  │   Cameras       │ │  │  │                    │  │
│  │   Relationships │ │  │  │  ReID Embeddings   │  │
│  └─────────────────┘ │  │  │  (2048-dim)        │  │
│                       │  │  └────────────────────┘  │
│  Graph Algorithms     │  │  Similarity Search       │
│  Cypher Queries       │  │  Index Management        │
└───────────────────────┘  └──────────────────────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
              ┌───────────────┐
              │  Sync Layer   │
              │  (Consistency)│
              └───────────────┘
```

### Data Flow

1. **Write Path**:
   - Application creates Appearance object
   - Extract face/ReID embeddings
   - Insert embeddings into FAISS (get index IDs)
   - Insert appearance node in Neo4j (reference FAISS IDs)
   - Create relationships (identity, camera, temporal)

2. **Query Path**:
   - **Graph Query**: Query Neo4j directly
   - **Similarity Query**: Query FAISS, then fetch from Neo4j
   - **Hybrid Query**: Query FAISS for candidates, filter/traverse in Neo4j

---

## Component 1: Neo4j Graph Database

### Purpose
Store identities, appearances, cameras, and their relationships in a graph structure optimized for traversal and pattern matching.

### Graph Schema

#### Node Types

##### Identity Node
```cypher
(:Identity {
  id: STRING,              // UUID primary key
  created_at: DATETIME,
  updated_at: DATETIME,
  first_seen: DATETIME,
  last_seen: DATETIME,
  num_appearances: INTEGER,
  num_cameras: INTEGER,
  confidence_score: FLOAT,
  has_face: BOOLEAN,
  cluster_method: STRING,
  merge_history: [STRING]  // Array of merged identity IDs
})
```

##### Appearance Node
```cypher
(:Appearance {
  id: STRING,              // UUID primary key
  timestamp: DATETIME,
  camera_id: STRING,
  tracking_id: INTEGER,

  // Bounding box
  bbox_x: INTEGER,
  bbox_y: INTEGER,
  bbox_w: INTEGER,
  bbox_h: INTEGER,

  // Embeddings (references to FAISS)
  face_embedding_idx: INTEGER,  // FAISS index, -1 if no face
  reid_embedding_idx: INTEGER,  // FAISS index

  // Quality metrics
  detection_confidence: FLOAT,
  face_quality: FLOAT,
  reid_quality: FLOAT,
  blur_score: FLOAT,

  // Metadata
  frame_path: STRING,
  match_confidence: FLOAT
})
```

##### Camera Node
```cypher
(:Camera {
  id: STRING,              // Camera identifier
  name: STRING,
  location: STRING,
  position_x: FLOAT,
  position_y: FLOAT,
  position_z: FLOAT,
  orientation: STRING,
  fov: FLOAT,              // Field of view
  created_at: DATETIME
})
```

#### Relationship Types

##### HAS_APPEARANCE
```cypher
(:Identity)-[:HAS_APPEARANCE {
  confidence: FLOAT,       // Confidence in this association
  method: STRING,          // 'deterministic' or 'probabilistic'
  created_at: DATETIME
}]->(:Appearance)
```

##### SAME_TRACK
```cypher
(:Appearance)-[:SAME_TRACK {
  method: 'deterministic',
  tracking_id: INTEGER
}]->(:Appearance)
```

##### PROBABLE_MATCH
```cypher
(:Appearance)-[:PROBABLE_MATCH {
  similarity: FLOAT,       // Combined similarity score
  face_similarity: FLOAT,
  reid_similarity: FLOAT,
  method: STRING,          // 'face', 'reid', 'hybrid'
  temporal_gap: INTEGER,   // Seconds
  created_at: DATETIME
}]->(:Appearance)
```

##### CAPTURED_BY
```cypher
(:Appearance)-[:CAPTURED_BY {
  timestamp: DATETIME
}]->(:Camera)
```

##### NEXT
```cypher
(:Appearance)-[:NEXT {
  time_delta: INTEGER,     // Milliseconds
  same_camera: BOOLEAN
}]->(:Appearance)
```

##### ADJACENT_TO
```cypher
(:Camera)-[:ADJACENT_TO {
  distance: FLOAT,         // Meters
  avg_transition_time: INTEGER  // Seconds
}]->(:Camera)
```

### Indexes & Constraints

```cypher
// Unique constraints
CREATE CONSTRAINT identity_id_unique IF NOT EXISTS
FOR (i:Identity) REQUIRE i.id IS UNIQUE;

CREATE CONSTRAINT appearance_id_unique IF NOT EXISTS
FOR (a:Appearance) REQUIRE a.id IS UNIQUE;

CREATE CONSTRAINT camera_id_unique IF NOT EXISTS
FOR (c:Camera) REQUIRE c.id IS UNIQUE;

// Indexes for common queries
CREATE INDEX appearance_timestamp IF NOT EXISTS
FOR (a:Appearance) ON (a.timestamp);

CREATE INDEX appearance_camera IF NOT EXISTS
FOR (a:Appearance) ON (a.camera_id);

CREATE INDEX appearance_tracking IF NOT EXISTS
FOR (a:Appearance) ON (a.tracking_id);

CREATE INDEX identity_confidence IF NOT EXISTS
FOR (i:Identity) ON (i.confidence_score);

CREATE INDEX identity_first_seen IF NOT EXISTS
FOR (i:Identity) ON (i.first_seen);
```

### Common Cypher Queries

#### Get All Appearances for Identity
```cypher
MATCH (i:Identity {id: $identity_id})-[:HAS_APPEARANCE]->(a:Appearance)
RETURN a
ORDER BY a.timestamp
```

#### Get Identity Trajectory
```cypher
MATCH (i:Identity {id: $identity_id})-[:HAS_APPEARANCE]->(a:Appearance)-[:CAPTURED_BY]->(c:Camera)
RETURN a.timestamp, c.name, c.location, a.bbox_x, a.bbox_y
ORDER BY a.timestamp
```

#### Find Co-Occurring Identities
```cypher
MATCH (i1:Identity)-[:HAS_APPEARANCE]->(a1:Appearance)-[:CAPTURED_BY]->(c:Camera),
      (i2:Identity)-[:HAS_APPEARANCE]->(a2:Appearance)-[:CAPTURED_BY]->(c)
WHERE i1.id < i2.id
  AND abs(duration.between(a1.timestamp, a2.timestamp).seconds) < 60
RETURN i1.id, i2.id, c.name, count(*) as co_occurrences
ORDER BY co_occurrences DESC
```

#### Find Path Between Cameras
```cypher
MATCH path = shortestPath(
  (c1:Camera {id: $camera1_id})-[:ADJACENT_TO*]-(c2:Camera {id: $camera2_id})
)
RETURN path
```

---

## Component 2: FAISS Vector Store

### Purpose
Store high-dimensional embeddings and perform fast k-nearest neighbor similarity search.

### Index Structure

#### Face Embedding Index
- **Dimensions**: 512 (from InsightFace ArcFace)
- **Index Type**: `IndexFlatIP` (inner product) or `IndexFlatL2` (L2 distance)
- **Metric**: Cosine similarity (via inner product on normalized vectors)
- **Size Estimate**: 512 dim × 4 bytes × 100K = ~200 MB

#### ReID Embedding Index
- **Dimensions**: 2048 (from FastReID)
- **Index Type**: `IndexFlatIP` or `IndexFlatL2`
- **Metric**: Cosine similarity
- **Size Estimate**: 2048 dim × 4 bytes × 100K = ~800 MB

### Index Selection Strategy

**Phase 1 (MVP)**: Flat indexes for exact search
- `IndexFlatIP` for cosine similarity
- Brute-force search, guaranteed exact results
- Good for <1M vectors

**Phase 2**: Approximate indexes for scale
- `IndexIVFFlat` for clustering-based search
- `IndexHNSW` for graph-based search
- Trade accuracy for speed (>1M vectors)

### Metadata Management

Since FAISS only stores vectors, we need external metadata storage:

```python
# appearance_id -> FAISS index mapping
face_index_map = {
    "app_abc123": 0,      # FAISS index 0
    "app_def456": 1,
    # ...
}

reid_index_map = {
    "app_abc123": 0,
    "app_def456": 1,
    # ...
}

# Reverse mapping
index_to_appearance = {
    0: "app_abc123",
    1: "app_def456",
    # ...
}
```

Store in:
- **Option 1**: Python dict + pickle (simple, local)
- **Option 2**: Redis (fast, network accessible)
- **Option 3**: SQLite (persistent, queryable)

**Recommendation**: Start with pickle, migrate to Redis if needed.

### FAISS Operations

#### Initialize Index
```python
import faiss
import numpy as np

# Face embeddings
face_dim = 512
face_index = faiss.IndexFlatIP(face_dim)  # Inner product (cosine sim)

# ReID embeddings
reid_dim = 2048
reid_index = faiss.IndexFlatIP(reid_dim)
```

#### Add Embeddings
```python
def add_embedding(index, embedding, appearance_id, index_map):
    # Normalize for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)

    # Add to FAISS
    faiss_idx = index.ntotal
    index.add(np.array([embedding]))

    # Update mapping
    index_map[appearance_id] = faiss_idx

    return faiss_idx
```

#### Search Similar
```python
def search_similar(index, query_embedding, k=10, min_similarity=0.6):
    # Normalize query
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search
    similarities, indices = index.search(
        np.array([query_embedding]), k
    )

    # Filter by threshold
    results = [
        (int(idx), float(sim))
        for idx, sim in zip(indices[0], similarities[0])
        if sim >= min_similarity
    ]

    return results
```

#### Save/Load Index
```python
# Save
faiss.write_index(face_index, "face_embeddings.index")
faiss.write_index(reid_index, "reid_embeddings.index")

# Save mappings
import pickle
with open("index_mappings.pkl", "wb") as f:
    pickle.dump({
        "face_index_map": face_index_map,
        "reid_index_map": reid_index_map,
        "index_to_appearance": index_to_appearance
    }, f)

# Load
face_index = faiss.read_index("face_embeddings.index")
reid_index = faiss.read_index("reid_embeddings.index")

with open("index_mappings.pkl", "rb") as f:
    mappings = pickle.load(f)
```

---

## Data Models & Python Interface

### Storage Manager Interface

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

class StorageManager(ABC):
    """Abstract interface for storage operations"""

    @abstractmethod
    def create_identity(self, identity: Identity) -> str:
        """Create new identity, return ID"""
        pass

    @abstractmethod
    def create_appearance(self, appearance: Appearance) -> str:
        """Create appearance, return ID"""
        pass

    @abstractmethod
    def link_appearance_to_identity(
        self,
        appearance_id: str,
        identity_id: str,
        confidence: float,
        method: str
    ) -> None:
        """Create HAS_APPEARANCE relationship"""
        pass

    @abstractmethod
    def find_similar_faces(
        self,
        embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.6
    ) -> List[tuple[str, float]]:
        """Find similar face embeddings, return appearance IDs"""
        pass

    @abstractmethod
    def get_identity(self, identity_id: str) -> Optional[Identity]:
        """Retrieve identity by ID"""
        pass

    @abstractmethod
    def get_identity_appearances(
        self,
        identity_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Appearance]:
        """Get all appearances for identity"""
        pass
```

### Neo4j Implementation

```python
from neo4j import GraphDatabase

class Neo4jStorageManager(StorageManager):
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_identity(self, identity: Identity) -> str:
        query = """
        CREATE (i:Identity {
            id: $id,
            created_at: datetime($created_at),
            confidence_score: $confidence_score,
            has_face: $has_face,
            cluster_method: $cluster_method
        })
        RETURN i.id as id
        """

        with self.driver.session() as session:
            result = session.run(query, **identity.dict())
            return result.single()["id"]

    def create_appearance(self, appearance: Appearance) -> str:
        query = """
        CREATE (a:Appearance {
            id: $id,
            timestamp: datetime($timestamp),
            camera_id: $camera_id,
            tracking_id: $tracking_id,
            bbox_x: $bbox[0],
            bbox_y: $bbox[1],
            bbox_w: $bbox[2],
            bbox_h: $bbox[3],
            face_embedding_idx: $face_embedding_idx,
            reid_embedding_idx: $reid_embedding_idx,
            detection_confidence: $detection_confidence,
            face_quality: $face_quality,
            reid_quality: $reid_quality,
            frame_path: $frame_path,
            match_confidence: $match_confidence
        })
        RETURN a.id as id
        """

        with self.driver.session() as session:
            result = session.run(query, **appearance.dict())
            return result.single()["id"]

    def link_appearance_to_identity(
        self,
        appearance_id: str,
        identity_id: str,
        confidence: float,
        method: str
    ) -> None:
        query = """
        MATCH (i:Identity {id: $identity_id})
        MATCH (a:Appearance {id: $appearance_id})
        MERGE (i)-[:HAS_APPEARANCE {
            confidence: $confidence,
            method: $method,
            created_at: datetime()
        }]->(a)

        // Update identity stats
        SET i.updated_at = datetime()
        SET i.num_appearances = i.num_appearances + 1
        """

        with self.driver.session() as session:
            session.run(
                query,
                identity_id=identity_id,
                appearance_id=appearance_id,
                confidence=confidence,
                method=method
            )
```

### Hybrid Storage Manager

```python
class HybridStorageManager:
    """Combines Neo4j + FAISS"""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        faiss_index_path: str
    ):
        self.graph = Neo4jStorageManager(neo4j_uri, neo4j_user, neo4j_password)
        self.vector = FAISSVectorStore(faiss_index_path)

    def store_appearance_with_embeddings(
        self,
        appearance: Appearance,
        face_embedding: Optional[np.ndarray],
        reid_embedding: np.ndarray
    ) -> str:
        # Store embeddings in FAISS
        if face_embedding is not None:
            face_idx = self.vector.add_face_embedding(
                face_embedding, appearance.id
            )
            appearance.face_embedding_idx = face_idx
        else:
            appearance.face_embedding_idx = -1

        reid_idx = self.vector.add_reid_embedding(
            reid_embedding, appearance.id
        )
        appearance.reid_embedding_idx = reid_idx

        # Store appearance in Neo4j
        appearance_id = self.graph.create_appearance(appearance)

        return appearance_id

    def find_matching_identities(
        self,
        face_embedding: Optional[np.ndarray],
        reid_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.6
    ) -> List[tuple[str, float]]:
        # Find similar appearances using FAISS
        face_matches = []
        if face_embedding is not None:
            face_matches = self.vector.search_faces(
                face_embedding, k, min_similarity
            )

        reid_matches = self.vector.search_reid(
            reid_embedding, k, min_similarity
        )

        # Combine results
        candidate_appearances = set(
            [m[0] for m in face_matches] + [m[0] for m in reid_matches]
        )

        # Get identities for these appearances from Neo4j
        identities = self.graph.get_identities_for_appearances(
            list(candidate_appearances)
        )

        return identities
```

---

## Configuration

### Neo4j Configuration

```yaml
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "${NEO4J_PASSWORD}"
  database: "neo4j"

  connection:
    max_pool_size: 50
    connection_timeout: 30
    max_transaction_retry_time: 30

  memory:
    heap_initial_size: "1G"
    heap_max_size: "2G"
    page_cache_size: "512M"
```

### FAISS Configuration

```yaml
faiss:
  indexes_dir: "./data/indexes"

  face:
    dimensions: 512
    index_type: "IndexFlatIP"
    metric: "cosine"

  reid:
    dimensions: 2048
    index_type: "IndexFlatIP"
    metric: "cosine"

  search:
    default_k: 10
    min_similarity: 0.6

  persistence:
    auto_save_interval: 300  # seconds
    backup_enabled: true
```

---

## Performance Optimization

### Neo4j Optimization

1. **Indexing Strategy**
   - Create indexes on frequently queried properties
   - Use composite indexes for multi-property queries
   - Monitor query plans with `EXPLAIN` and `PROFILE`

2. **Query Optimization**
   - Use parameters instead of string concatenation
   - Limit result sets with `LIMIT`
   - Use `WITH` for query pipelining
   - Avoid Cartesian products

3. **Memory Configuration**
   - Increase heap size for large graphs
   - Tune page cache for node/relationship storage
   - Monitor GC pauses

### FAISS Optimization

1. **Index Selection**
   - Flat indexes: exact search, <1M vectors
   - IVF indexes: 10x speedup, >1M vectors
   - HNSW indexes: best quality/speed tradeoff

2. **Search Optimization**
   - Pre-normalize embeddings for cosine similarity
   - Use GPU for large-scale search (optional)
   - Batch queries when possible

3. **Memory Management**
   - Memory-map large indexes
   - Use `IndexBinaryFlat` for binary embeddings
   - Compress embeddings with PQ/OPQ

---

## Testing Strategy

### Unit Tests

```python
def test_create_identity():
    storage = HybridStorageManager(...)
    identity = Identity(id="test_id", ...)
    id = storage.create_identity(identity)
    assert id == "test_id"

    retrieved = storage.get_identity(id)
    assert retrieved.id == identity.id

def test_vector_similarity_search():
    vector_store = FAISSVectorStore(...)

    # Add known embedding
    emb1 = np.random.rand(512)
    vector_store.add_face_embedding(emb1, "app_1")

    # Search with same embedding
    results = vector_store.search_faces(emb1, k=1)
    assert len(results) == 1
    assert results[0][0] == "app_1"
    assert results[0][1] > 0.99  # High similarity
```

### Integration Tests

```python
def test_end_to_end_appearance_storage():
    storage = HybridStorageManager(...)

    # Create appearance with embeddings
    appearance = Appearance(...)
    face_emb = np.random.rand(512)
    reid_emb = np.random.rand(2048)

    appearance_id = storage.store_appearance_with_embeddings(
        appearance, face_emb, reid_emb
    )

    # Verify stored
    retrieved = storage.get_appearance(appearance_id)
    assert retrieved.id == appearance_id

    # Verify embeddings searchable
    matches = storage.find_matching_identities(
        face_emb, reid_emb, k=5
    )
    assert len(matches) > 0
```

### Performance Tests

```python
import time

def test_query_latency():
    storage = HybridStorageManager(...)

    # Graph query latency
    start = time.time()
    identity = storage.get_identity("test_id")
    graph_latency = time.time() - start
    assert graph_latency < 0.1  # <100ms

    # Vector search latency
    query_emb = np.random.rand(512)
    start = time.time()
    results = storage.vector.search_faces(query_emb, k=10)
    vector_latency = time.time() - start
    assert vector_latency < 0.05  # <50ms
```

---

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.13-community
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/your_password
      - NEO4J_server_memory_heap_initial__size=1G
      - NEO4J_server_memory_heap_max__size=2G
      - NEO4J_server_memory_pagecache_size=512M
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
```

### Initialization Script

```python
# scripts/setup_neo4j.py

from neo4j import GraphDatabase

def setup_schema(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Constraints
        session.run("""
            CREATE CONSTRAINT identity_id_unique IF NOT EXISTS
            FOR (i:Identity) REQUIRE i.id IS UNIQUE
        """)

        session.run("""
            CREATE CONSTRAINT appearance_id_unique IF NOT EXISTS
            FOR (a:Appearance) REQUIRE a.id IS UNIQUE
        """)

        # Indexes
        session.run("""
            CREATE INDEX appearance_timestamp IF NOT EXISTS
            FOR (a:Appearance) ON (a.timestamp)
        """)

        print("✓ Neo4j schema initialized")

    driver.close()

if __name__ == "__main__":
    setup_schema(
        "bolt://localhost:7687",
        "neo4j",
        "your_password"
    )
```

---

## Monitoring & Maintenance

### Monitoring Metrics

1. **Neo4j Metrics**
   - Query execution time
   - Transactions per second
   - Cache hit ratio
   - Heap usage, GC pauses

2. **FAISS Metrics**
   - Index size (RAM usage)
   - Search latency (p50, p95, p99)
   - Throughput (queries per second)

3. **Data Metrics**
   - Number of identities
   - Number of appearances
   - Average appearances per identity
   - Storage size (disk usage)

### Backup Strategy

```bash
# Neo4j backup
neo4j-admin database dump neo4j --to=/backups/neo4j-$(date +%Y%m%d).dump

# FAISS backup
cp -r data/indexes data/indexes.backup.$(date +%Y%m%d)

# Automated daily backup
0 2 * * * /scripts/backup.sh
```

---

## Migration Path

### Phase 1 → Phase 2: Scale FAISS

When index size exceeds 1M vectors:

```python
# Migrate from Flat to IVF index
import faiss

# Load existing flat index
flat_index = faiss.read_index("face_embeddings.index")

# Create IVF index
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatIP(512)
ivf_index = faiss.IndexIVFFlat(quantizer, 512, nlist)

# Train on existing vectors
vectors = faiss.index_to_array(flat_index)
ivf_index.train(vectors)

# Add vectors
ivf_index.add(vectors)

# Save new index
faiss.write_index(ivf_index, "face_embeddings_ivf.index")
```

### Phase 2 → Phase 3: Migrate to Milvus

If FAISS becomes limiting:

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
]
schema = CollectionSchema(fields)

# Create collection
collection = Collection(name="face_embeddings", schema=schema)

# Migrate data from FAISS
# ... migration logic ...
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Neo4j memory exhaustion | High | Monitor memory, tune heap/cache, add indices |
| FAISS index too large for RAM | High | Use memory-mapped indexes, migrate to Milvus |
| Slow graph queries | Medium | Add indexes, optimize Cypher, use query profiling |
| Data corruption | High | Regular backups, transaction logging, ACID guarantees |
| Index-graph desync | Medium | Transactional writes, consistency checks, repair tools |

---

## Future Enhancements

- Distributed Neo4j (enterprise) for horizontal scaling
- GPU-accelerated FAISS search
- Milvus integration for production scale
- Automated index optimization and rebalancing
- Multi-tenancy support
- Time-based data partitioning and archival

---

**Document Version History**:
- v1.0 (2025-11-18): Initial draft
