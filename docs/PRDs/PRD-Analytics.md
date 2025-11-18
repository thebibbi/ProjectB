# PRD: Analytics & Reporting Layer

**Version**: 1.0
**Date**: 2025-11-18
**Status**: Draft
**Owner**: Core Team

---

## Executive Summary

The Analytics & Reporting Layer transforms raw identity tracking data into actionable insights through visualization, metrics, and analysis tools. It provides temporal tracking analysis, trajectory visualization, data quality assessment, and confidence scoring to help users understand identity patterns and system performance.

---

## Problem Statement

Raw identity graph data is difficult to interpret without proper visualization and analytics:
- **Trajectory Understanding**: Where do individuals go? What paths do they take?
- **Pattern Discovery**: Which identities appear together? What are common routes?
- **Quality Assessment**: How reliable is the data? Where are the weak points?
- **System Performance**: Is identity resolution working correctly? What's the accuracy?
- **Manual Review**: Which identities need human verification?

Without proper analytics tools, users cannot:
- Trust the system's outputs
- Identify and fix errors
- Discover meaningful patterns
- Monitor system health

---

## Goals & Objectives

### Primary Goals
1. **Trajectory Visualization**: Display identity movements across cameras over time
2. **Data Quality Metrics**: Provide confidence scores and quality assessments
3. **Graph Analytics**: Discover patterns, communities, and relationships
4. **Performance Monitoring**: Track accuracy, coverage, and system health
5. **Interactive Dashboard**: User-friendly interface for exploration and review

### Success Metrics
- **Dashboard Load Time**: <2 seconds for typical queries
- **Visualization Performance**: Render trajectories for 100+ identities smoothly
- **Metric Accuracy**: Calculated metrics match manual verification >95%
- **User Satisfaction**: Users can complete common tasks in <5 minutes
- **Export Functionality**: All data exportable in standard formats

### Non-Goals
- Real-time streaming dashboard (batch refresh is sufficient)
- Advanced ML model training interface
- Video playback synchronization
- Collaborative annotation tools (Phase 1)

---

## User Stories

### US-1: View Identity Trajectory
**As a** user
**I want** to see the complete trajectory of an identity across cameras
**So that** I can understand their movement patterns

**Acceptance Criteria**:
- Display timeline of all appearances
- Show camera locations on spatial map
- Indicate time gaps and transitions
- Color-code by confidence score
- Export trajectory data (CSV/JSON)

---

### US-2: Assess Data Quality
**As a** user
**I want** to see quality metrics for identities and the overall system
**So that** I can trust the results and identify issues

**Acceptance Criteria**:
- Display per-identity confidence scores
- Show distribution of confidence across all identities
- Identify low-quality detections
- Flag ambiguous matches for review
- Track quality trends over time

---

### US-3: Discover Identity Patterns
**As a** user
**I want** to find identities that appear together or follow similar paths
**So that** I can discover relationships and patterns

**Acceptance Criteria**:
- Find co-occurring identities (same time/place)
- Identify common trajectories
- Detect identity clusters/communities
- Visualize relationship graphs
- Export pattern analysis results

---

### US-4: Monitor System Performance
**As a** user
**I want** to track system accuracy and processing metrics
**So that** I can ensure the system is working correctly

**Acceptance Criteria**:
- Display identity resolution accuracy
- Show processing throughput (FPS, detections/sec)
- Track false positive/negative rates (if ground truth available)
- Monitor resource usage (CPU, memory, storage)
- Generate performance reports

---

### US-5: Review and Correct Identities
**As a** user
**I want** to manually review uncertain identities and make corrections
**So that** I can improve data quality

**Acceptance Criteria**:
- Browse all identities with filtering
- View all appearances for an identity
- Compare two identities side-by-side
- Manually merge incorrect splits
- Manually split incorrect merges
- Track manual review history

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Analytics Layer                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Graph      │  │  Trajectory  │  │   Quality    │     │
│  │  Analytics   │  │   Analysis   │  │  Assessment  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                   │                   │           │
│         └───────────────────┴───────────────────┘           │
│                             ↓                                │
│                    ┌────────────────┐                       │
│                    │  Visualization │                       │
│                    │    Dashboard   │                       │
│                    └────────────────┘                       │
│                             ↓                                │
│                    ┌────────────────┐                       │
│                    │  Export & API  │                       │
│                    └────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 1: Graph Analytics

### Purpose
Apply graph algorithms to discover patterns, communities, and key identities in the identity graph.

### Algorithms

#### 1. Community Detection
**Goal**: Find groups of identities that frequently appear together

**Algorithm**: Louvain Community Detection (via Neo4j GDS)

```cypher
// Create graph projection
CALL gds.graph.project(
  'identity-cooccurrence',
  'Identity',
  {
    CO_OCCURS: {
      type: 'CO_OCCURS',
      orientation: 'UNDIRECTED',
      properties: 'weight'
    }
  }
)

// Run Louvain
CALL gds.louvain.stream('identity-cooccurrence')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).id AS identity_id, communityId
ORDER BY communityId
```

#### 2. Centrality Measures
**Goal**: Identify "important" identities (most connected, most frequent)

**Algorithms**:
- **Degree Centrality**: Number of connections
- **Betweenness Centrality**: Bridge between communities
- **PageRank**: Overall importance

```cypher
// Degree centrality
MATCH (i:Identity)-[:HAS_APPEARANCE]->(a:Appearance)
RETURN i.id, count(a) as num_appearances
ORDER BY num_appearances DESC
LIMIT 10

// PageRank
CALL gds.pageRank.stream('identity-graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).id AS identity_id, score
ORDER BY score DESC
LIMIT 10
```

#### 3. Path Analysis
**Goal**: Find common routes and transition patterns

```cypher
// Find all paths between two cameras
MATCH (i:Identity)-[:HAS_APPEARANCE]->(a1:Appearance)-[:CAPTURED_BY]->(c1:Camera {id: $camera1}),
      (i)-[:HAS_APPEARANCE]->(a2:Appearance)-[:CAPTURED_BY]->(c2:Camera {id: $camera2})
WHERE a1.timestamp < a2.timestamp
WITH i, a1, a2, duration.between(a1.timestamp, a2.timestamp).seconds as transition_time
RETURN i.id,
       avg(transition_time) as avg_transition,
       count(*) as num_transitions
ORDER BY num_transitions DESC
```

### Python Interface

```python
class GraphAnalytics:
    def __init__(self, storage_manager):
        self.storage = storage_manager

    def find_communities(self) -> Dict[str, int]:
        """Find identity communities using Louvain"""
        query = """
        CALL gds.louvain.stream('identity-graph')
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).id AS identity_id, communityId
        """
        results = self.storage.graph.run_query(query)
        return {row['identity_id']: row['communityId'] for row in results}

    def find_cooccurring_identities(
        self,
        identity_id: str,
        time_window: int = 60
    ) -> List[tuple[str, int]]:
        """Find identities that appear near this one in time/space"""
        query = """
        MATCH (i1:Identity {id: $identity_id})-[:HAS_APPEARANCE]->(a1:Appearance)-[:CAPTURED_BY]->(c:Camera),
              (i2:Identity)-[:HAS_APPEARANCE]->(a2:Appearance)-[:CAPTURED_BY]->(c)
        WHERE i1.id <> i2.id
          AND abs(duration.between(a1.timestamp, a2.timestamp).seconds) < $time_window
        RETURN i2.id as identity_id, count(*) as co_occurrences
        ORDER BY co_occurrences DESC
        """
        results = self.storage.graph.run_query(
            query,
            identity_id=identity_id,
            time_window=time_window
        )
        return [(row['identity_id'], row['co_occurrences']) for row in results]

    def calculate_identity_importance(self) -> Dict[str, float]:
        """Calculate importance scores for all identities"""
        # Combination of appearance count, camera diversity, temporal coverage
        query = """
        MATCH (i:Identity)-[:HAS_APPEARANCE]->(a:Appearance)-[:CAPTURED_BY]->(c:Camera)
        WITH i,
             count(DISTINCT a) as num_appearances,
             count(DISTINCT c) as num_cameras,
             duration.between(i.first_seen, i.last_seen).seconds as duration
        RETURN i.id,
               num_appearances,
               num_cameras,
               duration,
               (num_appearances * 0.4 + num_cameras * 0.3 + log(duration+1) * 0.3) as importance
        ORDER BY importance DESC
        """
        results = self.storage.graph.run_query(query)
        return {row['i.id']: row['importance'] for row in results}
```

---

## Component 2: Trajectory Analysis

### Purpose
Analyze and visualize the spatial-temporal paths of identities.

### Trajectory Data Structure

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrajectoryPoint:
    timestamp: datetime
    camera_id: str
    camera_name: str
    camera_location: tuple[float, float]  # (x, y) or (lat, lon)
    appearance_id: str
    confidence: float

@dataclass
class Trajectory:
    identity_id: str
    points: List[TrajectoryPoint]
    start_time: datetime
    end_time: datetime
    total_duration: int  # seconds
    cameras_visited: int
    total_distance: float  # meters
```

### Trajectory Extraction

```python
class TrajectoryAnalyzer:
    def extract_trajectory(self, identity_id: str) -> Trajectory:
        """Extract complete trajectory for identity"""
        query = """
        MATCH (i:Identity {id: $identity_id})-[:HAS_APPEARANCE]->(a:Appearance)-[:CAPTURED_BY]->(c:Camera)
        RETURN a.timestamp as timestamp,
               c.id as camera_id,
               c.name as camera_name,
               c.position_x as x,
               c.position_y as y,
               a.id as appearance_id,
               a.match_confidence as confidence
        ORDER BY a.timestamp
        """

        results = self.storage.graph.run_query(query, identity_id=identity_id)

        points = [
            TrajectoryPoint(
                timestamp=row['timestamp'],
                camera_id=row['camera_id'],
                camera_name=row['camera_name'],
                camera_location=(row['x'], row['y']),
                appearance_id=row['appearance_id'],
                confidence=row['confidence']
            )
            for row in results
        ]

        return Trajectory(
            identity_id=identity_id,
            points=points,
            start_time=points[0].timestamp if points else None,
            end_time=points[-1].timestamp if points else None,
            total_duration=self._calculate_duration(points),
            cameras_visited=len(set(p.camera_id for p in points)),
            total_distance=self._calculate_distance(points)
        )

    def _calculate_duration(self, points: List[TrajectoryPoint]) -> int:
        if len(points) < 2:
            return 0
        return (points[-1].timestamp - points[0].timestamp).total_seconds()

    def _calculate_distance(self, points: List[TrajectoryPoint]) -> float:
        """Calculate total distance traveled"""
        distance = 0.0
        for i in range(1, len(points)):
            p1 = points[i-1].camera_location
            p2 = points[i].camera_location
            distance += self._euclidean_distance(p1, p2)
        return distance

    def _euclidean_distance(self, p1: tuple, p2: tuple) -> float:
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
```

### Trajectory Metrics

```python
class TrajectoryMetrics:
    def calculate_dwell_time(self, trajectory: Trajectory) -> Dict[str, int]:
        """Calculate time spent at each camera"""
        dwell_times = {}

        for i, point in enumerate(trajectory.points):
            if i == 0:
                continue

            if point.camera_id == trajectory.points[i-1].camera_id:
                # Same camera, add time
                delta = (point.timestamp - trajectory.points[i-1].timestamp).total_seconds()
                dwell_times[point.camera_id] = dwell_times.get(point.camera_id, 0) + delta

        return dwell_times

    def find_transition_patterns(
        self,
        trajectories: List[Trajectory]
    ) -> Dict[tuple[str, str], int]:
        """Find common camera transitions"""
        transitions = {}

        for traj in trajectories:
            for i in range(1, len(traj.points)):
                from_cam = traj.points[i-1].camera_id
                to_cam = traj.points[i].camera_id

                if from_cam != to_cam:
                    key = (from_cam, to_cam)
                    transitions[key] = transitions.get(key, 0) + 1

        return transitions

    def detect_anomalies(self, trajectory: Trajectory) -> List[str]:
        """Detect unusual patterns in trajectory"""
        anomalies = []

        for i in range(1, len(trajectory.points)):
            prev = trajectory.points[i-1]
            curr = trajectory.points[i]

            # Check for unrealistic speed
            time_delta = (curr.timestamp - prev.timestamp).total_seconds()
            distance = self._euclidean_distance(
                prev.camera_location,
                curr.camera_location
            )
            speed = distance / time_delta if time_delta > 0 else 0

            if speed > 10.0:  # 10 m/s = 36 km/h
                anomalies.append(f"Unrealistic speed: {speed:.2f} m/s between {prev.camera_id} and {curr.camera_id}")

            # Check for long gaps
            if time_delta > 3600:  # 1 hour
                anomalies.append(f"Long gap: {time_delta/60:.0f} minutes")

        return anomalies
```

---

## Component 3: Quality Assessment

### Purpose
Assess data quality and provide confidence scores for identities and matches.

### Quality Metrics

#### 1. Identity Confidence Score
Already defined in Identity Resolution PRD, calculated as:
```python
confidence = (
    0.30 * observation_score +
    0.25 * quality_score +
    0.25 * consistency_score +
    0.10 * temporal_score +
    0.10 * spatial_score
)
```

#### 2. System-Wide Quality Metrics

```python
class QualityAssessor:
    def calculate_system_metrics(self) -> Dict[str, float]:
        """Calculate overall system quality metrics"""

        # Average confidence across all identities
        avg_confidence = self._average_identity_confidence()

        # Coverage: % of appearances assigned to identities
        coverage = self._calculate_coverage()

        # Quality distribution
        quality_dist = self._confidence_distribution()

        # Orphaned appearances (no identity)
        orphaned_rate = self._orphaned_appearance_rate()

        return {
            "avg_confidence": avg_confidence,
            "coverage": coverage,
            "high_quality_rate": quality_dist["high"],  # >0.8
            "medium_quality_rate": quality_dist["medium"],  # 0.5-0.8
            "low_quality_rate": quality_dist["low"],  # <0.5
            "orphaned_rate": orphaned_rate
        }

    def _average_identity_confidence(self) -> float:
        query = """
        MATCH (i:Identity)
        RETURN avg(i.confidence_score) as avg_confidence
        """
        result = self.storage.graph.run_query(query)
        return result[0]['avg_confidence']

    def _calculate_coverage(self) -> float:
        query = """
        MATCH (a:Appearance)
        WITH count(a) as total
        MATCH (a:Appearance)<-[:HAS_APPEARANCE]-(:Identity)
        WITH total, count(a) as assigned
        RETURN toFloat(assigned) / total as coverage
        """
        result = self.storage.graph.run_query(query)
        return result[0]['coverage']

    def _confidence_distribution(self) -> Dict[str, float]:
        query = """
        MATCH (i:Identity)
        RETURN
            sum(CASE WHEN i.confidence_score >= 0.8 THEN 1 ELSE 0 END) as high,
            sum(CASE WHEN i.confidence_score >= 0.5 AND i.confidence_score < 0.8 THEN 1 ELSE 0 END) as medium,
            sum(CASE WHEN i.confidence_score < 0.5 THEN 1 ELSE 0 END) as low,
            count(i) as total
        """
        result = self.storage.graph.run_query(query)[0]
        total = result['total']

        return {
            "high": result['high'] / total if total > 0 else 0,
            "medium": result['medium'] / total if total > 0 else 0,
            "low": result['low'] / total if total > 0 else 0
        }
```

#### 3. Quality Issues Detection

```python
class QualityIssueDetector:
    def find_low_quality_identities(
        self,
        min_confidence: float = 0.5
    ) -> List[str]:
        """Find identities below confidence threshold"""
        query = """
        MATCH (i:Identity)
        WHERE i.confidence_score < $min_confidence
        RETURN i.id, i.confidence_score
        ORDER BY i.confidence_score
        """
        results = self.storage.graph.run_query(query, min_confidence=min_confidence)
        return [row['i.id'] for row in results]

    def find_ambiguous_matches(
        self,
        similarity_threshold: tuple = (0.5, 0.7)
    ) -> List[tuple[str, str, float]]:
        """Find appearance pairs with uncertain similarity"""
        query = """
        MATCH (a1:Appearance)-[m:PROBABLE_MATCH]->(a2:Appearance)
        WHERE m.similarity >= $min_sim AND m.similarity < $max_sim
        RETURN a1.id, a2.id, m.similarity
        ORDER BY m.similarity DESC
        """
        results = self.storage.graph.run_query(
            query,
            min_sim=similarity_threshold[0],
            max_sim=similarity_threshold[1]
        )
        return [(row['a1.id'], row['a2.id'], row['m.similarity']) for row in results]

    def find_singleton_identities(self) -> List[str]:
        """Find identities with only one appearance (might be noise)"""
        query = """
        MATCH (i:Identity)
        WHERE i.num_appearances = 1
        RETURN i.id
        """
        results = self.storage.graph.run_query(query)
        return [row['i.id'] for row in results]
```

---

## Component 4: Visualization Dashboard

### Purpose
Provide interactive web interface for exploring identities, trajectories, and analytics.

### Technology Choice

**Recommended**: Streamlit (Python-native, fast to build)

**Alternatives**:
- Dash (more flexible, production-ready)
- Gradio (simpler, ML-focused)
- Custom React + FastAPI (most flexible, more complex)

### Dashboard Layout

```
┌──────────────────────────────────────────────────────────┐
│                   Navigation Bar                          │
│  [Home] [Identities] [Trajectories] [Analytics] [Admin]  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    Overview Cards                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Total   │  │  Avg    │  │Coverage │  │Processing│    │
│  │Identities│  │Confidence│  │  Rate   │  │  Speed  │    │
│  │  1,234  │  │  0.87   │  │  94.2%  │  │  8.3 FPS│    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                 Main Content Area                         │
│  [Component-specific content based on selected page]     │
│                                                           │
│  - Identity browser and search                           │
│  - Trajectory visualization                              │
│  - Analytics charts and graphs                           │
│  - System metrics and logs                               │
└──────────────────────────────────────────────────────────┘
```

### Streamlit Implementation

```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ProjectB Identity Tracker",
    layout="wide"
)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Identities", "Trajectories", "Analytics", "Quality"]
)

if page == "Overview":
    st.title("Identity Tracking System - Overview")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Identities", "1,234", "+23")

    with col2:
        st.metric("Avg Confidence", "0.87", "+0.03")

    with col3:
        st.metric("Coverage Rate", "94.2%", "+1.2%")

    with col4:
        st.metric("Processing Speed", "8.3 FPS", "-0.2")

    # Charts
    st.subheader("Identity Confidence Distribution")
    # ... histogram of confidence scores

    st.subheader("Activity Timeline")
    # ... timeline of appearances over time

elif page == "Identities":
    st.title("Identity Browser")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5)
    with col2:
        min_appearances = st.number_input("Min Appearances", 1, 100, 5)
    with col3:
        camera_filter = st.multiselect("Cameras", get_all_cameras())

    # Identity list
    identities = fetch_identities(
        min_confidence=min_confidence,
        min_appearances=min_appearances,
        cameras=camera_filter
    )

    for identity in identities:
        with st.expander(f"Identity {identity.id} (Confidence: {identity.confidence_score:.2f})"):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.write(f"**First Seen:** {identity.first_seen}")
                st.write(f"**Last Seen:** {identity.last_seen}")
                st.write(f"**Appearances:** {identity.num_appearances}")
                st.write(f"**Cameras:** {identity.num_cameras}")

            with col2:
                # Show sample images
                appearances = get_identity_appearances(identity.id, limit=6)
                st.image([a.frame_path for a in appearances], width=100)

elif page == "Trajectories":
    st.title("Trajectory Viewer")

    # Identity selection
    identity_id = st.selectbox("Select Identity", get_all_identity_ids())

    if identity_id:
        trajectory = extract_trajectory(identity_id)

        # Trajectory map
        st.subheader("Spatial Trajectory")
        fig = go.Figure()

        # Add camera locations
        cameras = get_all_cameras()
        fig.add_trace(go.Scatter(
            x=[c.position_x for c in cameras],
            y=[c.position_y for c in cameras],
            mode='markers+text',
            name='Cameras',
            text=[c.name for c in cameras],
            marker=dict(size=15, color='lightblue')
        ))

        # Add trajectory path
        fig.add_trace(go.Scatter(
            x=[p.camera_location[0] for p in trajectory.points],
            y=[p.camera_location[1] for p in trajectory.points],
            mode='lines+markers',
            name='Trajectory',
            marker=dict(
                size=8,
                color=[p.confidence for p in trajectory.points],
                colorscale='Viridis',
                showscale=True
            )
        ))

        st.plotly_chart(fig, use_container_width=True)

        # Timeline
        st.subheader("Temporal Timeline")
        timeline_fig = px.timeline(
            trajectory_to_dataframe(trajectory),
            x_start="start_time",
            x_end="end_time",
            y="camera_name",
            color="confidence"
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Duration", f"{trajectory.total_duration/60:.1f} min")
        with col2:
            st.metric("Cameras Visited", trajectory.cameras_visited)
        with col3:
            st.metric("Distance Traveled", f"{trajectory.total_distance:.1f} m")

elif page == "Analytics":
    st.title("Analytics & Insights")

    # Community detection
    st.subheader("Identity Communities")
    communities = find_communities()
    # ... visualization

    # Co-occurrence analysis
    st.subheader("Co-Occurring Identities")
    # ... graph visualization

    # Transition patterns
    st.subheader("Common Camera Transitions")
    transitions = find_transition_patterns()
    # ... sankey diagram

elif page == "Quality":
    st.title("Data Quality Assessment")

    # System metrics
    metrics = calculate_system_metrics()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Confidence", f"{metrics['avg_confidence']:.2f}")
    with col2:
        st.metric("Coverage", f"{metrics['coverage']*100:.1f}%")
    with col3:
        st.metric("Orphaned Appearances", f"{metrics['orphaned_rate']*100:.1f}%")

    # Quality distribution
    st.subheader("Quality Distribution")
    quality_df = pd.DataFrame({
        "Quality": ["High (>0.8)", "Medium (0.5-0.8)", "Low (<0.5)"],
        "Percentage": [
            metrics['high_quality_rate'] * 100,
            metrics['medium_quality_rate'] * 100,
            metrics['low_quality_rate'] * 100
        ]
    })
    fig = px.bar(quality_df, x="Quality", y="Percentage")
    st.plotly_chart(fig)

    # Issues
    st.subheader("Quality Issues")
    low_quality = find_low_quality_identities()
    st.write(f"Found {len(low_quality)} low-quality identities")

    ambiguous = find_ambiguous_matches()
    st.write(f"Found {len(ambiguous)} ambiguous matches needing review")
```

---

## Component 5: Export & Reporting

### Export Formats

#### 1. Identity Export (CSV)
```csv
identity_id,first_seen,last_seen,num_appearances,num_cameras,confidence_score
id_001,2025-11-18T10:00:00,2025-11-18T15:30:00,47,3,0.87
id_002,2025-11-18T09:15:00,2025-11-18T14:20:00,32,2,0.92
```

#### 2. Trajectory Export (JSON)
```json
{
  "identity_id": "id_001",
  "trajectory": [
    {
      "timestamp": "2025-11-18T10:00:00Z",
      "camera_id": "cam_001",
      "camera_name": "Entrance",
      "location": [10.5, 20.3],
      "confidence": 0.95
    },
    ...
  ],
  "metrics": {
    "total_duration": 19800,
    "cameras_visited": 3,
    "distance_traveled": 125.4
  }
}
```

#### 3. Graph Export (GraphML)
```python
def export_graph_to_graphml(output_path: str):
    """Export Neo4j graph to GraphML format"""
    query = """
    CALL apoc.export.graphml.all($output_path, {})
    YIELD file, source, format, nodes, relationships, time
    RETURN file, nodes, relationships
    """
    # ... implementation
```

### Report Generation

```python
class ReportGenerator:
    def generate_summary_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Generate comprehensive summary report"""

        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "identities": self._identity_stats(start_date, end_date),
            "appearances": self._appearance_stats(start_date, end_date),
            "quality": self._quality_stats(),
            "performance": self._performance_stats(),
            "top_identities": self._top_identities(limit=10)
        }

        return report

    def export_report_pdf(self, report: Dict, output_path: str):
        """Export report as PDF"""
        # Use reportlab or similar
        pass

    def export_report_html(self, report: Dict, output_path: str):
        """Export report as HTML"""
        # Use Jinja2 template
        pass
```

---

## API Endpoints

### Analytics Endpoints

```python
# Get system summary
GET /api/v1/analytics/summary
Response: {
  "total_identities": 1234,
  "total_appearances": 45678,
  "avg_confidence": 0.87,
  "coverage": 0.942
}

# Get identity trajectory
GET /api/v1/analytics/identities/{id}/trajectory
Response: {
  "identity_id": "id_001",
  "trajectory": [...],
  "metrics": {...}
}

# Get co-occurring identities
GET /api/v1/analytics/identities/{id}/cooccurrences?time_window=60
Response: {
  "identity_id": "id_001",
  "cooccurrences": [
    {"identity_id": "id_002", "count": 15},
    ...
  ]
}

# Get quality metrics
GET /api/v1/analytics/quality
Response: {
  "avg_confidence": 0.87,
  "quality_distribution": {...},
  "issues": [...]
}

# Export data
GET /api/v1/export/identities?format=csv&min_confidence=0.5
Response: CSV file download

GET /api/v1/export/graph?format=graphml
Response: GraphML file download
```

---

## Performance Requirements

- **Dashboard Load Time**: <2 seconds
- **Trajectory Rendering**: <1 second for 100 points
- **Analytics Query**: <5 seconds for complex queries
- **Export**: <10 seconds for 1000 identities

---

## Testing Strategy

### Unit Tests
- Test metric calculations
- Test trajectory extraction
- Test quality assessment algorithms

### Integration Tests
- Test dashboard rendering
- Test API endpoints
- Test export functionality

### User Testing
- Usability testing with real users
- Gather feedback on interface
- Iterate on design

---

## Future Enhancements

- Real-time dashboard updates
- Advanced ML-based anomaly detection
- Collaborative annotation interface
- Custom report templates
- Integration with external BI tools
- Mobile dashboard interface

---

**Document Version History**:
- v1.0 (2025-11-18): Initial draft
