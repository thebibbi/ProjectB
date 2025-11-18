#!/usr/bin/env python3
"""
Neo4j Schema Initialization Script

This script initializes the Neo4j database with the required schema:
- Constraints (unique identifiers)
- Indexes (for query performance)
- Sample data (optional)

Usage:
    python scripts/setup_neo4j.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class Neo4jSchemaSetup:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed")

    def create_constraints(self):
        """Create unique constraints for node types"""
        logger.info("Creating constraints...")

        constraints = [
            # Identity constraints
            """
            CREATE CONSTRAINT identity_id_unique IF NOT EXISTS
            FOR (i:Identity) REQUIRE i.id IS UNIQUE
            """,
            # Appearance constraints
            """
            CREATE CONSTRAINT appearance_id_unique IF NOT EXISTS
            FOR (a:Appearance) REQUIRE a.id IS UNIQUE
            """,
            # Camera constraints
            """
            CREATE CONSTRAINT camera_id_unique IF NOT EXISTS
            FOR (c:Camera) REQUIRE c.id IS UNIQUE
            """,
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.success(f"✓ Created constraint")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")

    def create_indexes(self):
        """Create indexes for frequently queried properties"""
        logger.info("Creating indexes...")

        indexes = [
            # Appearance indexes
            """
            CREATE INDEX appearance_timestamp IF NOT EXISTS
            FOR (a:Appearance) ON (a.timestamp)
            """,
            """
            CREATE INDEX appearance_camera IF NOT EXISTS
            FOR (a:Appearance) ON (a.camera_id)
            """,
            """
            CREATE INDEX appearance_tracking IF NOT EXISTS
            FOR (a:Appearance) ON (a.tracking_id)
            """,
            # Identity indexes
            """
            CREATE INDEX identity_confidence IF NOT EXISTS
            FOR (i:Identity) ON (i.confidence_score)
            """,
            """
            CREATE INDEX identity_first_seen IF NOT EXISTS
            FOR (i:Identity) ON (i.first_seen)
            """,
            """
            CREATE INDEX identity_last_seen IF NOT EXISTS
            FOR (i:Identity) ON (i.last_seen)
            """,
            # Camera indexes
            """
            CREATE INDEX camera_name IF NOT EXISTS
            FOR (c:Camera) ON (c.name)
            """,
        ]

        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                    logger.success(f"✓ Created index")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")

    def verify_schema(self):
        """Verify that schema was created successfully"""
        logger.info("Verifying schema...")

        with self.driver.session() as session:
            # Check constraints
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            logger.info(f"Found {len(constraints)} constraints")

            # Check indexes
            result = session.run("SHOW INDEXES")
            indexes = list(result)
            logger.info(f"Found {len(indexes)} indexes")

    def clear_database(self):
        """Clear all data from database (USE WITH CAUTION!)"""
        logger.warning("Clearing database...")

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.success("✓ Database cleared")

    def create_sample_cameras(self):
        """Create sample camera nodes for testing"""
        logger.info("Creating sample cameras...")

        cameras = [
            {
                "id": "cam_entrance",
                "name": "Main Entrance",
                "location": "Building A - Entrance",
                "position_x": 0.0,
                "position_y": 0.0,
                "position_z": 0.0,
                "orientation": "north",
                "fov": 90.0,
            },
            {
                "id": "cam_hallway_1",
                "name": "Hallway 1",
                "location": "Building A - Hallway",
                "position_x": 10.0,
                "position_y": 0.0,
                "position_z": 0.0,
                "orientation": "east",
                "fov": 90.0,
            },
            {
                "id": "cam_lobby",
                "name": "Lobby",
                "location": "Building A - Lobby",
                "position_x": 20.0,
                "position_y": 5.0,
                "position_z": 0.0,
                "orientation": "south",
                "fov": 110.0,
            },
        ]

        with self.driver.session() as session:
            for camera in cameras:
                query = """
                MERGE (c:Camera {id: $id})
                SET c.name = $name,
                    c.location = $location,
                    c.position_x = $position_x,
                    c.position_y = $position_y,
                    c.position_z = $position_z,
                    c.orientation = $orientation,
                    c.fov = $fov,
                    c.created_at = datetime()
                """
                session.run(query, **camera)
                logger.success(f"✓ Created camera: {camera['name']}")

            # Create adjacency relationships
            adjacencies = [
                ("cam_entrance", "cam_hallway_1", 10.0, 15),
                ("cam_hallway_1", "cam_lobby", 12.0, 20),
            ]

            for cam1_id, cam2_id, distance, avg_time in adjacencies:
                query = """
                MATCH (c1:Camera {id: $cam1_id})
                MATCH (c2:Camera {id: $cam2_id})
                MERGE (c1)-[r:ADJACENT_TO]->(c2)
                SET r.distance = $distance,
                    r.avg_transition_time = $avg_time
                MERGE (c2)-[r2:ADJACENT_TO]->(c1)
                SET r2.distance = $distance,
                    r2.avg_transition_time = $avg_time
                """
                session.run(
                    query,
                    cam1_id=cam1_id,
                    cam2_id=cam2_id,
                    distance=distance,
                    avg_time=avg_time,
                )
                logger.success(f"✓ Created adjacency: {cam1_id} <-> {cam2_id}")


def main():
    """Main setup function"""
    logger.info("=== Neo4j Schema Setup ===")

    # Get configuration from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "projectb_password")

    if password == "your_secure_password_here":
        logger.error("Please set NEO4J_PASSWORD in .env file")
        sys.exit(1)

    # Initialize setup
    setup = Neo4jSchemaSetup(uri, user, password)

    try:
        # Create schema
        setup.create_constraints()
        setup.create_indexes()
        setup.verify_schema()

        # Optional: Create sample data
        create_sample = input("\nCreate sample cameras? (y/n): ").lower().strip()
        if create_sample == "y":
            setup.create_sample_cameras()

        logger.success("\n✓ Neo4j schema setup complete!")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    finally:
        setup.close()


if __name__ == "__main__":
    main()
