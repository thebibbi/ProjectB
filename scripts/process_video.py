#!/usr/bin/env python3
"""
Video Processing Script

Process a video file to detect, track, and identify people.

Usage:
    python scripts/process_video.py --video path/to/video.mp4 --camera-id cam_001
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import log
from src.config import settings
from src.ingestion.video_loader import VideoLoader
from src.detection.detector import PersonDetector
from src.detection.tracker import SimpleTracker
from src.features.face_extractor import FaceExtractor
from src.storage.vector_store import FAISSVectorStore
from src.storage.models import AppearanceCreate


def process_video(
    video_path: str,
    camera_id: str,
    output_dir: str = None,
    save_frames: bool = False
):
    """
    Process video end-to-end

    Args:
        video_path: Path to video file
        camera_id: Camera identifier
        output_dir: Output directory for results
        save_frames: Whether to save detection frames
    """
    log.info("="*60)
    log.info("Starting video processing")
    log.info("="*60)

    output_dir = Path(output_dir or settings.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_frames:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

    # Initialize components
    log.info("\n1. Initializing components...")

    try:
        detector = PersonDetector()
        tracker = SimpleTracker()
        face_extractor = FaceExtractor()
        vector_store = FAISSVectorStore()
    except Exception as e:
        log.error(f"Failed to initialize components: {e}")
        log.error("Make sure all dependencies are installed:")
        log.error("  pip install ultralytics insightface onnxruntime faiss-cpu")
        return

    # Load video
    log.info(f"\n2. Loading video: {video_path}")
    loader = VideoLoader(video_path)

    # Process frames
    log.info("\n3. Processing frames...")

    appearances = []
    identities_by_track = {}  # tracking_id -> {appearances, face_embeddings}

    total_frames_to_process = loader.total_frames // settings.fps_sample_rate

    with tqdm(total=total_frames_to_process, desc="Processing") as pbar:
        for frame_number, frame, timestamp in loader.frames():
            # Detect people
            detections = detector.detect(frame, frame_number, timestamp)

            if len(detections) == 0:
                pbar.update(1)
                continue

            # Track people
            tracking_results = tracker.update(detections)

            # Extract faces and create appearances
            for track_result in tracking_results:
                tracking_id = track_result.tracking_id
                detection = track_result.detection

                # Initialize tracking data if needed
                if tracking_id not in identities_by_track:
                    identities_by_track[tracking_id] = {
                        'appearances': [],
                        'face_embeddings': []
                    }

                # Extract face from detection bbox
                bbox_tuple = (
                    detection.bbox.x,
                    detection.bbox.y,
                    detection.bbox.w,
                    detection.bbox.h
                )

                face_embedding = face_extractor.extract_from_bbox(frame, bbox_tuple)

                # Create appearance record
                frame_path = ""
                if save_frames:
                    frame_path = str(frames_dir / f"frame_{frame_number:06d}_track_{tracking_id}.jpg")
                    loader.save_frame(frame, frame_path, bbox_tuple)

                appearance = AppearanceCreate(
                    timestamp=datetime.fromtimestamp(timestamp),
                    camera_id=camera_id,
                    tracking_id=tracking_id,
                    bbox=detection.bbox,
                    detection_confidence=detection.confidence,
                    face_quality=face_embedding.quality_score if face_embedding else None,
                    frame_path=frame_path
                )

                appearances.append(appearance)
                identities_by_track[tracking_id]['appearances'].append(appearance)

                if face_embedding:
                    identities_by_track[tracking_id]['face_embeddings'].append(face_embedding)

            pbar.update(1)

    loader.release()

    # Summarize results
    log.info("\n4. Processing complete!")
    log.info("="*60)
    log.info(f"Results:")
    log.info(f"  Total appearances: {len(appearances)}")
    log.info(f"  Unique tracks: {len(identities_by_track)}")

    # Show per-track statistics
    log.info(f"\nPer-track statistics:")
    for track_id, track_data in sorted(identities_by_track.items()):
        num_appearances = len(track_data['appearances'])
        num_faces = len(track_data['face_embeddings'])
        face_rate = (num_faces / num_appearances * 100) if num_appearances > 0 else 0

        log.info(f"  Track {track_id:3d}: {num_appearances:3d} appearances, {num_faces:3d} faces ({face_rate:.1f}%)")

    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Camera: {camera_id}\n")
        f.write(f"Processed: {datetime.now()}\n")
        f.write(f"\nTotal appearances: {len(appearances)}\n")
        f.write(f"Unique tracks: {len(identities_by_track)}\n")
        f.write(f"\nPer-track statistics:\n")
        for track_id, track_data in sorted(identities_by_track.items()):
            num_appearances = len(track_data['appearances'])
            num_faces = len(track_data['face_embeddings'])
            f.write(f"Track {track_id}: {num_appearances} appearances, {num_faces} faces\n")

    log.info(f"\nResults saved to: {output_dir}")
    log.success("Processing complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Process video for identity tracking")

    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file"
    )

    parser.add_argument(
        "--camera-id",
        default="cam_default",
        help="Camera identifier"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory"
    )

    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save detection frames"
    )

    args = parser.parse_args()

    # Validate video exists
    if not Path(args.video).exists():
        log.error(f"Video file not found: {args.video}")
        sys.exit(1)

    process_video(
        video_path=args.video,
        camera_id=args.camera_id,
        output_dir=args.output,
        save_frames=args.save_frames
    )


if __name__ == "__main__":
    main()
