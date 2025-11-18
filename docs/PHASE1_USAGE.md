# Phase 1: Core Pipeline - Usage Guide

Welcome to Phase 1! You now have a working end-to-end video processing pipeline.

---

## 🎯 What's Working

Phase 1 MVP provides:
- ✅ Video loading and frame extraction
- ✅ YOLOv8 person detection
- ✅ Multi-object tracking (IOU-based)
- ✅ InsightFace face recognition
- ✅ FAISS vector storage
- ✅ Per-track identity clustering
- ✅ Complete processing script

---

## 📦 Installation

### 1. Install Dependencies

Make sure your virtual environment is activated:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Phase 1 requirements:

```bash
# Core CV libraries
pip install opencv-python ultralytics insightface onnxruntime faiss-cpu tqdm
```

### 2. Download Models (Automatic)

Models will be automatically downloaded on first run:
- **YOLOv8n**: ~6MB (person detection)
- **InsightFace buffalo_l**: ~500MB (face recognition)

The script will handle downloads automatically!

---

## 🚀 Quick Start

### Process a Video

```bash
python scripts/process_video.py \
    --video path/to/your/video.mp4 \
    --camera-id cam_001
```

### With Frame Saving

```bash
python scripts/process_video.py \
    --video path/to/your/video.mp4 \
    --camera-id cam_entrance \
    --save-frames
```

### Specify Output Directory

```bash
python scripts/process_video.py \
    --video path/to/your/video.mp4 \
    --camera-id cam_001 \
    --output ./my_results
```

---

## 📊 Expected Output

### Console Output

```
============================================================
Starting video processing
============================================================

1. Initializing components...
Loading YOLO model: data/models/yolov8n.pt
  Device: cpu
  Confidence threshold: 0.5
✓ YOLOv8 detector ready

Loading InsightFace model: buffalo_l
  Detection threshold: 0.8
✓ InsightFace extractor ready

FAISS vector store initialized
  Face dimension: 512
  ReID dimension: 2048

2. Loading video: video.mp4
Video loaded: video.mp4
  Resolution: 1920x1080
  FPS: 30.00
  Frames: 9000
  Duration: 300.00s

3. Processing frames...
Processing: 100%|████████████████| 1800/1800 [05:23<00:00, 5.56it/s]

4. Processing complete!
============================================================
Results:
  Total appearances: 342
  Unique tracks: 5

Per-track statistics:
  Track   1: 124 appearances,  89 faces (71.8%)
  Track   2:  67 appearances,  51 faces (76.1%)
  Track   3:  89 appearances,  72 faces (80.9%)
  Track   4:  34 appearances,  12 faces (35.3%)
  Track   5:  28 appearances,  19 faces (67.9%)

Results saved to: data/output/run_20251118_103045
✓ Processing complete!
```

### Output Files

```
data/output/run_20251118_103045/
├── results.txt          # Summary statistics
└── frames/             # If --save-frames was used
    ├── frame_000042_track_1.jpg
    ├── frame_000087_track_2.jpg
    └── ...
```

---

## 🎥 Getting Test Videos

### Option 1: Use Your Own Videos
- Any MP4, AVI, or MOV file
- Works best with people visible
- Recommended: 720p or 1080p

### Option 2: Download Test Videos

Public datasets with videos:
- **MOT Challenge**: https://motchallenge.net/
- **Sample surveillance footage**: Search "pedestrian tracking dataset"

### Option 3: Create Test Video

```bash
# Download a sample video using yt-dlp (if you have it)
yt-dlp -f 'bestvideo[height<=720]' 'YOUR_VIDEO_URL' -o test_video.mp4
```

---

## ⚙️ Configuration

Edit `.env` to adjust parameters:

```bash
# Processing
FPS_SAMPLE_RATE=5          # Process every 5th frame (faster)
DEVICE=cpu                  # or 'cuda' if GPU available

# Detection
DETECTION_CONFIDENCE_THRESHOLD=0.5    # Lower = more detections
MIN_DETECTION_SIZE=64                 # Minimum person size (pixels)

# Face Recognition
FACE_DETECTION_THRESHOLD=0.8          # Higher = stricter face detection
FACE_MIN_SIZE=64                      # Minimum face size
```

---

## 🐛 Troubleshooting

### Issue: "ultralytics not installed"

```bash
pip install ultralytics
```

### Issue: "insightface not installed"

```bash
pip install insightface onnxruntime
```

### Issue: Models downloading slowly

Models are downloaded to:
- **YOLOv8**: `~/.cache/torch/hub/ultralytics/`
- **InsightFace**: `~/.insightface/models/`

Be patient on first run!

### Issue: GPU not detected

Set in `.env`:
```bash
DEVICE=cpu
```

### Issue: Out of memory

Reduce frame sampling:
```bash
# In .env
FPS_SAMPLE_RATE=10  # Process fewer frames
```

### Issue: Too slow

**CPU optimization**:
1. Use smaller YOLO model (already using yolov8n)
2. Increase `FPS_SAMPLE_RATE` to process fewer frames
3. Reduce video resolution before processing

**GPU acceleration** (if available):
```bash
# In .env
DEVICE=cuda

# Install CUDA-enabled packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
```

---

## 📈 Performance Expectations

On typical hardware (CPU):

| Resolution | FPS Processing Speed | Real-time Factor |
|------------|---------------------|------------------|
| 720p       | ~5 FPS              | 0.17x (6x slower)|
| 1080p      | ~3 FPS              | 0.10x (10x slower)|

On GPU (NVIDIA RTX 3060):
| Resolution | FPS Processing Speed | Real-time Factor |
|------------|---------------------|------------------|
| 720p       | ~25 FPS             | 0.83x (near real-time)|
| 1080p      | ~15 FPS             | 0.50x (2x slower)|

**Note**: Phase 1 is optimized for batch processing, not real-time.

---

## 🔍 Understanding the Output

### Track IDs
- Each unique person gets a tracking ID (1, 2, 3, ...)
- Same person across frames keeps the same ID (within video)
- IDs are **not** persistent across videos yet (Phase 2 feature)

### Face Detection Rate
- **>70%**: Good! Person facing camera most of the time
- **40-70%**: Moderate. Person sometimes facing away
- **<40%**: Low. Person rarely facing camera or face too small

### Appearances vs Tracks
- **Appearance**: Single detection in one frame
- **Track**: Collection of appearances for same person
- Example: Track 1 with 124 appearances = person detected in 124 frames

---

## 🎯 Next Steps

### Phase 2 Features (Coming Next):
- ✨ Neo4j graph storage
- ✨ Cross-camera identity matching
- ✨ ReID embeddings for body matching
- ✨ Advanced identity resolution
- ✨ Confidence scoring

### Test Your Setup:
1. Process a test video
2. Check results in output directory
3. Verify face detection is working
4. Review track statistics

---

## 💡 Tips for Best Results

1. **Good Lighting**: Better face detection
2. **Clear Views**: Avoid heavy occlusions
3. **Appropriate Distance**: People should be >100px tall
4. **Frontal Views**: More faces detected = better results
5. **Stable Camera**: Easier tracking

---

## 🆘 Need Help?

Check:
1. Logs in `logs/projectb_YYYY-MM-DD.log`
2. Error logs in `logs/errors_YYYY-MM-DD.log`
3. Output `results.txt` for statistics

Common Issues:
- No detections → Lower `DETECTION_CONFIDENCE_THRESHOLD`
- Too many false detections → Raise threshold
- No faces → Check `FACE_DETECTION_THRESHOLD` or face visibility

---

**Phase 1 Status**: ✅ Complete and Ready!

Happy tracking! 🎉
