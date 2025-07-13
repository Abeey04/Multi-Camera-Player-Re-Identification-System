# Multi-Camera Player Re-Identification System

## Overview

This system implements a sophisticated computer vision pipeline for real-time player re-identification across multiple camera views. The solution addresses the fundamental challenge of maintaining consistent player identity across different viewing angles, lighting conditions, and occlusions commonly encountered in sports analytics.

![alt text](https://github.com/Abeey04/Multi-Camera-Player-Re-Identification-System/blob/main/misc/output_reid.gif)


## System Architecture

### Core Components

1. **Detection Module**: YOLOv8-based player detection with configurable confidence thresholds
2. **Tracking Module**: ByteTracker implementation for temporal consistency within each camera
3. **Feature Extraction**: Multi-modal feature representation combining color and texture descriptors
4. **Cross-Camera Matching**: Hungarian algorithm-based optimal assignment with stability constraints
5. **Visualization Pipeline**: Real-time correspondence visualization with color-coded matching

### Technical Innovation

The system introduces a **Stable Matching System** that addresses temporal instability through:
- Historical consistency tracking over configurable time windows
- Majority voting for match stabilization
- One-to-one mapping enforcement to prevent identity ambiguity

## Installation & Setup

### Dependencies

```bash
# Core computer vision libraries
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install ultralytics>=8.0.0

# Scientific computing
pip install scipy>=1.10.0
pip install scikit-image>=0.20.0

# ByteTracker dependencies (modified version included in submission)
pip install cython_bbox
pip install lap
```

**Note on ByteTracker Integration**: This submission includes a modified version of ByteTracker source code rather than using the standard pip installation. This ensures:
- Complete experimental reproducibility
- Transparency of algorithmic modifications
- Independence from external package version conflicts
- Full control over tracking pipeline behavior

### Environment Requirements

- Python 3.8+
- GPU support recommended (CUDA-compatible)
- Minimum 8GB RAM for video processing
- Storage space for model weights and video files

### File Structure

```
project/
├── mapping.py              # Main processing script
├── bytetracker/            # Modified ByteTracker implementation
│   ├── __init__.py
│   ├── byte_tracker.py     # Core tracking algorithm
│   ├── kalman_filter.py    # Modified Kalman filter
│   └── matching.py         # Association algorithms
├── best.pt                 # Trained YOLO model weights
├── broadcast.mp4           # Primary camera view
├── tacticam.mp4           # Secondary camera view
├── output_reid.mp4        # Generated output video
├── requirements.txt       # Exact dependency specifications
└── README.md              # This documentation
```

### ByteTracker Modifications

The included ByteTracker implementation contains research-specific modifications:

1. **Tracking Parameter Optimization**: Adjusted default parameters for sports-specific scenarios
2. **Feature Integration**: Enhanced support for cross-camera feature propagation
3. **Stability Enhancements**: Modified track lifecycle management for improved temporal consistency
4. **Performance Optimizations**: Computational efficiency improvements for real-time processing

These modifications are documented within the ByteTracker source files and represent original contributions to the tracking methodology.

## Usage

### Basic Execution

```bash
python mapping.py
```

### Configuration Parameters

The system provides extensive configuration options:

```python
# Detection and Tracking Parameters
CONF_THRESHOLD = 0.7        # Detection confidence threshold
PLAYER_CLASS_ID = 2         # YOLO class ID for players

# Tracker Configuration
TRACKER_CONFIG = {
    'track_thresh': 0.7,     # Tracking confidence threshold
    'track_buffer': 120,     # Frame buffer for track management
    'match_thresh': 0.7,     # Track association threshold
    'frame_rate': 30         # Video frame rate
}

# Feature Extraction Parameters
PATCH_SIZE = (64, 128)      # Standardized patch dimensions
MATCH_THRESHOLD = 0.3       # Cross-camera matching threshold
STABILITY_FRAMES = 15       # Frames for stable match consensus
```

### Input/Output Specifications

**Input Requirements:**
- Two synchronized video streams (broadcast.mp4, tacticam.mp4)
- Pre-trained YOLO model (best.pt) for player detection
- Videos should have consistent frame rates and timing

**Output Format:**
- Side-by-side video visualization (output_reid.mp4)
- Color-coded bounding boxes with consistent player IDs
- Connection lines indicating cross-camera correspondences

## Technical Methodology

### Feature Extraction Pipeline

The system employs a sophisticated multi-modal feature extraction approach:

1. **Color Features**: HSV histogram analysis focusing on hue and saturation components for lighting invariance
2. **Texture Features**: HOG (Histogram of Oriented Gradients) descriptors for structural pattern recognition
3. **Normalization**: L2 normalization for feature scaling and robustness

### Cross-Camera Matching Algorithm

The matching process implements several advanced techniques:

1. **Cost Matrix Computation**: Cosine distance between feature vectors across camera views
2. **Hungarian Algorithm**: Optimal assignment ensuring globally minimal matching cost
3. **Threshold Filtering**: Rejection of low-confidence matches based on empirical thresholds
4. **Temporal Stabilization**: Historical consistency tracking to prevent ID switching

### Stability Mechanisms

The **StableMatchingSystem** class addresses temporal inconsistencies through:
- Rolling window analysis of match history
- Majority voting for stable assignment
- One-to-one mapping enforcement
- Configurable stability thresholds

## Performance Characteristics

### Computational Complexity

- Detection: O(n) per frame per camera
- Feature Extraction: O(k) where k is number of detected players
- Matching: O(k₁ × k₂) for cross-camera assignment
- Overall: Real-time processing at 30 FPS on modern hardware

### Accuracy Considerations

The system balances precision and recall through:
- Configurable confidence thresholds
- Multi-modal feature representation
- Temporal consistency constraints
- Robust matching algorithms

## Limitations and Future Work

### Current Limitations

1. **Appearance-Based Matching**: Limited performance under severe appearance changes
2. **Occlusion Handling**: Reduced accuracy during prolonged occlusions
3. **Scale Invariance**: Performance degrades with significant scale differences
4. **Lighting Sensitivity**: HSV features may be affected by dramatic lighting changes

### Potential Improvements

1. **Deep Learning Features**: Integration of CNN-based re-identification features
2. **Pose Estimation**: Incorporation of skeletal features for additional discrimination
3. **Motion Analysis**: Temporal motion patterns for enhanced matching
4. **Active Learning**: Adaptive threshold adjustment based on scene analysis

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce video resolution or increase system RAM
2. **Model Loading**: Ensure best.pt is in the correct directory
3. **Video Sync**: Verify input videos are temporally aligned
4. **Performance**: Enable GPU acceleration for real-time processing

### Debug Mode

Enable verbose logging by modifying the YOLO detection call:
```python
results_b = model(frame_b, conf=CONF_THRESHOLD, classes=[PLAYER_CLASS_ID], verbose=True)
```

## Research Applications

This system provides a foundation for:
- Sports analytics and performance evaluation
- Multi-camera surveillance systems
- Crowd analysis and tracking
- Automated video annotation
- Real-time broadcast enhancement

## License

This project is released under the MIT License. See LICENSE file for details.

## Contact

For technical questions or collaboration opportunities, please reach out through the appropriate channels.
