import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from skimage.feature import hog
from bytetracker.byte_tracker import BYTETracker

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Ultralytics version: {ultralytics.__version__}")

# --- Configuration Parameters ---
MODEL_PATH = 'best.pt'
BROADCAST_VIDEO_PATH = 'broadcast.mp4'
TACTICAM_VIDEO_PATH = 'tacticam.mp4'
OUTPUT_VIDEO_PATH = 'output_reid.mp4'

# Detection and Tracking Parameters
CONF_THRESHOLD = 0.7
PLAYER_CLASS_ID = 2

# Tracker Configuration
TRACKER_CONFIG = {
    'track_thresh': 0.7,
    'track_buffer': 120,
    'match_thresh': 0.7,
    'frame_rate': 30
}

# Feature Extraction Parameters
PATCH_SIZE = (64, 128)

# Matching Parameters
MATCH_THRESHOLD = 0.3  
MIN_PATCH_SIZE = 20    # Minimum width/height for valid patches
STABILITY_FRAMES = 15   # Frames to maintain stable matches

# --- Enhanced Feature Extraction ---
def extract_visual_features(patch):
    if patch.size == 0 or patch.shape[0] < MIN_PATCH_SIZE or patch.shape[1] < MIN_PATCH_SIZE:
        return None
    
    # Resize to standard size
    resized_patch = cv2.resize(patch, PATCH_SIZE)

    hsv_patch = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2HSV)
    
    #hue and saturation
    hist_h = cv2.calcHist([hsv_patch], [0], None, [36], [0, 180])
    hist_s = cv2.calcHist([hsv_patch], [1], None, [32], [0, 256])
    
    # Normalize histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    
    # HOG features for texture/shape
    gray_patch = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_patch, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
    
    # Combine and normalize features
    combined_features = np.concatenate((hist_h, hist_s, hog_features))
    return cv2.normalize(combined_features, combined_features).flatten()

# --- Stable Matching System ---
class StableMatchingSystem:
    def __init__(self, stability_frames=STABILITY_FRAMES):
        self.stability_frames = stability_frames
        self.match_history = {}  # track_id -> [recent_matches]
        self.stable_matches = {}  # broadcast_id -> tacticam_id
        
    def update_matches(self, frame_matches):
        for b_id, t_id in frame_matches.items():
            if b_id not in self.match_history:
                self.match_history[b_id] = []
            self.match_history[b_id].append(t_id)

            if len(self.match_history[b_id]) > self.stability_frames:
                self.match_history[b_id].pop(0)
y
        new_stable_matches = {}
        for b_id, history in self.match_history.items():
            if len(history) >= min(3, self.stability_frames):
                unique, counts = np.unique(history, return_counts=True)
                most_frequent = unique[np.argmax(counts)]
                
                if counts[np.argmax(counts)] >= len(history) * 0.6:
                    new_stable_matches[b_id] = most_frequent

        self.stable_matches = self._ensure_one_to_one_mapping(new_stable_matches)
        return self.stable_matches
    
    def _ensure_one_to_one_mapping(self, matches):
        used_tacticam_ids = set()
        final_matches = {}
        
        for b_id in sorted(matches.keys()):
            t_id = matches[b_id]
            if t_id not in used_tacticam_ids:
                final_matches[b_id] = t_id
                used_tacticam_ids.add(t_id)
        
        return final_matches

# --- Main Processing Function ---
def process_videos():
    model = YOLO(MODEL_PATH)
    cap_broadcast = cv2.VideoCapture(BROADCAST_VIDEO_PATH)
    cap_tacticam = cv2.VideoCapture(TACTICAM_VIDEO_PATH)
    
    if not cap_broadcast.isOpened() or not cap_tacticam.isOpened():
        print("Error: Could not open video files.")
        return
    
    # Setup video writer
    w_b = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_b = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_t = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_t = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_broadcast.get(cv2.CAP_PROP_FPS))
    
    scale_factor = h_b / h_t
    w_t_scaled = int(w_t * scale_factor)
    output_width = w_b + w_t_scaled
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (output_width, h_b))
    
    # Initialize trackers and matching system
    tracker_b = bytetracker.BYTETracker(**TRACKER_CONFIG)
    tracker_t = bytetracker.BYTETracker(**TRACKER_CONFIG)
    matching_system = StableMatchingSystem()
    
    frame_count = 0
    
    while True:
        ret_b, frame_b = cap_broadcast.read()
        ret_t, frame_t = cap_tacticam.read()
        
        if not ret_b or not ret_t:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing Frame: {frame_count}")
        
        # Detection
        results_b = model(frame_b, conf=CONF_THRESHOLD, classes=[PLAYER_CLASS_ID], verbose=False)
        results_t = model(frame_t, conf=CONF_THRESHOLD, classes=[PLAYER_CLASS_ID], verbose=False)
        
        # Convert to tracker format
        detections_b = np.array([
            box.xyxy[0].tolist() + [box.conf[0].item(), box.cls[0].item()]
            for res in results_b for box in res.boxes
        ]) if results_b[0].boxes else np.empty((0, 6))
        
        detections_t = np.array([
            box.xyxy[0].tolist() + [box.conf[0].item(), box.cls[0].item()]
            for res in results_t for box in res.boxes
        ]) if results_t[0].boxes else np.empty((0, 6))
        
        # Tracking
        tracks_b = tracker_b.update(detections_b, frame_b.shape[:2]) if detections_b.size > 0 else []
        tracks_t = tracker_t.update(detections_t, frame_t.shape[:2]) if detections_t.size > 0 else []
        
        # Feature extraction with validation
        features_b, bboxes_b = {}, {}
        for track in tracks_b:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            if x2 > x1 and y2 > y1:
                patch = frame_b[y1:y2, x1:x2]
                features = extract_visual_features(patch)
                if features is not None:
                    features_b[track_id] = features
                    bboxes_b[track_id] = (x1, y1, x2, y2)
        
        features_t, bboxes_t = {}, {}
        for track in tracks_t:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            if x2 > x1 and y2 > y1: 
                patch = frame_t[y1:y2, x1:x2]
                features = extract_visual_features(patch)
                if features is not None:
                    features_t[track_id] = features
                    bboxes_t[track_id] = (x1, y1, x2, y2)

        frame_matches = {}
        if features_b and features_t:
            broadcast_ids = list(features_b.keys())
            tacticam_ids = list(features_t.keys())
            
            # Calculate cost matrix
            feature_matrix_b = np.array([features_b[id] for id in broadcast_ids])
            feature_matrix_t = np.array([features_t[id] for id in tacticam_ids])
            cost_matrix = distance.cdist(feature_matrix_b, feature_matrix_t, 'cosine')
            
            # Hungarian algorithm for optimal assignment
            if cost_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Apply threshold and create matches
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < MATCH_THRESHOLD:
                        frame_matches[broadcast_ids[r]] = tacticam_ids[c]

        stable_matches = matching_system.update_matches(frame_matches)
        
        # Visualization
        frame_t_resized = cv2.resize(frame_t, (w_t_scaled, h_b))
        output_frame = np.zeros((h_b, output_width, 3), dtype=np.uint8)
        output_frame[:, :w_b] = frame_b
        output_frame[:, w_b:] = frame_t_resized
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, (b_id, t_id) in enumerate(stable_matches.items()):
            color = colors[i % len(colors)]
            
            if b_id in bboxes_b:
                x1_b, y1_b, x2_b, y2_b = bboxes_b[b_id]
                center_b = ((x1_b + x2_b) // 2, (y1_b + y2_b) // 2)
                cv2.rectangle(output_frame, (x1_b, y1_b), (x2_b, y2_b), color, 2)
                cv2.putText(output_frame, f"B:{b_id}", (x1_b, y1_b - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if t_id in bboxes_t:
                x1_t, y1_t, x2_t, y2_t = bboxes_t[t_id]
                x1_t_s = int(x1_t * scale_factor) + w_b
                y1_t_s = int(y1_t * scale_factor)
                x2_t_s = int(x2_t * scale_factor) + w_b
                y2_t_s = int(y2_t * scale_factor)
                
                center_t = ((x1_t_s + x2_t_s) // 2, (y1_t_s + y2_t_s) // 2)
                cv2.rectangle(output_frame, (x1_t_s, y1_t_s), (x2_t_s, y2_t_s), color, 2)
                cv2.putText(output_frame, f"T:{t_id}", (x1_t_s, y1_t_s - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if b_id in bboxes_b:
                    cv2.line(output_frame, center_b, center_t, color, 2)
        
        video_writer.write(output_frame)
    
    # Cleanup
    print("Processing complete.")
    cap_broadcast.release()
    cap_tacticam.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_videos()
