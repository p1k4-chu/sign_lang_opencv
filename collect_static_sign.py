import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# --------------------
# Configuration
# --------------------
DATA_PATH = Path("dataset/static")

# Smoothing factor for landmarks. 0 = no smoothing, 1 = no change.
SMOOTHING_FACTOR = 0.85

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
HandLandmark = mp_hands.HandLandmark

# --------------------
# Smoothing Class
# --------------------
class Smoother:
    """Applies exponential moving average smoothing to landmarks."""
    def __init__(self, alpha=0.85):
        self.alpha = alpha
        self.smoothed_landmarks = None

    def __call__(self, landmarks):
        if self.smoothed_landmarks is None:
            self.smoothed_landmarks = landmarks
        else:
            self.smoothed_landmarks = self.alpha * landmarks + (1 - self.alpha) * self.smoothed_landmarks
        return self.smoothed_landmarks

# --------------------
# Feature Extraction
# --------------------
def extract_features(lm: np.ndarray):
    """
    Processes a landmark array (21, 3) to extract normalized coordinates.
    Performs Translation, Rotation, and Scale normalization.
    Returns: flattened array of shape (63,)
    """
    # 1. Translation: Center around WRIST
    lm_translated = lm - lm[HandLandmark.WRIST]

    # 2. Rotation: Align hand to axes
    # Y-axis: Wrist to Middle Finger MCP
    y_axis = lm_translated[HandLandmark.MIDDLE_FINGER_MCP] / (np.linalg.norm(lm_translated[HandLandmark.MIDDLE_FINGER_MCP]) + 1e-9)
    
    # Z-axis: Perpendicular to palm (Index MCP to Pinky MCP cross Y)
    palm_span = lm_translated[HandLandmark.INDEX_FINGER_MCP] - lm_translated[HandLandmark.PINKY_MCP]
    z_axis = np.cross(y_axis, palm_span)
    z_axis /= (np.linalg.norm(z_axis) + 1e-9)
    
    # X-axis: Perpendicular to Y and Z
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= (np.linalg.norm(x_axis) + 1e-9)
    
    # Apply rotation
    rotation_matrix = np.array([x_axis, y_axis, z_axis])
    lm_rotated = np.dot(lm_translated, rotation_matrix.T)

    # 3. Scale: Normalize so max distance is 1.0
    max_dist = np.max(np.linalg.norm(lm_rotated, axis=1)) + 1e-9
    lm_norm = lm_rotated / max_dist
    
    # Return only the flattened normalized landmarks (63 values)
    return lm_norm.flatten().astype(np.float32)

# --------------------
# Main Collection Logic
# --------------------
def main():
    print("Collecting 63-D samples (21 landmarks * 3 coords)")
    
    classes_to_record = input("\nEnter classes to record, separated by comma (e.g., A,B,C): ")
    classes = [c.strip().upper() for c in classes_to_record.split(",") if c.strip()]

    for cls in classes:
        (DATA_PATH / cls).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize the smoother
    landmark_smoother = Smoother(alpha=SMOOTHING_FACTOR)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        for cls in classes:
            print(f"\nReady to record class '{cls}'. Press 'r' to capture, 'q' to skip.")
            sample_count = len(list((DATA_PATH / cls).glob('*.npy')))
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks_proto = results.multi_hand_landmarks[0]
                    
                    # Convert landmarks to a NumPy array
                    raw_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks_proto.landmark], dtype=np.float32)
                    
                    # Apply smoothing
                    smoothed_landmarks = landmark_smoother(raw_landmarks)
                    
                    # Draw the smoothed landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS)
                
                else:
                    # If no hand is detected, reset the smoother
                    landmark_smoother.smoothed_landmarks = None

                cv2.putText(frame, f"Class: {cls}  Samples: {sample_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, "r: record | s: verify (console) | q: next class", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Data Collection", frame)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'): break
                
                if k in [ord('r'), ord('s')]:
                    if 'smoothed_landmarks' in locals() and smoothed_landmarks is not None:
                        # Extract only normalized landmarks
                        features = extract_features(smoothed_landmarks)
                        
                        if k == ord('r'):
                            filepath = DATA_PATH / cls / f"{cls}_{sample_count:04d}.npy"
                            np.save(filepath, features)
                            sample_count += 1
                            print(f"Saved {filepath} (Shape: {features.shape})")
                        elif k == ord('s'):
                            print(f"--- Captured Frame ---")
                            print(f"Feature vector shape: {features.shape}")
                            print(f"First 5 values: {features[:5]}")
                    else:
                        print("No hand detected. Cannot capture.")
            
            if k == ord('q'):
                print(f"Skipping class '{cls}'...")
                continue

    cap.release()
    cv2.destroyAllWindows()
    print("Collection finished. 👍")

if __name__ == "__main__":
    main()