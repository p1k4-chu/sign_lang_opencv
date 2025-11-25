import numpy as np
import mediapipe as mp

# Define HandLandmark here so it can be used inside the function
mp_hands = mp.solutions.hands
HandLandmark = mp_hands.HandLandmark

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

def extract_features(lm: np.ndarray):
    """
    Processes a landmark array (21, 3) to extract normalized coordinates.
    Returns: flattened array of shape (63,)
    """
    # 1. Translation: Center around WRIST (Point 0)
    wrist_idx = int(HandLandmark.WRIST)
    lm_translated = lm - lm[wrist_idx]

    # 2. Rotation: Align hand to axes
    # Use Integer indices to be safe with numpy
    middle_mcp = int(HandLandmark.MIDDLE_FINGER_MCP)
    index_mcp = int(HandLandmark.INDEX_FINGER_MCP)
    pinky_mcp = int(HandLandmark.PINKY_MCP)

    # Y-axis: Wrist to Middle Finger MCP
    y_vec = lm_translated[middle_mcp]
    y_axis = y_vec / (np.linalg.norm(y_vec) + 1e-9)
    
    # Z-axis: Perpendicular to palm (Index MCP to Pinky MCP cross Y)
    palm_span = lm_translated[index_mcp] - lm_translated[pinky_mcp]
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