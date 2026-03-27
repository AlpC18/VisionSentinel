import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseDetector:
    def __init__(self, model_path='pose_landmarker.task', min_detection_con=0.5, min_tracking_con=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_con,
            min_tracking_confidence=min_tracking_con,
            num_poses=1
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.results = None

    def find_pose(self, img, draw=True, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        
        try:
            self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            return img

        if self.results and self.results.pose_landmarks:
            for pose_lms in self.results.pose_landmarks:
                if draw:
                    self.draw_pose_skeleton(img, pose_lms)
        return img

    def draw_pose_skeleton(self, img, landmarks):
        h, w, c = img.shape
        points = []
        # Draw all 33 landmarks
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
        
        # Define connections (Standard Pose)
        # Simplified set of connections for body
        connections = [
            (11, 12), (11, 13), (13, 15),       # Left Arm
            (12, 14), (14, 16),                 # Right Arm
            (11, 23), (12, 24), (23, 24),       # Torso
            (23, 25), (25, 27),                 # Left Leg
            (24, 26), (26, 28)                  # Right Leg
        ]
        
        for p1, p2 in connections:
            if p1 < len(points) and p2 < len(points):
                cv2.line(img, points[p1], points[p2], (255, 255, 255), 2)
