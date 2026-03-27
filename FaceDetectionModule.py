import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    def __init__(self, model_path='blaze_face_short_range.tflite', min_detection_con=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=min_detection_con
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self.results = None

    def find_faces(self, img, draw=True, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        
        try:
            self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            # Handle timestamps out of order or other issues
            return img, []

        bboxs = []
        if self.results and self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                
                score = 0
                if detection.categories:
                    score = detection.categories[0].score
                
                bbox_tuple = (x, y, w, h)
                bboxs.append([id, bbox_tuple, score])
                
                if draw:
                    self.fancy_draw(img, bbox_tuple)
                    cv2.putText(img, f'{int(score * 100)}%',
                                (x, y - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
        return img, bboxs

    def fancy_draw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        # Top Left
        cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y + l), (0, 255, 0), t)
        # Top Right
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 255, 0), t)
        # Bottom Left
        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)
        # Bottom Right
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 0), t)
        return img
