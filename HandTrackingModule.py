import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, model_path='hand_landmarker.task', num_hands=2, min_detection_con=0.5, min_tracking_con=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_con,
            min_hand_presence_confidence=min_detection_con,
            min_tracking_confidence=min_tracking_con
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, draw=True, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        
        try:
            self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Hand Tracking Error: {e}")
            self.results = None
            return img

        if self.results and self.results.hand_landmarks:
            for hand_lms in self.results.hand_landmarks:
                if draw:
                    # self.draw_hand_skeleton(img, hand_lms) # Eski çizim
                    self.draw_cyber_mesh(img, hand_lms) # Yeni "Cyber" çizim
        return img

    def draw_cyber_mesh(self, img, landmarks):
        h, w, c = img.shape
        points = []
        # Landmarkleri al
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
        
        # Overlay katmanı oluştur (Transparanlık için)
        overlay = img.copy()
        
        # 1. Standart İskelet Bağlantıları
        skeleton_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),         # Index
            (9, 10), (10, 11), (11, 12),            # Middle
            (13, 14), (14, 15), (15, 16),           # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)               # Palm base (Knuckles)
        ]
        
        # 2. "Web" Ağı Bağlantıları (Parmaklar arası)
        web_connections = [
            (4, 8), (8, 12), (12, 16), (16, 20),   # Parmak uçları
            (3, 7), (7, 11), (11, 15), (15, 19),   # Üst eklemler
            (2, 6), (6, 10), (10, 14), (14, 18),   # Alt eklemler
            (5, 9), (9, 13), (13, 17)              # Ana eklemler (zaten var ama vurgu için)
        ]

        # RENKLER (BGR Formatında)
        NEON_PURPLE = (255, 0, 255)  # Magenta/Mor
        DEEP_PURPLE = (128, 0, 128)
        CYBER_BLUE = (255, 255, 0)   # Cyan (Opencv'de BGR -> 255,255,0 = Cyan mı? Hayır, (255,255,0) Teal/Cyan kirmizi yok. Mavi+Yesil)
        GLOW_COLOR = (200, 50, 200)

        # Çizim - İskelet (Kalın ve Parlak)
        for p1, p2 in skeleton_connections:
            cv2.line(overlay, points[p1], points[p2], NEON_PURPLE, 4) # Kalın çizgi
            
        # Çizim - Ağ (Daha ince)
        for p1, p2 in web_connections:
            cv2.line(overlay, points[p1], points[p2], (200, 100, 255), 1) # İnce ağ çizgisi

        # Alpha Blending (Saydamlık ekle)
        alpha = 0.6 # %60 görünürlük (Hologram efekti)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Eklemler (Noktalar) - Bunları blend etme, üstte parlasın
        for id, (cx, cy) in enumerate(points):
            # Dış halka
            cv2.circle(img, (cx, cy), 6, NEON_PURPLE, cv2.FILLED)
            # İç nokta (Beyaz parıltı)
            cv2.circle(img, (cx, cy), 2, (255, 255, 255), cv2.FILLED)

    def draw_hand_skeleton(self, img, landmarks):
        # ... (Eski fonksiyonu backup olarak tutabiliriz veya silebiliriz)
        pass

    def find_position(self, img, hand_no=0):
        lm_list = []
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                my_hand = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
        return lm_list