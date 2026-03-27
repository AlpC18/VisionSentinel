import cv2
import time
import numpy as np
from HandTrackingModule import HandDetector
from FaceDetectionModule import FaceDetector
from PoseDetectionModule import PoseDetector
from NotificationModule import NotificationSystem # YENI
from ObjectDetectionModule import ObjectDetector # YENI NESNE TANIMA

# --- YARDIMCI FONSİYONLAR ---
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_zoom(img, zoom_factor=1.0):
    if zoom_factor == 1.0: return img
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    radius_x, radius_y = int(w / (2 * zoom_factor)), int(h / (2 * zoom_factor))
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y
    min_x, max_x = max(0, min_x), min(w, max_x)
    min_y, max_y = max(0, min_y), min(h, max_y)
    cropped = img[min_y:max_y, min_x:max_x]
    return cv2.resize(cropped, (w, h))

def apply_thermal_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal

def init_camera(camera_index):
    print(f"Kamera başlatılıyor: Index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened(): return None
    return cap

# --- ANA PROGRAM ---
def main():
    current_cam_index = 0
    cap = init_camera(current_cam_index)
    
    if cap is None or not cap.isOpened():
        current_cam_index = 1
        cap = init_camera(current_cam_index)

    # MODELLER
    hand_detector = HandDetector(model_path='hand_landmarker.task', num_hands=4, min_detection_con=0.3, min_tracking_con=0.3)
    face_detector = FaceDetector(model_path='blaze_face_short_range.tflite', min_detection_con=0.3)
    pose_detector = PoseDetector(model_path='pose_landmarker.task', min_detection_con=0.3, min_tracking_con=0.3)
    object_detector = ObjectDetector()
    
    # BILDIRIM SISTEMI (YENI)
    notifier = NotificationSystem()

    avg_frame = None
    motion_status = "Durgun"
    
    p_time = 0
    gamma_value = 1.0 
    zoom_level = 1.0

    current_mode = 0 
    dark_mode_boost = False 
    
    # GUVENLIK MODU: Eger True ise, insan algilandiginda SMS atar
    security_active = False 

    print("-------------------------------------------------")
    print(" SİSTEM BAŞLATILDI v5")
    print(" 's': GÜVENLİK/ALARM MODU (AÇ/KAPA) -> [Bildirim Atar]")
    print(" 't': TERMAL | 'm': HAREKET | 'z': ZOOM | 'o': NESNE TANIMA")
    print(" 'c': KAMERA | 'q': ÇIKIŞ")
    print("-------------------------------------------------")

    while True:
        if cap is None:
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        success, img = cap.read()
        if not success:
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        img = apply_zoom(img, zoom_factor=zoom_level)
        
        # Orijinal (Detection) Image
        detection_img = img.copy() 

        # Display Image
        if current_mode == 2:
            display_img = apply_thermal_effect(img)
        elif current_mode == 3:
            display_img, det_labels = object_detector.process(img)
        else:
            if dark_mode_boost:
                img = adjust_gamma(img, gamma=2.0)
                detection_img = img.copy() 
            display_img = img.copy()

        # --- AI ALGORITMALARI ---
        timestamp_ms = int(time.time() * 1000)
        
        # 1. AI Taramasi
        pose_detector.find_pose(detection_img, draw=False, timestamp_ms=timestamp_ms)
        face_detector.find_faces(detection_img, draw=False, timestamp_ms=timestamp_ms)
        hand_detector.find_hands(detection_img, draw=False, timestamp_ms=timestamp_ms)
        
        detected_types = []
        if current_mode == 3:
            for lbl in det_labels:
                detected_types.append(lbl.split(' ')[0].upper())
        
        # 2. Cizim ve Kontrol
        
        # Hands
        if hand_detector.results and hand_detector.results.hand_landmarks:
            detected_types.append("EL")
            for hand_lms in hand_detector.results.hand_landmarks:
                hand_detector.draw_cyber_mesh(display_img, hand_lms)
                
        # Faces
        if face_detector.results and face_detector.results.detections:
             detected_types.append("YUZ")
             for detection in face_detector.results.detections:
                bbox = detection.bounding_box
                face_detector.fancy_draw(display_img, (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))
        
        # Pose
        if pose_detector.results and pose_detector.results.pose_landmarks:
            detected_types.append("INSAN")
            for pose_lms in pose_detector.results.pose_landmarks:
                pose_detector.draw_pose_skeleton(display_img, pose_lms)

        # 3. MOTION SENSOR (Motion Modunda zaten ciziyoruz, ama Security modu icin de kontrol edelim)
        motion_detected = False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if avg_frame is None: avg_frame = gray.copy().astype("float")
        cv2.accumulateWeighted(gray, avg_frame, 0.05)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
        thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if current_mode == 1:
            motion_status = "Durgun"
            for c in cnts:
                 if cv2.contourArea(c) < 100: continue
                 motion_status = "HAREKET"
                 motion_detected = True
                 (x, y, w, h) = cv2.boundingRect(c)
                 cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # --- GUVENLIK / BILDIRIM LOGIG ---
        if security_active:
            # Alarm Condition:
            # Eger AI bir sey bulduysa VE (Karanlik moddaysa ve insan goruyorsa daha tehlikeli olabilir ama genel kural su an: Herhangi bir insan aktivitesi)
            if len(detected_types) > 0:
                alert_msg = f"Kamera hareket algiladi! Tespit edilen: {', '.join(set(detected_types))}"
                sent = notifier.send_alert(alert_msg)
                if sent:
                    print(f"ALARM GONDERILDI: {alert_msg}")
                    # Ekrana da yazalim
                    cv2.putText(display_img, "ALARM SENT!", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        # HUD
        cv2.rectangle(display_img, (10, 10), (420, 300), (0, 0, 0), cv2.FILLED)
        cv2.putText(display_img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        mode_str = "NORMAL"
        if current_mode == 1: mode_str = "MOTION SENSOR"
        if current_mode == 2: mode_str = "THERMAL VISION"
        if current_mode == 3: mode_str = "OBJECT DETECTION"
        cv2.putText(display_img, f'MODE: {mode_str}', (20, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        
        # Security Status
        sec_color = (0, 0, 255) if security_active else (100, 100, 100)
        sec_text = "ARMED (GUVENLIK ACIK)" if security_active else "DISARMED (KAPALI)"
        cv2.putText(display_img, f'SECURITY (s): {sec_text}', (20, 130), cv2.FONT_HERSHEY_PLAIN, 1.3, sec_color, 2)
        
        # Notification Status
        if security_active:
             if not notifier.enabled:
                  cv2.putText(display_img, "Warning: No Twilio Config!", (20, 160), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)
             else:
                  cv2.putText(display_img, "Ready to SMS...", (20, 160), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
        
        if zoom_level > 1.0:
            cv2.putText(display_img, f'ZOOM: {zoom_level}x', (20, 200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        cv2.imshow("Super Sense Tracker", display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('m'): current_mode = 1
        elif key == ord('o'): current_mode = 3 if current_mode != 3 else 0
        elif key == ord('t'): current_mode = 2 if current_mode != 2 else 0
        elif key == ord('n'): dark_mode_boost = not dark_mode_boost
        elif key == ord('s'): 
            security_active = not security_active
            if security_active and not notifier.enabled:
                print("UYARI: Guvenlik modu acildi ama NotificationModule.py icinde Twilio ayarlari yapilmamis!")
        elif key == ord('z'):
            if zoom_level == 1.0: zoom_level = 1.5
            elif zoom_level == 1.5: zoom_level = 2.0
            else: zoom_level = 1.0
        elif key == ord('c'):
            cap.release()
            current_cam_index = (current_cam_index + 1) % 2
            cap = init_camera(current_cam_index)

    if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()