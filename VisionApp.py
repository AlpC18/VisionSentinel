import cv2
import numpy as np
import time
import math
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

try:
    from NotificationModule import NotificationSystem
except ImportError:
    NotificationSystem = None

try:
    from HandTrackingModule import HandDetector
    from PoseDetectionModule import PoseDetector
except ImportError:
    HandDetector = None
    PoseDetector = None

try:
    import supervision as sv
    from ultralytics import YOLO
except ImportError:
    print("Lütfen terminalinizde (venv aktifken) aşağıdaki komutu çalıştırarak gerekli kütüphaneleri yükleyin:")
    print("pip install supervision ultralytics")
    sv = None
    YOLO = None

def draw_transparent_bg(img, x, y, w, h, color=(0,0,0), alpha=0.5):
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
    
    sub_img = img[y1:y2, x1:x2]
    if sub_img.size > 0:
        rect = np.full(sub_img.shape, color, dtype=np.uint8)
        cv2.addWeighted(rect, alpha, sub_img, 1 - alpha, 0, sub_img)

def draw_beautiful_text(img, text, pos, font_scale=0.7, text_color=(255, 255, 255), bg_color=(0,0,0), alpha=0.6, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    draw_transparent_bg(img, max(0, x - 10), max(0, y - th - 10), tw + 20, th + 15 + baseline, bg_color, alpha)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def get_menu_rects(width):
    buttons = [
        {"id": 1, "type": "mode", "text": "1. Object Detection", "rect": (width - 210, 75, 200, 35)},
        {"id": 2, "type": "mode", "text": "2. Line Counter", "rect": (width - 210, 115, 200, 35)},
        {"id": 3, "type": "mode", "text": "3. Skeleton & Hands", "rect": (width - 210, 155, 200, 35)},
        {"id": 4, "type": "mode", "text": "4. Thermal Vision", "rect": (width - 210, 195, 200, 35)},
        {"id": 5, "type": "mode", "text": "5. Night Vision", "rect": (width - 210, 235, 200, 35)},
        {"id": 6, "type": "mode", "text": "6. Face Detection", "rect": (width - 210, 275, 200, 35)},
        {"id": 7, "type": "mode", "text": "7. Heatmap Track", "rect": (width - 210, 315, 200, 35)},
        {"id": 8, "type": "mode", "text": "8. Time in Zone", "rect": (width - 210, 355, 200, 35)},
        {"id": 9, "type": "mode", "text": "9. Fast Tracker", "rect": (width - 210, 395, 200, 35)},
        {"id": 0, "type": "mode", "text": "0. Distance/Depth", "rect": (width - 210, 435, 200, 35)},
        
        {"id": 'switch', "type": "action", "text": "S. Switch Target Cam", "rect": (width - 420, 75, 200, 35)},
        {"id": 'c', "type": "action", "text": "C. Screenshot", "rect": (width - 420, 115, 200, 35)},
        {"id": 'r', "type": "action", "text": "R. Record Video", "rect": (width - 420, 155, 200, 35)},
        {"id": 'z', "type": "action", "text": "Z. Zoom Toggle", "rect": (width - 420, 195, 200, 35)},
        {"id": 'a', "type": "action", "text": "A. Security Mode", "rect": (width - 420, 235, 200, 35)},
        {"id": 'f', "type": "action", "text": "F. Auto-Framing", "rect": (width - 420, 275, 200, 35)}
    ]
    return buttons

# Global Durum
cam_modes = {0: 1, 1: 5} 
menu_target_cam = 0      
zoom_factors = {0: 1.0, 1: 1.0}
auto_framings = {0: False, 1: False}
fast_tracker_initialized = {0: False, 1: False}
trackers = {0: None, 1: None}
menu_expanded = False 
gui_action = None
menu_selected_idx = 0
security_armed = False
is_recording = False

def mouse_callback(event, x, y, flags, param):
    global menu_expanded, gui_action, menu_selected_idx, menu_target_cam
    
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
        width = param['width']
        
        opt_rx, opt_ry, opt_rw, opt_rh = width - 160, 20, 140, 45
        if opt_rx <= x <= opt_rx + opt_rw and opt_ry <= y <= opt_ry + opt_rh:
            menu_expanded = not menu_expanded
            return

        if menu_expanded:
            buttons = get_menu_rects(width)
            for i, btn in enumerate(buttons):
                rx, ry, rw, rh = btn['rect']
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    menu_selected_idx = i
                    if btn["type"] == "mode":
                        if cam_modes[menu_target_cam] != btn['id']:
                            cam_modes[menu_target_cam] = btn['id']
                            menu_expanded = False
                            
                            if cam_modes[menu_target_cam] == 9:
                                fast_tracker_initialized[menu_target_cam] = False
                                try: trackers[menu_target_cam] = cv2.TrackerCSRT_create()
                                except:
                                    try: trackers[menu_target_cam] = cv2.legacy.TrackerCSRT_create()
                                    except: trackers[menu_target_cam] = cv2.TrackerKCF_create()
                    elif btn["type"] == "action":
                        gui_action = btn['id']
                        menu_expanded = False
                    break


def main_app(sources):
    if isinstance(sources, str):
        sources = [sources]
        
    global menu_expanded, gui_action, menu_selected_idx, menu_target_cam
    global security_armed, is_recording
    
    model = None
    byte_tracker = None
    box_annotator = None
    label_annotator = None
    heatmap_annotator = None
    if YOLO is not None:
        model = YOLO("yolov8n.pt")
        byte_tracker = sv.ByteTrack() 
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        try:
            heatmap_annotator = sv.HeatMapAnnotator(
                position=sv.Position.BOTTOM_CENTER, opacity=0.6, radius=25, kernel_size=31, top_hue=0, low_hue=120
            )
        except: pass
            
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    hand_detector = HandDetector() if HandDetector is not None else None
    pose_detector = PoseDetector() if PoseDetector is not None else None
    
    captures = []
    for src in sources:
        cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        if cap.isOpened(): captures.append(cap)
            
    if not captures:
        print("Hiçbir kamera açılamadı!")
        return
        
    width = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0: width, height = 1280, 720
        
    cv2.namedWindow("Cv_Projem - Gelismis Arayuz")
    cv2.setMouseCallback("Cv_Projem - Gelismis Arayuz", mouse_callback, param={'width': width})
    
    line_zone, line_zone_annotator, poly_zone, poly_zone_annotator = None, None, None, None
    if sv is not None:
        line_zone = sv.LineZone(start=sv.Point(0, height // 2), end=sv.Point(width, height // 2))
        line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)
        
        polygon_area = np.array([[int(width * 0.2), int(height * 0.2)], [int(width * 0.8), int(height * 0.2)],
                                 [int(width * 0.8), int(height * 0.8)], [int(width * 0.2), int(height * 0.8)]])
        poly_zone = sv.PolygonZone(polygon=polygon_area, triggering_anchors=(sv.Position.BOTTOM_CENTER,))
        try: poly_zone_annotator = sv.PolygonZoneAnnotator(zone=poly_zone, color=sv.Color.RED, thickness=2, text_color=sv.Color.WHITE)
        except: poly_zone_annotator = sv.PolygonZoneAnnotator(zone=poly_zone, color=sv.Color.white(), thickness=2, text_thickness=2, text_scale=1)
            
    tracker_entry_times = {0: {}, 1: {}} 
    fps_monitor = cv2.TickMeter()

    af_states = {0: {'cx': width//2, 'cy': height//2, 'scale': 1.0}, 1: {'cx': width//2, 'cy': height//2, 'scale': 1.0}}
    af_smoothing = 0.1
    
    notifier = NotificationSystem() if NotificationSystem is not None else None
    nv_sweep_ys = {0: 0, 1: 0}
    nv_prev_greens = {0: None, 1: None}
    
    video_writer = None
    
    # Kayıt Klasörlerini Oluştur
    os.makedirs("Vault/Photos", exist_ok=True)
    os.makedirs("Vault/Videos", exist_ok=True)
    os.makedirs("Vault/Security", exist_ok=True)

    while True:
        frames_orig = []
        for cap in captures:
            ret, f = cap.read()
            if ret: frames_orig.append(f)
            else: frames_orig.append(None)
                
        if len(frames_orig) == 0 or frames_orig[0] is None:
            break
            
        annotated_frames = []
        fps_monitor.start()
        any_found_objects = False

        for c_idx, frame_orig in enumerate(frames_orig):
            if frame_orig is None:
                annotated_frames.append(None)
                continue
                
            cur_mode = cam_modes.get(c_idx, 1)
            zf = zoom_factors.get(c_idx, 1.0)
            af = auto_framings.get(c_idx, False)
            
            # --- ZOM ---
            if zf > 1.0 and not af:
                ho, wo = frame_orig.shape[:2]
                cx, cy = wo // 2, ho // 2
                rx, ry = int(wo / (2 * zf)), int(ho / (2 * zf))
                cropped = frame_orig[max(0, cy-ry):min(ho, cy+ry), max(0, cx-rx):min(wo, cx+rx)]
                frame_orig = cv2.resize(cropped, (wo, ho))
                
            ho, wo = frame_orig.shape[:2]

            # --- AUTO-FRAMING ---
            target_bbox = None
            if af and model is not None:
                res = model.predict(frame_orig, classes=[0], conf=0.4, verbose=False, max_det=1)
                bboxes = res[0].boxes.xyxy.cpu().numpy()
                if len(bboxes) > 0: target_bbox = bboxes[0]

            if af:
                tcx, tcy = wo // 2, ho // 2
                ts = 1.0
                if target_bbox is not None:
                    bx1, by1, bx2, by2 = target_bbox
                    bw, bh = bx2 - bx1, by2 - by1
                    tcx = int(bx1 + bw / 2)
                    tcy = int(by1 + bh / 2 - bh * 0.1)
                    desired_vh = bh * 2.0 
                    ts = ho / desired_vh
                    if ts < 1.0: ts = 1.0
                    if ts > 3.0: ts = 3.0
                    
                st = af_states[c_idx]
                st['cx'] += (tcx - st['cx']) * af_smoothing
                st['cy'] += (tcy - st['cy']) * af_smoothing
                st['scale'] += (ts - st['scale']) * af_smoothing
                
                rx = int(wo / (2 * st['scale']))
                ry = int(ho / (2 * st['scale']))
                cx_smooth, cy_smooth = int(st['cx']), int(st['cy'])
                
                if cx_smooth - rx < 0: cx_smooth = rx
                if cx_smooth + rx > wo: cx_smooth = wo - rx
                if cy_smooth - ry < 0: cy_smooth = ry
                if cy_smooth + ry > ho: cy_smooth = ho - ry
                
                cropped_af = frame_orig[cy_smooth-ry : cy_smooth+ry, cx_smooth-rx : cx_smooth+rx]
                if cropped_af.size > 0:
                    frame_orig = cv2.resize(cropped_af, (wo, ho))
                    ho, wo = frame_orig.shape[:2]

            frame = frame_orig.copy()
            annotated_frame = frame_orig.copy()
            found_objects = False

            if cur_mode in [0, 1, 2, 7, 8] and model is not None:
                result = model(frame, agnostic_nms=True, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.confidence > 0.4]
                if len(detections) > 0: found_objects = True
                
                if cur_mode == 1:
                    labels = [f"{model.names[cid]} {conf:.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                    
                elif cur_mode == 2:
                    detections = byte_tracker.update_with_detections(detections)
                    line_zone.trigger(detections=detections)
                    labels = [f"#{tid} {model.names[cid]}" for cid, tid in zip(detections.class_id, detections.tracker_id)]
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                    line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
                    
                elif cur_mode == 7:
                    detections = byte_tracker.update_with_detections(detections)
                    if heatmap_annotator is not None:
                        try: heatmap_annotator.annotate(annotated_frame, detections=detections)
                        except: pass
                        
                elif cur_mode == 8:
                    detections = byte_tracker.update_with_detections(detections)
                    in_zone = poly_zone.trigger(detections=detections)
                    labels = []
                    for is_inside, class_id, tracker_id in zip(in_zone, detections.class_id, detections.tracker_id):
                        if is_inside:
                            if tracker_id not in tracker_entry_times[c_idx]:
                                tracker_entry_times[c_idx][tracker_id] = time.time()
                            elapsed = time.time() - tracker_entry_times[c_idx][tracker_id]
                            labels.append(f"#{tracker_id} {model.names[class_id]} ({int(elapsed)}s)")
                        else:
                            if tracker_id in tracker_entry_times[c_idx]: del tracker_entry_times[c_idx][tracker_id]
                            labels.append(f"#{tracker_id} {model.names[class_id]}")
                            
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                    try: sv.draw_polygon(annotated_frame, polygon_area, sv.Color.RED)
                    except: poly_zone_annotator.annotate(scene=annotated_frame)
                        
                elif cur_mode == 0:
                    FOCAL_LENGTH = 700 
                    for xyxy, class_id in zip(detections.xyxy, detections.class_id):
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                        px_h, px_w = y2 - y1, x2 - x1
                        if px_h == 0 or px_w == 0: continue
                        dist_text = ""
                        if class_id == 0:
                            dist = (1.70 * FOCAL_LENGTH) / px_h
                            dist_text = f"Distance: {dist:.1f}m"
                        elif class_id == 2:
                            dist = (1.80 * FOCAL_LENGTH) / px_w
                            dist_text = f"Distance: {dist:.1f}m"

                        if dist_text:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 150, 50), 2)
                            draw_beautiful_text(annotated_frame, dist_text, (x1, max(20, y1 - 10)), 0.6, (50, 255, 255), (0,0,0), 0.7, 2)
                            cv2.line(annotated_frame, (wo//2, ho), (x1 + px_w//2, y2), (255, 100, 50), 1)
                            cv2.circle(annotated_frame, (x1 + px_w//2, y2), 5, (50, 255, 255), -1)

            elif cur_mode == 3:
                if pose_detector: annotated_frame = pose_detector.find_pose(annotated_frame)
                if hand_detector: annotated_frame = hand_detector.find_hands(annotated_frame)

            elif cur_mode == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)
                
                blur = cv2.GaussianBlur(gray_clahe, (15, 15), 0)
                thermal = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
                annotated_frame = thermal
                
                cx, cy = wo // 2, ho // 2
                
                # Temperature Scale Bar
                cv2.rectangle(annotated_frame, (wo - 40, ho // 4), (wo - 20, ho * 3 // 4), (255, 255, 255), 2)
                for i in range(ho // 4, ho * 3 // 4, 2):
                    val = 255 - int(((i - ho // 4) / ((ho * 3 // 4) - (ho // 4))) * 255)
                    val = max(0, min(255, val))
                    col = cv2.applyColorMap(np.array([[[val]]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
                    col = (int(col[0]), int(col[1]), int(col[2]))
                    cv2.line(annotated_frame, (wo - 39, i), (wo - 21, i), col, 2)
                
                cv2.putText(annotated_frame, "45C", (wo - 80, ho // 4 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(annotated_frame, "10C", (wo - 80, ho * 3 // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                center_heat = gray_clahe[cy, cx]
                estimated_temp = 10.0 + (center_heat / 255.0) * 35.0 
                
                cv2.drawMarker(annotated_frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                if estimated_temp > 35.0:
                    draw_beautiful_text(annotated_frame, f"BODY HEAT: {estimated_temp:.1f}C", (cx + 20, cy - 20), 0.6, (0, 0, 255))
                else:
                    draw_beautiful_text(annotated_frame, f"TARGET TEMP: {estimated_temp:.1f}C", (cx + 20, cy - 20), 0.6, (255, 255, 255))
                cv2.putText(annotated_frame, "THERMAL IMAGING ACTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2, cv2.LINE_AA)

            elif cur_mode == 5:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
                cl = clahe.apply(gray)
                lap = cv2.Laplacian(cl, cv2.CV_16S, ksize=3)
                lap_abs = cv2.convertScaleAbs(lap)
                cl = np.clip(cl.astype(np.int16) + lap_abs.astype(np.int16) * 0.45, 0, 255).astype(np.uint8)

                nv_frame = np.zeros((ho, wo, 3), dtype=np.uint8)
                nv_frame[:, :, 1] = cl
                bright_mask = np.where(cl > 170, cl, 0).astype(np.uint8)
                glow = cv2.GaussianBlur(bright_mask, (25, 25), 0)
                glow_3ch = np.zeros((ho, wo, 3), dtype=np.uint8)
                glow_3ch[:, :, 1] = glow
                nv_frame = cv2.addWeighted(nv_frame, 1.0, glow_3ch, 0.55, 0)

                nv_p = nv_prev_greens[c_idx]
                if nv_p is not None and nv_p.shape == nv_frame.shape:
                    nv_frame = cv2.addWeighted(nv_frame, 0.82, nv_p, 0.18, 0)
                nv_prev_greens[c_idx] = nv_frame.copy()

                dark_weight = (1.0 - cl.astype(np.float32) / 255.0) * 0.65 + 0.35
                noise_base = np.random.randint(0, 55, (ho, wo)).astype(np.float32)
                noise_scaled = (noise_base * dark_weight).astype(np.uint8)
                nv_frame[:, :, 1] = np.clip(nv_frame[:, :, 1].astype(np.int16) + noise_scaled.astype(np.int16), 0, 255).astype(np.uint8)

                scanline_mask = np.ones((ho, wo, 3), dtype=np.float32)
                scanline_mask[::2, :] = 0.48 
                scanline_mask[1::4, :] = 0.72  
                nv_frame = (nv_frame.astype(np.float32) * scanline_mask).astype(np.uint8)

                vig_x = cv2.getGaussianKernel(wo, int(wo * 0.50))
                vig_y = cv2.getGaussianKernel(ho, int(ho * 0.50))
                vignette = (vig_y * vig_x.T)
                vignette = vignette / vignette.max()
                vignette_3ch = np.stack([vignette] * 3, axis=2).astype(np.float32)
                nv_frame = np.clip(nv_frame.astype(np.float32) * vignette_3ch * 1.75, 0, 255).astype(np.uint8)

                nv_frame = cv2.GaussianBlur(nv_frame, (3, 3), 0)

                nv_sweep_ys[c_idx] = (nv_sweep_ys[c_idx] + 4) % ho
                sweep_layer = np.zeros((ho, wo, 3), dtype=np.float32)
                for dy in range(-12, 13):
                    sy = (nv_sweep_ys[c_idx] + dy) % ho
                    intensity = max(0.0, 1.0 - abs(dy) / 12.0) * 55
                    sweep_layer[sy, :, 1] = intensity
                nv_frame = np.clip(nv_frame.astype(np.float32) + sweep_layer, 0, 255).astype(np.uint8)

                flicker = np.random.uniform(0.91, 1.0)
                nv_frame = np.clip(nv_frame.astype(np.float32) * flicker, 0, 255).astype(np.uint8)
                annotated_frame = nv_frame

                cx, cy = wo // 2, ho // 2
                grid_col = (0, 38, 0)
                for gx in range(0, wo, wo // 8): cv2.line(annotated_frame, (gx, 0), (gx, ho), grid_col, 1)
                for gy in range(0, ho, ho // 6): cv2.line(annotated_frame, (0, gy), (wo, gy), grid_col, 1)
                for r, alpha_col in [(60, (0, 45, 0)), (130, (0, 55, 0)), (220, (0, 45, 0))]:
                    cv2.circle(annotated_frame, (cx, cy), r, alpha_col, 1)

                cv2.line(annotated_frame, (cx - 90, cy), (cx - 22, cy), (0, 210, 0), 1)
                cv2.line(annotated_frame, (cx + 22, cy), (cx + 90, cy), (0, 210, 0), 1)
                cv2.line(annotated_frame, (cx, cy - 90), (cx, cy - 22), (0, 210, 0), 1)
                cv2.line(annotated_frame, (cx, cy + 22), (cx, cy + 90), (0, 210, 0), 1)
                cv2.circle(annotated_frame, (cx, cy), 18, (0, 195, 0), 1)
                cv2.circle(annotated_frame, (cx, cy), 4,  (0, 255, 0), -1)
                for angle_deg, r_in, r_out in [(45, 22, 35), (135, 22, 35), (225, 22, 35), (315, 22, 35)]:
                    rad = math.radians(angle_deg)
                    p1 = (int(cx + r_in  * math.cos(rad)), int(cy + r_in  * math.sin(rad)))
                    p2 = (int(cx + r_out * math.cos(rad)), int(cy + r_out * math.sin(rad)))
                    cv2.line(annotated_frame, p1, p2, (0, 150, 0), 1)

                clen, ct = 38, 2
                for (cpx, cpy, dx, dy) in [(15,15,1,1),(wo-15,15,-1,1),(15,ho-15,1,-1),(wo-15,ho-15,-1,-1)]:
                    cv2.line(annotated_frame, (cpx, cpy), (cpx + dx * clen, cpy), (0, 195, 0), ct)
                    cv2.line(annotated_frame, (cpx, cpy), (cpx, cpy + dy * clen), (0, 195, 0), ct)
                    cv2.circle(annotated_frame, (cpx, cpy), 3, (0, 255, 0), -1)

                bx, by = wo - 115, 14
                cv2.rectangle(annotated_frame, (bx, by), (bx + 68, by + 14), (0, 140, 0), 1)
                cv2.rectangle(annotated_frame, (bx + 2, by + 2), (bx + 60, by + 12), (0, 175, 0), -1)
                cv2.rectangle(annotated_frame, (bx + 68, by + 4), (bx + 73, by + 10), (0, 140, 0), -1)
                cv2.putText(annotated_frame, "PWR", (bx - 38, by + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 155, 0), 1, cv2.LINE_AA)

                ts = time.strftime('%H:%M:%S')
                cv2.putText(annotated_frame, f"NV-GEN4  ACTIVE [CAM-{c_idx+1}]",   (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 230, 0), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"TIME: {ts}",        (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 185, 0), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, "GAIN: AUTO | IR-ON", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 155, 0), 1, cv2.LINE_AA)

                cv2.putText(annotated_frame, "[[ NIGHT VISION ++ ]]", (cx - 115, ho - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 0), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"RES {wo}x{ho}  |  PHOSPHOR ENHANCED", (cx - 160, ho - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 155, 0), 1, cv2.LINE_AA)

                nv_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(28, 28))
                for (fx, fy, fw, fh) in nv_faces:
                    found_objects = True
                    tl = 22
                    for (px, py, ddx, ddy) in [(fx,fy,1,1),(fx+fw,fy,-1,1),(fx,fy+fh,1,-1),(fx+fw,fy+fh,-1,-1)]:
                        cv2.line(annotated_frame, (px, py), (px + ddx * tl, py), (0, 255, 80), 2)
                        cv2.line(annotated_frame, (px, py), (px, py + ddy * tl), (0, 255, 80), 2)
                    fc_x, fc_y = fx + fw // 2, fy + fh // 2
                    cv2.circle(annotated_frame, (fc_x, fc_y), 4, (0, 255, 80), -1)
                    cv2.putText(annotated_frame, "HUMAN", (fx, max(12, fy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 80), 1, cv2.LINE_AA)

            elif cur_mode == 6:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0: found_objects = True
                for (nx, ny, nw, nh) in faces:
                    cv2.rectangle(annotated_frame, (nx, ny), (nx+nw, ny+nh), (255, 100, 100), 2)
                    cv2.putText(annotated_frame, "Face", (nx, ny-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 150, 150), 2, cv2.LINE_AA)
                    
            elif cur_mode == 9:
                if not fast_tracker_initialized[c_idx]:
                    draw_beautiful_text(annotated_frame, "LOCK ON FOR TRACKING (Press 'A')", (20, 100), 0.8, (100, 255, 255))
                    # Tracking secimi c_idx = 0 icinse aktif, 2. kamera icin bypass ediyoruz simdilik
                    if c_idx == 0:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('k'):
                            bbox = cv2.selectROI("Cv_Projem - Gelismis Arayuz", frame_orig, fromCenter=False, showCrosshair=True)
                            if bbox[2] > 0 and bbox[3] > 0:
                                if trackers[c_idx] is None: trackers[c_idx] = cv2.TrackerCSRT_create()
                                trackers[c_idx].init(frame_orig, bbox)
                                fast_tracker_initialized[c_idx] = True
                else:
                    try:
                        success, bbox = trackers[c_idx].update(frame_orig)
                        if success:
                            tx, ty, tw, th = [int(v) for v in bbox]
                            cv2.rectangle(annotated_frame, (tx, ty), (tx+tw, ty+th), (50, 255, 50), 3)
                            draw_beautiful_text(annotated_frame, "LOCKED", (tx, max(20, ty - 10)), 0.6, (50, 255, 50), (0,0,0), 0.8)
                        else:
                            draw_beautiful_text(annotated_frame, "OBJECT LOST!", (20, 150), 1.0, (50, 50, 255))
                    except: pass
                    
            if found_objects: any_found_objects = True
            annotated_frames.append(annotated_frame)

        # Main frame display logic
        main_frame = annotated_frames[0]
        h, w = main_frame.shape[:2]

        if security_armed and any_found_objects:
            if notifier is not None:
                if notifier.send_alert(f"ALARM! Motion detected! Time: {time.strftime('%H:%M:%S')}"):
                    ts_str = time.strftime("%Y-%m-%d_%H-%M-%S")
                    cv2.imwrite(f"Vault/Security/SECURITY_{ts_str}.jpg", main_frame)
            draw_beautiful_text(main_frame, "!!! INTRUDER ALERT !!!", (w//2 - 150, 50), 1.0, (50, 50, 255), (0,0,0), 0.8, 3)

        if is_recording:
            if video_writer is None:
                video_writer = cv2.VideoWriter(f"RECORD_{int(time.time())}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
            video_writer.write(main_frame)
            cv2.circle(main_frame, (w - 40, h - 40), 10, (50, 50, 255), -1)
            draw_beautiful_text(main_frame, "REC", (w - 100, h - 30), 0.6, (50, 50, 255))

        draw_beautiful_text(main_frame, f"CAM {menu_target_cam+1} CONTROL | S:SwitchCam | C:Photo | R:Record | Z:Zoom | A:Arm | M:Menu", (20, h - 100), 0.45, (200, 200, 200), thickness=1)
        if zoom_factors[0] > 1.0: draw_beautiful_text(main_frame, f"Z1: {zoom_factors[0]}X", (20, h - 30), 0.55, (50, 255, 255), thickness=1)
        if security_armed: draw_beautiful_text(main_frame, "ARMED", (20, h - 65), 0.55, (50, 50, 255), thickness=1)
        
        opt_rx, opt_ry, opt_rw, opt_rh = w - 160, 20, 140, 45
        draw_transparent_bg(main_frame, opt_rx, opt_ry, opt_rw, opt_rh, (20, 20, 20), 0.8)
        cv2.rectangle(main_frame, (opt_rx, opt_ry), (opt_rx+opt_rw, opt_ry+opt_rh), (150, 150, 150), 1)
        opt_icon = "[-]" if menu_expanded else "[+]"
        cv2.putText(main_frame, f"{opt_icon} MENU (M)", (opt_rx + 15, opt_ry + 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        if menu_expanded:
            buttons = get_menu_rects(w)
            bg_start_y = 70
            bg_height = (435 - 70) + 45
            draw_transparent_bg(main_frame, w - 430, bg_start_y, 420, bg_height, (10, 10, 10), 0.75)
            cv2.rectangle(main_frame, (w - 430, bg_start_y), (w - 10, bg_start_y + bg_height), (100, 100, 100), 1)
            
            for i, btn in enumerate(buttons):
                rx, ry, rw, rh = btn['rect']
                is_active = False
                if btn['type'] == 'mode' and btn['id'] == cam_modes[menu_target_cam]: is_active = True
                elif btn['type'] == 'action' and btn['id'] == 'r' and is_recording: is_active = True
                elif btn['type'] == 'action' and btn['id'] == 'a' and security_armed: is_active = True
                elif btn['type'] == 'action' and btn['id'] == 'z' and zoom_factors[menu_target_cam] > 1.0: is_active = True
                elif btn['type'] == 'action' and btn['id'] == 'f' and auto_framings[menu_target_cam]: is_active = True
                
                is_hovered = (i == menu_selected_idx)
                
                bg_color = (200, 100, 50) if is_active else (40, 40, 40)
                if is_hovered and not is_active: bg_color = (80, 80, 80)
                
                draw_transparent_bg(main_frame, rx, ry, rw, rh, bg_color, 0.9 if is_active or is_hovered else 0.4)
                if is_active: cv2.rectangle(main_frame, (rx, ry), (rx+rw, ry+rh), (255, 200, 100), 2)
                elif is_hovered: cv2.rectangle(main_frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 255), 1)
                    
                text_color = (255, 255, 255) if is_active or is_hovered else (180, 180, 180)
                btn_txt = btn['text']
                if btn['id'] == 'switch': btn_txt = f"S. Target: CAM {menu_target_cam+1}"
                cv2.putText(main_frame, btn_txt, (rx + 8, ry + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 2 if is_active else 1, cv2.LINE_AA)

        fps_monitor.stop()
        fps = fps_monitor.getTimeSec()
        if fps > 0: draw_beautiful_text(main_frame, f"FPS: {1/fps:.1f}", (20, 50), 1.0, (50, 255, 50), thickness=2)
        fps_monitor.reset()
        
        # P.I.P
        if len(annotated_frames) > 1 and annotated_frames[1] is not None:
            pip_frame = annotated_frames[1]
            pip_h, pip_w = int(h * 0.28), int(w * 0.28)
            pip_resized = cv2.resize(pip_frame, (pip_w, pip_h))
            
            cv2.rectangle(pip_resized, (0, 0), (pip_w, pip_h), (255, 0, 0), 2)
            cv2.rectangle(pip_resized, (0, 0), (160, 25), (255, 0, 0), -1)
            pip_text = f"CAM-2 ({cam_modes.get(1, 1)})"
            cv2.putText(pip_resized, pip_text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            pip_y, pip_x = h - pip_h - 20, w - pip_w - 20
            try: main_frame[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = pip_resized
            except: pass
                 
        cv2.imshow("Cv_Projem - Gelismis Arayuz", main_frame)
        key_ex = cv2.waitKeyEx(1)
        key = key_ex & 0xFF
        
        simulated_key = None
        if gui_action:
            simulated_key = ord(str(gui_action).lower())
            gui_action = None
            
        if menu_expanded and key_ex != -1:
            buttons = get_menu_rects(w)
            bt_len = len(buttons)
            if key_ex in [63232, 2490368, 82, ord('w'), ord('W')]: menu_selected_idx = (menu_selected_idx - 1) % bt_len
            elif key_ex in [63233, 2621440, 84, ord('s'), ord('S')]: menu_selected_idx = (menu_selected_idx + 1) % bt_len
            elif key_ex in [63234, 2424832, 81, ord('a'), ord('A')]: 
                if menu_selected_idx < 10: menu_selected_idx = min(bt_len - 1, menu_selected_idx + 10)
            elif key_ex in [63235, 2555904, 83, ord('d'), ord('D')]: 
                if menu_selected_idx >= 10: menu_selected_idx = max(0, menu_selected_idx - 10)
            elif key in [13, 32]:
                btn = buttons[menu_selected_idx]
                if btn['type'] == 'mode':
                    cam_modes[menu_target_cam] = btn['id']
                    if cam_modes[menu_target_cam] == 9:
                        fast_tracker_initialized[menu_target_cam] = False
                elif btn['type'] == 'action': simulated_key = ord(str(btn['id']).lower())
                menu_expanded = False
        
        if key == ord('q'): break
        elif key == ord('c') or key == ord('C') or simulated_key == ord('c'):
            ts_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Vault/Photos/SCREENSHOT_{ts_str}.jpg"
            cv2.imwrite(filename, frames_orig[menu_target_cam])
            print(f"Fotograf kaydedildi: {filename}")
        elif key == ord('r') or key == ord('R') or simulated_key == ord('r'):
            is_recording = not is_recording
            if not is_recording and video_writer: video_writer.release()
            elif is_recording and video_writer is None:
                ts_str = time.strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"Vault/Videos/RECORD_{ts_str}.mp4"
                video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                print(f"Video kaydi basladi: {filename}")
        elif key == ord('z') or key == ord('Z') or simulated_key == ord('z'):
            auto_framings[menu_target_cam] = False
            zoom_factors[menu_target_cam] += 0.5
            if zoom_factors[menu_target_cam] > 2.0: zoom_factors[menu_target_cam] = 1.0
        elif key == ord('f') or key == ord('F') or simulated_key == ord('f'):
            zoom_factors[menu_target_cam] = 1.0
            auto_framings[menu_target_cam] = not auto_framings[menu_target_cam]
        elif key == ord('s') or key == ord('S') or simulated_key == ord('s'):
            if len(captures) > 1: menu_target_cam = 1 - menu_target_cam
        elif key == ord('a') or key == ord('A') or simulated_key == ord('a'):
            security_armed = not security_armed
        elif key == ord('m') or key == ord('M'):
            menu_expanded = not menu_expanded
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:
            cam_modes[menu_target_cam] = int(chr(key))
            if cam_modes[menu_target_cam] == 9: fast_tracker_initialized[menu_target_cam] = False
            
    for cap in captures: cap.release()
    cv2.destroyAllWindows()

def select_camera_source():
    root = tk.Tk()
    root.title("Kamera Bağlantı Seçenekleri (Dashboard Modu)")
    root.geometry("680x480")
    root.configure(padx=20, pady=20)
    
    selected_sources = []
    info_frame = tk.Frame(root, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
    info_frame.pack(fill=tk.X, pady=(0, 20), ipady=10)
    
    lbl_info = tk.Label(info_frame, text="Seçilen Kameralar: YOK (En az 1 kamera seçin)", font=("Arial", 14, "bold"), fg="#cc0000", bg="#f0f0f0")
    lbl_info.pack(pady=5)
    
    lbl_desc = tk.Label(info_frame, text="Birden fazla kamera seçtiğinizde P.I.P (Picture-in-Picture)Dashboard modu açılır.", font=("Arial", 10), bg="#f0f0f0", fg="#555555")
    lbl_desc.pack()

    def update_label():
        if not selected_sources: lbl_info.config(text="Seçilen Kameralar: YOK (En az 1 kamera seçin)", fg="#cc0000")
        else: lbl_info.config(text=f"Seçilen Kameralar ({len(selected_sources)}/2): " + " | ".join([f"CAM-{i+1}({s})" for i, s in enumerate(selected_sources)]), fg="#008800")

    def add_source(src):
        if str(src) not in [str(s) for s in selected_sources]:
            if len(selected_sources) >= 2: return messagebox.showwarning("Limit", "Maksimum 2 kamera desteklenmektedir!", parent=root)
            selected_sources.append(str(src))
            update_label()
        else: messagebox.showinfo("Bilgi", "Bu kamera zaten seçili!", parent=root)

    def start_app():
        if selected_sources: root.destroy()
        else: messagebox.showwarning("Eksik", "Lütfen en az 1 kamera ekleyin!", parent=root)

    main_content = tk.Frame(root)
    main_content.pack(fill=tk.BOTH, expand=True)

    left_frame = tk.Frame(main_content)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
    tk.Frame(main_content, width=2, bg="#cccccc").pack(side=tk.LEFT, fill=tk.Y, pady=10)
    right_frame = tk.Frame(main_content)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

    tk.Label(left_frame, text="Kablosuz Kaynaklar", font=("Arial", 14, "bold"), fg="#333333").pack(pady=(5, 15))
    tk.Button(left_frame, text="➕ WiFi Kamera Ekle", command=lambda: add_source(simpledialog.askstring("WiFi Kamera", "WiFi Kamera IP/URL girin:", parent=root)), height=2, width=22, font=("Arial", 12)).pack(pady=8)
    
    def connect_bluetooth():
        scan_win = tk.Toplevel(root)
        scan_win.title("Kameraları Tarayıcı")
        scan_win.geometry("380x320")
        scan_win.configure(padx=20, pady=20)
        lbl = tk.Label(scan_win, text="Lütfen Bekleyin... Cihazlar Aranıyor", font=("Arial", 12))
        lbl.pack(pady=10)
        scan_win.update()

        found_cams = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                if cap.read()[0]: found_cams.append(i)
            cap.release()
            
        for widget in scan_win.winfo_children(): widget.destroy()
            
        tk.Label(scan_win, text="✅ Bulunan Cihazlar", font=("Arial", 14, "bold"), fg="#118811").pack(pady=10)
        if not found_cams: tk.Label(scan_win, text="Cihaz bulunamadı.", fg="red").pack(pady=10)
        else:
            for c in found_cams:
                name = "Dahili Mac" if c==0 else "iPhone/iPad vb." if c==1 else f"Harici {c}"
                tk.Button(scan_win, text=f"➕ {name} Ekle", command=lambda idx=c: [scan_win.destroy(), add_source(idx)], font=("Arial", 12), width=30, height=2, bg="#e0ffe0").pack(pady=5)

    tk.Button(left_frame, text="➕ Bluetooth Kamera Ekle", command=connect_bluetooth, height=2, width=25, font=("Arial", 12)).pack(pady=8)

    tk.Label(right_frame, text="Hızlı Port Ekleme", font=("Arial", 14, "bold"), fg="#333333").pack(pady=(5, 15))
    tk.Button(right_frame, text="➕ Kamera [0] (Ana)", command=lambda: add_source("0"), height=2, width=22, font=("Arial", 12)).pack(pady=5)
    tk.Button(right_frame, text="➕ Kamera [1] (İkincil)", command=lambda: add_source("1"), height=2, width=22, font=("Arial", 12)).pack(pady=5)
    
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
    tk.Button(bottom_frame, text="🚀 SİSTEMİ BAŞLAT", command=start_app, font=("Arial", 16, "bold"), bg="#0066cc", fg="blue", height=2, width=30).pack(pady=5)

    try: root.eval('tk::PlaceWindow . center')
    except: pass
    root.mainloop()
    return selected_sources

if __name__ == "__main__":
    sources = select_camera_source()
    if sources: main_app(sources)
    else: print("Kamera seçilmedi, başlatılamadı.")
