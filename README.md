# 🛡️ VisionSentinel (v1.0)
### *Next-Generation AI-Driven Computer Vision & Real-Time Intelligence Suite*

**VisionSentinel** is a comprehensive, multi-module computer vision platform designed for security, analytics, and interactive AI. It leverages state-of-the-art architectures (**YOLOv8**, **MediaPipe**, and **Supervision**) to provide an all-in-one vision engine capable of everything from simple motion detection to complex spatial behavior analysis.

---

## 🧭 The Vision
VisionSentinel was built to bridge the gap between "standard surveillance" and "intelligent decision-making." Whether it’s monitoring a secure perimeter, analyzing retail foot traffic via heatmaps, or interacting with software through hand gestures, VisionSentinel provides the raw data and visual clarity needed for modern AI applications.

---

## 🚀 Core Technologies
VisionSentinel is powered by a robust stack of industry-leading libraries:
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics):** The world's fastest and most accurate real-time object detection model.
- **[Roboflow Supervision](https://github.com/roboflow/supervision):** A powerful toolkit for filtering, annotating, and counting detected objects.
- **[MediaPipe](https://github.com/google-ai-edge/mediapipe):** High-fidelity solutions for human pose, face, and hand landmarking.
- **[OpenCV](https://opencv.org/):** The foundation for hardware acceleration, image processing, and the custom Interactive HUD.

---

## 🛠️ Feature Deep-Dive

### 🔍 1. Intelligent Object Identification & Tracking
- **Automated Labeling:** Real-time identification of 80+ COCO classes (Persons, Vehicles, Animals, etc.).
- **Temporal Consistency (ByteTrack):** Objects aren't just detected; they are tracked with unique IDs that persist as long as they are in view.
- **Manual Lock-On (Mission Mode):** Using the **CSRT/KCF** algorithms, users can manually select any region of the screen (ROI) for dedicated, hardware-accelerated tracking.

### 👤 2. Advanced Human Biometrics
- **Cyber-Mesh Hand Tracking:** 21-point tracking for up to 4 hands. Perfect for gesture-based control systems.
- **Kinematic Pose Analysis:** Skeletal landmarking to recognize human posture, repetitive movements, or falls.
- **Facial Geometry Monitoring:** Real-time face detection with BlazeFace, providing smooth bounding boxes even during rapid motion.

### 📊 3. Spatial Behavior & Crowd Analytics
- **Live Occupancy Heatmaps:** Understand "Hot Zones" in any environment by visualizing where objects spend the most time.
- **Line-Crossing Perimeters:** Set up virtual "tripwires" to count objects moving in or out of a specific direction.
- **Polygon Zone Triggering:** Define complex areas of interest (AOIs). The system calculates **Time-in-Zone** statistics for every unique ID.
- **Dynamic Distance Estimation:** Uses camera geometry to approximate the physical distance (in meters) between the lens and detected objects like people or cars.

### 🌓 4. Tactical Perception Enhancements
- **Multi-Gen Night Vision:** A sophisticated filter stack (Laplacian sharpening + Phosphor-green glow) providing visibility in extremely low-light conditions.
- **Thermal Vision Signature:** Simulates heat maps by analyzing grayscale pixel intensities, highlighting "warm" targets through the JET colormap.
- **Auto-Framing AI:** A virtual camera operator that uses detection data to smoothly zoom and pan, keeping targets centered in the frame.

### 🛡️ 5. Integrated Security & The vault
- **Motion Perimeter Sensing:** Pixel-perfect motion detection using weighted accumulation.
- **SMS Security Protocol:** When armed, the system uses **Twilio API** to send instant intrusion alerts to a mobile device.
- **The Vault:** All security logs, high-res screenshots, and video recordings are automatically organized into daily sub-directories.

---

## 🌍 Real-World Use Cases

| **Industry** | **Implementation** |
| :--- | :--- |
| **Retail & Business** | Use Heatmaps and Time-in-Zone to optimize store layouts and analyze customer interaction with products. |
| **Physical Security** | Deploy tripwires and SMS alerts for 24/7 automated monitoring of restricted zones. |
| **Smart Homes** | Use hand-gesture tracking to control connected devices or trigger automation routines. |
| **Sports Science** | Use Pose Estimation to analyze athlete form (e.g., squat depth, bowling actions) in real-time. |
| **Industrial Safety** | Detect if personnel are entering dangerous machinery zones (Polygon Zones) and trigger an instant alarm. |
| **Transportation** | Monitor vehicle traffic flow and count cars crossing specific road intersections. |

---

## 🖥️ Interactive Dashboard (GUI) Controls

The VisionSentinel dashboard is built for high-speed operation:
- **[M] Menu Toggle:** Expands the sidebar for access to all AI modes.
- **[S] Source Switch:** Rotate between available camera feeds (Supports USB, WiFi, or Internal cams).
- **[C] Instant Capture:** Save a frame to `Vault/Photos/` without interrupting the stream.
- **[R] Video Recording:** Toggle 20 FPS high-definition recording directly to `Vault/Videos/`.
- **[Z] Cycle Zoom:** Step up from 1.0x to 1.5x and 2.0x zoom levels.
- **[A] Arm Security:** Activates the intrusion detection and Twilio SMS alert server.
- **Numbers [0-9]:** Direct hotkeys to switch vision modes (e.g., `4` for Thermal, `5` for Night Vision).

---

## 📂 Project Organization

```text
📁 VisionSentinel/
├── 📄 VisionApp.py             # Main AI Execution Kernel
├── 📄 Main.py                  # Legacy / Alternative Launcher
├── 📁 VisionEngine_Core/       # Core Library Source (OpenCV samples)
├── 📁 VisionEngine_Analytics/  # Supervision Logic & Utilities
├── 📄 NotificationModule.py    # Twilio SMS / Cloud Communication
├── 📄 HandTrackingModule.py    # Hand-Gesture Recognition Module
├── 📄 PoseDetectionModule.py    # Human Skeleton Landmark Logic
├── 📄 FaceDetectionModule.py    # Facial Biometrics Implementation
├── 📄 ObjectDetectionModule.py  # YOLOv8 & Supervision Integration
├── 📁 Vault/                   # Organized Media Archive
│   ├── 📸 Photos/              # Snapshots & Evidence
│   ├── 🎥 Videos/               # Session Recordings
│   └── 🚨 Security/             # Intruder Alert Artifacts
├── 📄 Start.command            # macOS Launcher (GUI Mode)
└── 📄 Start_Camera.command     # macOS Direct Kernel Launch
```

---

## ⚙️ Installation & Setup

1. **Verify Python 3.9+:**
   ```bash
   python3 --version
   ```
2. **Setup and Activate Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Automated Dependency Install:**
   ```bash
   pip install opencv-python mediapipe numpy supervision ultralytics twilio
   ```
4. **Configuration (Twilio):**
   Open `NotificationModule.py` and input your API keys to enable remote security alerts.

---

## 🤝 Contributing
VisionSentinel is open to contributions from researchers and developers.
1. Fork it!
2. Create your feature branch (`git checkout -b feature/CoolNewAI`)
3. Commit your changes (`git commit -am 'Add some cool features'`)
4. Push to the branch (`git push origin feature/CoolNewAI`)
5. Create a new Pull Request.

---

## 📜 License & Credits
- **License:** MIT License.
- **Credits:** Special thanks to the **Ultralytics**, **Roboflow**, and **Google MediaPipe** teams for their incredible contributions to the vision community.

---
**VisionSentinel: Intelligence in every pixel.** 🛡️🔍
