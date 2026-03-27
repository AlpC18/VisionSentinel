# 🛡️ VisionSentinel: Advanced AI-Powered Computer Vision Suite

**VisionSentinel** is a high-performance, real-time computer vision engine designed for advanced monitoring, surveillance, and interactive AI analysis. Built on top of **OpenCV**, **MediaPipe**, **Supervision**, and **YOLOv8**, it transforms standard camera inputs into an intelligent data stream.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)
![AI](https://img.shields.io/badge/AI-YOLOv8--Enabled-red.svg)

---

## 🌟 Key Features

VisionSentinel is not just a tracker; it's a modular intelligence system.

### 🔍 1. Multi-Modal Object Detection & Tracking
- **YOLOv8 Integration:** Real-time identification of 80+ object classes.
- **ByteTrack Algorithm:** Robust multi-object tracking that maintains IDs even during occlusions.
- **Fast Tracker (CSRT/KCF):** High-precision manual locking onto specific targets for dedicated tracking.

### 👤 2. Advanced Human Analysis
- **Cyber-Mesh Hand Tracking:** Identifies up to 4 hands simultaneously with 21 landmark points.
- **Kinematic Pose Estimation:** Complete skeletal tracking for motion analysis.
- **Biometric Face Detection:** High-speed facial recognition and framing using the BlazeFace model.

### 🌙 3. Tactical Vision Modes
- **Thermal Vision Simulation:** High-contrast heat mapping to highlight body temperatures and movement.
- **Night Vision (Gen 4):** Signal-enhanced phosphor-green simulation with digital noise reduction and grid overlays.
- **Dark Mode Boost:** Gamma-corrected image processing for low-light environments.

### 🛡️ 4. Active Security & Surveillance
- **Motion Perimeter Sensing:** Detects subtle pixel-level movements.
- **Security Armed Mode:** Automatically triggers SMS alerts (via Twilio) and logs snapshots upon intruder detection.
- **Vault System:** Dedicated categorized storage for screenshots, security alerts, and high-definition recordings.

### 📊 5. Spatial Analytics
- **Heatmap Generation:** Visualize occupancy patterns and high-traffic areas over time.
- **Line Counting & Zone Analysis:** Count objects crossing specific perimeters or calculate "time-in-zone" statistics.
- **Distance Estimation:** Real-time depth approximation based on focal length and object geometry.

---

## 🎮 Interactive Dashboard (GUI)

VisionSentinel features a sophisticated heads-up display (HUD) with mouse-interactive controls:
- **Expanded Sidebar Menu:** Switch between 10 different vision modes instantly.
- **Picture-in-Picture (P.I.P):** Support for multiple camera sources (up to 2) with a secondary dashboard view.
- **Live HUD:** Real-time FPS monitoring, mode indicators, and security status.
- **Auto-Framing (A.I. Zoom):** Automatically centers and zooms into detected human targets.

---

## 📂 Project Structure

```text
VisionSentinel/
├── VisionApp.py                # Main application entry point
├── VisionEngine_Core/          # OpenCV source and resources
├── VisionEngine_Analytics/     # Supervision source and modules
├── NotificationModule.py       # SMS / Cloud alerting system
├── HandTrackingModule.py       # AI Hand landmarking logic
├── PoseDetectionModule.py       # Skeletal analysis logic
├── FaceDetectionModule.py       # Biometric face detection logic
├── ObjectDetectionModule.py     # YOLO / Supervision integration
├── Vault/                      # Classified output storage
│   ├── Photos/                 # Captured high-res screenshots
│   ├── Videos/                 # Recorded mp4 sequences
│   └── Security/               # Intruder alerts and logs
├── Start.command               # macOS Quick Launch script
└── Start_Camera.command        # macOS Direct Camera Launch
```

---

## 🚀 Installation & Quick Start

### 📋 Prerequisites
- Python 3.9 or higher
- [Homebrew](https://brew.sh/) (Recommended for macOS)

### 🛠️ Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/AlpC18/VisionSentinel.git
   cd VisionSentinel
   ```
2. **Setup Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install opencv-python mediapipe numpy supervision ultralytics twilio
   ```

### ⚡ Launch
On macOS, you can simply run the provided command scripts:
- **`./Start.command`**: Launches the main dashboard with source selection.
- **`./Start_Camera.command`**: Launches the primary camera instantly.

---

## 🛠️ Configuration (SMS Alerts)
To enable real-time SMS alerts, edit `NotificationModule.py` and provide your **Twilio** credentials:
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_NUMBER`
- `TARGET_PHONE_NUMBER`

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.

## 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Built with ❤️ for the Advanced AI Community.**
