#!/bin/bash
# Bulunduğumuz klasöre git
cd "$(dirname "$0")"

# Gerekli kütüphanelerin yüklü olduğundan emin ol (opsiyonel, hata verirse diye)
# pip install opencv-python mediapipe numpy

# Uygulamayı başlat
./venv/bin/python3 VisionApp.py
