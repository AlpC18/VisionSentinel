#!/bin/bash
# Proje klasörüne git
cd /Users/alpci/Desktop/Cv_projem.py || { echo "Proje klasörü bulunamadı!"; exit 1; }

# Python uygulamasını başlat
echo "Süper Sense Başlatılıyor..."
./venv/bin/python3 VisionApp.py
