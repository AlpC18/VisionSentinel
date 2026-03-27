import cv2
import supervision as sv
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        # Ultralytics YOLO modelini yükle
        self.model = YOLO(model_name)
        # Supervision annotators
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def process(self, img):
        # Frame üzerinde tahmin yap (Konsolu kirletmemek için verbose=False)
        results = self.model(img, verbose=False)[0]
        
        # Ultralytics sonuçlarını Supervision objesine çevir
        detections = sv.Detections.from_ultralytics(results)
        
        # Güven skoru %30'dan büyük olanları filtrele
        detections = detections[detections.confidence > 0.3]
        
        # Etiketleri oluştur
        labels = [
            f"{self.model.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        # Çizimleri yap
        annotated_image = self.box_annotator.annotate(
            scene=img.copy(), detections=detections)
        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)
        
        return annotated_image, labels
