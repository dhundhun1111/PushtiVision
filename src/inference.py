import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

def ensemble_inference(model_paths, source_dir, conf_thres=0.25, iou_thres=0.45, output_dir='runs/ensemble_predictions', weights=None):
    if weights is None:
        weights = [1.0] * len(model_paths)
    
    models = [YOLO(mp) for mp in model_paths]
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        all_boxes, all_scores, all_labels = [], [], []
        
        for model in models:
            res = model.predict(img, conf=conf_thres, iou=iou_thres, verbose=False)
            bboxes, scores, labels = [], [], []
            
            for box in res[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                bboxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
                scores.append(conf)
                labels.append(cls_id)
            
            all_boxes.append(bboxes)
            all_scores.append(scores)
            all_labels.append(labels)
        
        fused_bboxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels, weights=weights, iou_thr=0.5, skip_box_thr=conf_thres
        )
        
        fused_bboxes_px = [(int(b[0] * width), int(b[1] * height), int(b[2] * width), int(b[3] * height)) for b in fused_bboxes]
        final_img = img.copy()
        
        for (x1, y1, x2, y2), score, cls_id in zip(fused_bboxes_px, fused_scores, fused_labels):
            color = (0, 255, 0)
            cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)
            label = f"{models[0].names[int(cls_id)]} {score:.2f}"
            cv2.putText(final_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), final_img)

def detect_calories(model_path, source, food_calories, conf_threshold=0.25, iou_threshold=0.45, save_results=True):
    model = YOLO(model_path)
    results = model.predict(source=source, conf=conf_threshold, iou=iou_threshold, save=save_results, save_txt=save_results)
    
    for result in results:
        class_counts = {}
        
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        total_calories = sum(food_calories.get(cls_name, 0) * count for cls_name, count in class_counts.items())
        
        print(f"Image: {os.path.basename(result.path)}")
        print(f" Detected items: {class_counts}")
        print(f" Estimated total calories: {total_calories}\n")

if __name__ == "__main__":
    # Example usage:
    model_paths = ["yolov8x.pt"]
    source_dir = "test_images"
    food_calories = {"apple": 52, "banana": 89, "burger": 295}  # Example food calorie values
    
    ensemble_inference(model_paths, source_dir)
    detect_calories("yolov8x.pt", source_dir, food_calories)
