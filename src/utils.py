import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def preprocess_image(input_path, output_path, transform=None, size=(640, 640)):
    """
    Reads an image, applies transformations, resizes it, and saves the preprocessed image.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        transform (albumentations.Compose, optional): Augmentations to apply.
        size (tuple): Target size (width, height).
    """
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    if transform:
        augmented = transform(image=img)
        img = augmented['image']
    
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, img)

def load_food_calories(calories_file):
    """
    Loads food calorie information from a file.

    Args:
        calories_file (str): Path to the text or JSON file containing food calorie data.

    Returns:
        dict: A dictionary mapping food names to their respective calorie values.
    """
    food_calories = {}
    if calories_file.endswith('.txt'):
        with open(calories_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    food, cal = parts[0].strip(), parts[1].strip()
                    try:
                        food_calories[food] = float(cal)
                    except ValueError:
                        print(f"Warning: Skipping invalid calorie value for {food}")
    elif calories_file.endswith('.json'):
        import json
        with open(calories_file, 'r') as f:
            food_calories = json.load(f)
    return food_calories

def draw_bounding_boxes(image, detections, class_names):
    """
    Draws bounding boxes on the image with labels.

    Args:
        image (numpy.ndarray): Input image.
        detections (list): List of detections with (x1, y1, x2, y2, score, class_id).
        class_names (dict): Dictionary mapping class indices to names.

    Returns:
        numpy.ndarray: Image with bounding boxes drawn.
    """
    for (x1, y1, x2, y2, score, cls_id) in detections:
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[int(cls_id)]} {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def save_results(image, output_path):
    """
    Saves the processed image with detections.

    Args:
        image (numpy.ndarray): Image with bounding boxes.
        output_path (str): Path to save the output image.
    """
    cv2.imwrite(output_path, image)

def normalize_bbox(bbox, width, height):
    """
    Normalizes bounding box coordinates to the range [0,1].

    Args:
        bbox (tuple): (x1, y1, x2, y2) in pixel format.
        width (int): Image width.
        height (int): Image height.

    Returns:
        list: Normalized bounding box [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    return [x1 / width, y1 / height, x2 / width, y2 / height]

def denormalize_bbox(bbox, width, height):
    """
    Converts normalized bounding box coordinates back to pixel values.

    Args:
        bbox (list): Normalized bounding box [x1, y1, x2, y2].
        width (int): Image width.
        height (int): Image height.

    Returns:
        tuple: Bounding box (x1, y1, x2, y2) in pixel format.
    """
    x1, y1, x2, y2 = bbox
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

def setup_albumentations():
    """
    Defines Albumentations transforms for image preprocessing.

    Returns:
        albumentations.Compose: A transformation pipeline.
    """
    return A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

if __name__ == "__main__":
    # Example usage:
    img_path = "sample.jpg"
    output_path = "preprocessed.jpg"
    transform = setup_albumentations()

    preprocess_image(img_path, output_path, transform)
