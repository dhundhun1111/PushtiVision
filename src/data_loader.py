import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def preprocess_image(input_path, output_path, transform=None, size=(640, 640)):
    """
    Load an image, apply transformations, and save the preprocessed image.
    
    :param input_path: Path to the input image
    :param output_path: Path to save the processed image
    :param transform: Albumentations transformations to apply (optional)
    :param size: Target image size (width, height)
    """
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Unable to load image {input_path}")
        return
    
    if transform:
        augmented = transform(image=img)
        img = augmented['image']
    
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, img)

def load_images_from_directory(directory, extensions=('jpg', 'png')):
    """
    Load image paths from a directory.
    
    :param directory: Directory path containing images
    :param extensions: Allowed image file extensions
    :return: List of image paths
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extensions)]

def get_default_transforms():
    """
    Define default image transformations.
    
    :return: Albumentations Compose object with transformations
    """
    return A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
