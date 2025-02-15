import torch
from ultralytics import YOLO

# def train_yolo(data_yaml, pretrained_weights='yolov8x.pt', hyp=None, epochs=100, imgsz=640, lr=1e-3, project_name='runs/train', experiment_name='food_yolo'):
#     model = YOLO(pretrained_weights)
#     results = model.train(
#         data=data_yaml,
#         epochs=epochs,
#         batch=16,
#         imgsz=imgsz,
#         lr0=lr,
#         patience=25,
#         project=project_name,
#         name=experiment_name,
#         verbose=True,
#         hyp=hyp
#     )
#     return model


def train_yolo(data_yaml, pretrained_weights):
    model = YOLO(pretrained_weights)  # Load the YOLO model
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=512,
        batch=2,
        lr0=0.01,
        device="cpu",
        amp=True,  # Automatic mixed precision
    )
    return results