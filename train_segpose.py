from ultralytics import YOLO

# Create a new YOLOv8n-OBB model from scratch
if __name__=="__main__":
    model = YOLO(r"yolov8s-segpose.yaml")

    # Train the model
    results = model.train(data=r"/Users/shenyiru/PycharmProjects/ultralytics/pole_dateset/dataset.yaml", epochs=300, imgsz=1050, batch=2, rect=True, project="test_segpose", mask_ratio=1, workers=0)