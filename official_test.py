from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model(data=r"/home/shenyiru/pythonProject/ultralytics/ultralytics/assets/bus.jpg", save=True, device=0)
