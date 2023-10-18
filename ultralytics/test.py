from ultralytics.models.yolo import YOLO
import cv2
def detect():
    model = YOLO('yolov8n-pose.pt')

    # model.predict(source=0,show=True)

    result =model.predict(source="D:\\test\\stand2.jpg", save=True)


detect()

