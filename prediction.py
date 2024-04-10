import cv2
import argparse

import supervision as sv
from ultralytics import YOLO
import numpy as np


ZONE_POLYGON = np.array([
    [0,0],
    [1280,0],
    [1250,720],
    [0,720]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOV8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width,frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1.3
    )
    # zone = sv.PolygonZone()
    while True:
        ret , frame = cap.read()

        result = model(frame)[0]
        detection = sv.Detections.from_yolov8(result)
        frame = box_annotator.annotate(scene=frame,detections=detection)
        cv2.imshow('yolov8',frame)
        if(cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()