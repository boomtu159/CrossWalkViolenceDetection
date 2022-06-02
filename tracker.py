import argparse

import numpy as np
from sympy import false
import torch
import yolov5
import os
import sys
import cv2
from typing import Union, List, Optional
from utils.dataloaders import LoadImages
from utils.plots import Annotator, colors
from pathlib import Path
import norfair
from norfair import Detection, Tracker, Video, Paths
from objectDetected import ObjectDetected
import time

max_distance_between_points: int = 30

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

classes = {"car": 0, "motorcycle": 1, "crosswalk": 2}

class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load model
        self.model = yolov5.load(model_path, device=device)

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def center(points):
    return [np.mean(np.array(points), axis=0)]


def norfair_to_yolo(detection):
    bbox = np.array([detection[0][0], detection[0][1], detection[1][0], detection[1][1]])
    return bbox

def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor,
    track_points: str = 'centroid'  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections
    """
    norfair_detections: List[Detection] = []

    if track_points == 'centroid':
        detections_as_xywh = yolo_detections.xywh[0]
        labels, cord_thres = yolo_detections.xyxyn[0][:, -1].numpy(), yolo_detections.xyxyn[0][:, :-1].numpy()
        for i, detection_as_xywh in enumerate(detections_as_xywh):
            cls = int(labels[i])
            label = yolo_detections.names[cls]  
            centroid = np.array(
                [
                    detection_as_xywh[0].item(),
                    detection_as_xywh[1].item()
                ]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(points=centroid, scores=scores, label=label)
            )
    elif track_points == 'bbox':
        detections_as_xyxy = yolo_detections.xyxy[0]
        labels, cord_thres = yolo_detections.xyxyn[0][:, -1].numpy(), yolo_detections.xyxyn[0][:, :-1].numpy()

        for i, detection_as_xyxy in enumerate(detections_as_xyxy):
            cls = int(labels[i])
            label = yolo_detections.names[cls]  

            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                ]
            )
            scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
            norfair_detections.append(
                Detection(points=bbox, scores=scores,label=label)
            )

    return norfair_detections

def check_stopped_vechicle(current_vechicles, previous_vechicles):

    vechicles = []

    for v1 in current_vechicles:
        for v2 in previous_vechicles:
            
            if v1.id == v2.id and abs(v1.x - v2.x) < 2 and abs(v1.y - v2.y) < 2 and abs(v1.w - v2.w) < 2 and abs(v1.h - v2.h) < 2:
                add_unique_vechicle(v1)

                if should_be_add_into_stop_vechicle(v1):
                    print(f"5s stopped {v1.id}")
                    vechicles.append(v1)
        
    return vechicles

def should_be_add_into_stop_vechicle(vechicle):
    current_time = time.time()
    
    for v1 in vechicles_unique_list:
        diff_time = current_time - v1.timestart
        if v1.id == vechicle.id and diff_time > 20:
            return True

    return False


def check_traffic_violation(img, vechicles, crosswalks):
    for crosswalk in crosswalks:
        for vechicle in vechicles:
            vleft, vright = vechicle.p1[0], vechicle.p2[0]
            cleft, cright = crosswalk.p1[0], crosswalk.p2[0]

            vtop, vbot = vechicle.p1[1], vechicle.p2[1]
            ctop, cbot = crosswalk.p1[1], crosswalk.p2[1]

            # if (vbot > ctop) and vleft >= cleft and vright <= cright:
            if (vbot > ctop) and vleft >= cleft and vright <= cright:
                add_only_new_vechicle(vechicle=vechicle)
                traffic_violence_box(img, vechicle.p1, vechicle.p2)
    
    return img

def traffic_violence_box(img, p1, p2):
    x, y, w, h = p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]

    sub_img = img[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    img[y:y+h, x:x+w] = res

def add_unique_vechicle(vechicle: ObjectDetected):
    
    if len(vechicles_unique_list) == 0:
        vechicle.timestart = time.time()
        vechicles_unique_list.append(vechicle)
        return

    for v1 in vechicles_unique_list:
        if v1.id == vechicle.id:
            return
            
    
    vechicle.timestart = time.time()
    vechicles_unique_list.append(vechicle)

    # print(f'Car: {car_violations}, Motorcycle: {motorcycle_violation}')

def add_only_new_vechicle(vechicle: ObjectDetected):
    
    if vechicle.class_name == 'car':
        if vechicle.id not in car_violations:
            car_violations.append(vechicle.id)
    else:
        if vechicle.id not in motorcycle_violation:
            motorcycle_violation.append(vechicle.id)

    # print(f'Car: {car_violations}, Motorcycle: {motorcycle_violation}')



parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
parser.add_argument("--detector_path", type=str, default="yolov5m6.pt", help="YOLOv5 model path")
parser.add_argument("--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)")
parser.add_argument("--conf_thres", type=float, default="0.7", help="YOLOv5 object confidence threshold")
parser.add_argument("--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS")
parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--track_points", type=str, default="centroid", help="Track points: 'centroid' or 'bbox'")
parser.add_argument('--source_type', type=str, default="images", help='source type')

args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)
source = args.source
imgsz = args.imgsz
source_type = args.source_type
print(source)



dataset = LoadImages(source, img_size=(640, 640), stride=32)

tracker = Tracker(
    distance_function=euclidean_distance,
    distance_threshold=max_distance_between_points,
)

paths_drawer = Paths(center, attenuation=0.01)
previous_vechicles = []
current_crosswalk = None

car_violations = []
motorcycle_violation = []

vechicles_unique_list = []

for path, frame, im0s, vid_cap, s in dataset:

    
    yolo_detections = model(
        im0s,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thresh,
        image_size=args.img_size,
        classes=args.classes
    )

    annotator = Annotator(im0s, line_width=1)

    detections = yolo_detections_to_norfair_detections(yolo_detections, track_points=args.track_points)

    tracked_objects = tracker.update(detections=detections)

    crosswalk_boxes = []
    vechicle_boxes = []

    for index, obj in enumerate(tracked_objects):
        boxes = obj.estimate[obj.live_points]
        if len(boxes) != 2: continue
        
        xyxy = norfair_to_yolo(boxes)
        cls = classes[obj.label]
        class_name = obj.label.lower()
        if class_name == 'crosswalk':
            label = f'{class_name}'
        else:
            # label = f'ID{obj.id}: {class_name}'
            label = f'{obj.id}'

        annotator.box_label(xyxy, label, color=colors(cls, True))

        obj_detected = ObjectDetected(img=annotator.result(), class_name=obj.label.lower(), p1=annotator.p1, p2=annotator.p2,id=obj.id,xyxy=xyxy)
        
        if obj_detected.class_name == 'crosswalk':
            crosswalk_boxes.append(obj_detected)
            current_crosswalk = obj_detected
        else:
            vechicle_boxes.append(obj_detected)
            


    if len(crosswalk_boxes) == 0 and current_crosswalk != None:
        annotator.box_label(current_crosswalk.xyxy, 'crosswalk', color=colors(2, True))
        crosswalk_boxes.append(current_crosswalk)

    im0 = annotator.result()

    stopped_vechicles = check_stopped_vechicle(vechicle_boxes, previous_vechicles)
    
    previous_vechicles = vechicle_boxes.copy()

    result = check_traffic_violation(im0, vechicles=stopped_vechicles, crosswalks=crosswalk_boxes)

    print(f"Unique count: {len(vechicles_unique_list)}")

    annotator.violence_count(len(car_violations), len(motorcycle_violation))
    cv2.imshow("Img", annotator.result())
    cv2.waitKey(1)
