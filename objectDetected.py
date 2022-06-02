import numpy as np
import cv2

class ObjectDetected:
    def __init__(self, img=None,class_name=None, p1=(0,0), p2=(0,0), id=0, xyxy=None, timestart=None):
        self.class_name=class_name
        self.p1=p1
        self.p2=p2
        self.img=img
        self.x=p1[0]
        self.y=p1[1]
        self.w=p2[0]-p1[0]
        self.h=p2[1]-p1[1]
        self.id=id
        self.xyxy=xyxy
        self.timestart=timestart

    def traffic_violence_box(self):
        x, y, w, h = self.p1[0], self.p1[1], self.p2[0]-self.p1[0], self.p2[1]-self.p1[1]

        print(f'x = {x}, y = {y}, w = {w}, h = {h}')
        sub_img = self.img[y:y+h, x:x+w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        
        self.img[y:y+h, x:x+w] = res