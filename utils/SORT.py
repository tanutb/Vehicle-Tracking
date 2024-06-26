# REFERENCE: SORT Algorithm from https://github.com/abramjos/Writeups/blob/main/Tracker-SORT/sort.py
# Author: Abramjos
# Description: Implementation of the SORT (Simple Online and Realtime Tracking) algorithm.
# Modifications: Adapted the algorithm for specific use-case requirements.


from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
from utils.hungarian import Hungarian
from filterpy.kalman import KalmanFilter
import numpy as np


def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    boxA (list or tuple): Bounding box in xyxy format (xmin, ymin, xmax, ymax).
    boxB (list or tuple): Bounding box in xyxy format (xmin, ymin, xmax, ymax).

    Returns:
    float: IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection rectangle
    interWidth = xB - xA
    interHeight = yB - yA
    if interWidth <= 0 or interHeight <= 0:
        return 0.0
    interArea = interWidth * interHeight
    
    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Compute the union area
    unionArea = boxAArea + boxBArea - interArea
    
    # Compute the IoU
    iou = interArea / unionArea
    return iou

###### Vehicle Class
def xywh_to_xyxy(xywh):
    """
    Convert bounding box from xywh format to xyxy format.
    
    Parameters:
    xywh (list or tuple): A bounding box in xywh format (center_x, center_y, width, height).
    
    Returns:
    list: A bounding box in xyxy format (xmin, ymin, xmax, ymax).
    """
    center_x, center_y, width, height = xywh
    xmin = center_x - (width / 2)
    ymin = center_y - (height / 2)
    xmax = center_x + (width / 2)
    ymax = center_y + (height / 2)
    
    return [xmin, ymin, xmax, ymax]


class Vehicle:
    def __init__(self, position ,ID, _class) -> None:
        self.x = position[0]
        self.y = position[1]
        self.w = position[2]
        self.h = position[3]
        self.vx = 0.0
        self.vy = 0.0
        self.ID = ID
        self._max_age = 20
        self.age = 0
        self.dt = 0
        self._class = _class


        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.preprocess_kf([self.x, self.y, self.w, self.h]).reshape(4,1)
    
    def preprocess_kf(self , xywh):
        '''
        xywh -> xysr
        - s = w * h : ares
        - r = w / h : ratio
        '''
        x,y,w,h = xywh
        s = w * h    #scale is just area
        r = w / h
        return np.array([x, y, s, r])
    
    def postprocess_kf(self,xysr):
        '''
        xysr -> xywh
        '''
        x,y,s,r = xysr
        w = np.sqrt(s * r)
        h = s / w

        return np.array([x,y,w,h])

    def get_conf(self) :
        return 0
    def get_class(self):
        return self._class

    def get_xywh(self) : 
        
        # return np.array([self.x,self.y,self.w,self.h])
        return self.postprocess_kf(self.kf.x[:4])
    
    def temp_update(self,pos):
        self.x = pos[0]
        self.y = pos[1]
        self.w = pos[2]
        self.h = pos[3]

    def update(self , xywh, _class = None):
        if type(self._class) != None:
            if self._class != _class:
                print('  Class Confused in tracker {}/{}'.format(self._class,_class))
                self._class = _class

        self.kf.update(self.preprocess_kf(xywh) )


    def predict(self):

        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return self.postprocess_kf(self.kf.x[:4])

       



class Tracking:
    def __init__(self) -> None:
        self.model = YOLO('./weight/Detection_weight.pt')
        self.tracking_ob = {}
        self.ID = 0
        self.class_name = list(self.model.names)

    def detect(self, frame) :
        results = self.model(frame,verbose=False , conf=0.5)[0]
        if results.boxes is not None:
            re = {
                'ID' : [0 for _ in range(len(results.boxes.xywhn))],
                'conf' : results.boxes.conf.cpu().tolist(),
                'cls' : results.boxes.cls.cpu().tolist(),
                'xywh': results.boxes.xywhn.cpu().tolist()
            }
        else : assert "detect NONE"
        return re


    def tracking(self, frame) -> list:
        
        ### Start Detection ###
        results = self.model(frame,verbose=False , conf=0.5)[0]
        if self.tracking_ob == {} :
            ### First detection
            if results.boxes is not None:
                for i in range(len(results.boxes.xywhn)):
                    xywhn = results.boxes.xywhn[i].cpu()
                    cls = results.boxes.cls[i].cpu()
                    v = Vehicle(xywhn,self.ID,cls)
                    self.tracking_ob[self.ID] = v
                    self.ID +=1

        else : 
            if results.boxes is not None:
                bb = results.boxes.xywhn.tolist()
                cls = results.boxes.cls.cpu()

            ### 

            ID_list = []
            IOU = np.zeros((len(bb),(len(self.tracking_ob))))
            for index,ID in enumerate(self.tracking_ob):
                ID_list.append(ID)
                for j in range(len(bb)):
                    IOU[j, index] = \
                     calculate_iou(
                        xywh_to_xyxy(np.array(bb[j])),
                        xywh_to_xyxy(np.array(self.tracking_ob[ID].get_xywh()))
                        )
            
            ## hungarian 
            if min(IOU.shape) > 0:
                hungarian = Hungarian(-IOU)
                hungarian.calculate()
                association = np.array(hungarian.get_results())
            else : 
                association = np.empty(shape=(0,2))
            unmatched_detections = []
            for d, det in enumerate(bb):
                if(d not in association[:,0]):
                    unmatched_detections.append(d)  
                
            unmatched_trackers = []
            for t, trk in enumerate(self.tracking_ob):
                if(t not in association[:,1]):
                    unmatched_trackers.append(t)

            #filter out matched with low IOU
            matches = []
            for m in association:
                if(IOU[m[0], m[1]]< 0.3):
                    unmatched_detections.append(m[0])
                    unmatched_trackers.append(m[1])
                else:
                    matches.append(m.reshape(1,2))
            if(len(matches)==0):
                matches = np.empty((0,2),dtype=int)
            else:
                matches = np.concatenate(matches,axis=0)
                
            association = matches


            ## New Track
            for i in unmatched_detections :
                
                if np.any(IOU[i,:] > 0.0) : 
                    pass
                else : 
                    v = Vehicle(bb[i],self.ID,cls[i])
                    self.tracking_ob[self.ID] = v
                    self.ID +=1

            ## Missing Track
            for i in unmatched_trackers :
                self.tracking_ob[ID_list[i]].age += 1

            #### Update
            for i , j in association :
                self.tracking_ob[ID_list[j]].update(bb[i] , cls[i])


        I = list(self.tracking_ob.keys())
        for ID in reversed(I):
            xyxy_values = np.abs(xywh_to_xyxy(self.tracking_ob[ID].get_xywh()[:4]))
            if np.any(np.isclose(xyxy_values, 1, atol=0.1)):
                del self.tracking_ob[ID]
            elif self.tracking_ob[ID].age >= self.tracking_ob[ID]._max_age:
                del self.tracking_ob[ID]
            
        
        ID_list = []
        conf_list = []
        cls_list = []
        xywh_list = []
        for ID in self.tracking_ob :
            ID_list.append(ID)
            conf_list.append(self.tracking_ob[ID].get_conf())
            cls_list.append(self.tracking_ob[ID].get_class())
            xywh_list.append(self.tracking_ob[ID].predict())
        re = {
                'ID' : ID_list,
                'conf' : conf_list,
                'cls' : cls_list,
                'xywh': xywh_list
            }
        
        return re

