#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys 
import random
import json
import cv2 
import base64
import requests as req
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from io import BytesIO  

#To compute intersection of lines
import shapely
from shapely.geometry import LineString, Point
from scipy.spatial import distance
#Import pose drawing module 
import import_ipynb
import DrawPose


# How to use the Gaze module:
# 
# - 1. set the vision line for the frame: 
#     - set_vision_line(posenetPred)
# - 2. Get the number of time bbox has intersected with the vision lines
#     - check_bboxs_intersection(bboxs)
#     

# In[2]:


#constants
PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]
PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)} ;PART_IDS
CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES];

CONNECTED_EYELINE_NAMES = [
    ("rightShoulder", 'leftEye') , ('rightEar', 'rightEye')
]
CONNECTED_EYELINE_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_EYELINE_NAMES];

VISION_LINE = []


# #Helper functions

# In[3]:


def get_adjacent_keypoints(keypoints):
    '''
    Helper function of draw_skel_and _kp
    Returns 2 coord of 2 points where line needs to be drawn
    EXAMPLE: [[X1,Y1],[X2,Y2]]
    '''
    results = []
    for left, right in CONNECTED_PART_INDICES:
       results.append(
           np.array([
                [ keypoints[left]['position']['x'] , keypoints[left]['position']['y'] ],
                [ keypoints[right]['position']['x'] , keypoints[right]['position']['y'] ]
                    ]
           ).astype(np.int32)
        )
    return results
            


# In[4]:


def get_adjacent_keypoint_eyeline(keypoints):
    results = []
    for left, right in CONNECTED_EYELINE_INDICES:
       results.append(
           np.array([
                [ keypoints[left]['position']['x'] , keypoints[left]['position']['y'] ],
                [ keypoints[right]['position']['x'] , keypoints[right]['position']['y'] ]
                    ]
           ).astype(np.int32)
        )
    return results


# In[5]:


def draw_extended_line(out_img,p1,p2,color):
    '''
    Helper function of draw_visibility_cone
    Given 2 points and 1 image, draw a lines extends it
    '''
    theta1 = np.arctan2(p1[1]-p2[1], p1[0]-p2[0]) 
    endpt_x = int(p1[0] - 1000*np.cos(theta1))
    endpt_y = int(p1[1] - 1000*np.sin(theta1))
    cv2.line(out_img, (p1[0], p1[1]), (endpt_x, endpt_y), color, 2)
    
    return out_img,theta1,[endpt_x, endpt_y]

def get_extended_line(p1,p2):
    '''
    Helper function of draw_visibility_cone
    Given 2 points and 1 image, draw a lines extends it
    '''
    theta1 = np.arctan2(p1[1]-p2[1], p1[0]-p2[0]) 
    endpt_x = int(p1[0] - 10000*np.cos(theta1))
    endpt_y = int(p1[1] - 10000*np.sin(theta1))
    extended_line = get_line_object(p1,[endpt_x, endpt_y])
    return theta1,extended_line


# In[6]:


def get_line_object(p1,p2):
    '''
    Helper function of get_line_intersection
    This function creates a line object
    Parameter
    -------------
    p1 : A list. [left,top] or [x,y] or [width,hegiht] or [col,row]
    p2 : A list. [left,top] or [x,y] or [width,hegiht] or [col,row]
    
    Return
    -------
    Line : A line object
    ''' 
    return LineString([[p1[0] , p1[1]], [p2[0] , p2[1]] ])
    

def get_line_intersection(line1,line2):
    '''
    Helper function of drawvisibilty_cone
    This function finds  2 line intersects or not.
    Parameter 
    ---------
    line1 : A line object
    line2 :  A line object
    
    Return
    ------------
    p5 : The list containing intersection point [x,y]
    
    False  : There is no intersection 
   
    '''
    try:
        int_pt = line1.intersection(line2)
        #print(int_pt)
        p5 = [int_pt.x, int_pt.y]        
    except AttributeError:
        #print("Returing None")
        return False
    return p5


# In[7]:


def get_general_line(p5,theta3):
    '''
    This function find the the middle/general line
    Parameter
    ---------
    p5 : The intersection point of the player cone lines 
    theta3 : The angle that the new line need to be in
    
    Return 
    -----------
    line1 : A line object. The middle line
    '''
    endpt_x = int(p5[0] - 1000*np.cos(theta3))
    endpt_y = int(p5[1] - 1000*np.sin(theta3))
    line1 = get_line_object([p5[0] , p5[1]], [endpt_x , endpt_y])
    return line1
def draw_line(img,line,color):
    p1,p2= (int(line.coords[0][0]) , int(line.coords[0][1])) ,(int(line.coords[1][0]) , int(line.coords[1][1])) 
    cv2.line(img, p1, p2, color, 2)
    
    


# In[8]:


def get_cone_line_intersection(keypoints):
    '''
    Helper function of set_vision_lines()
    Parameter
    ---------
    keypoints : The keypoint dictionary containing all the keypoints 
    
    Return 
    ---------
     p5 : The intersection point. False,There is no intersection 
     theta1 : Angle(radians) between the horizontal and line1
     theta2 : Angle(radians) between the horizontal and line2
    '''
    new_keypoints = get_adjacent_keypoint_eyeline(keypoints)
    p1,p2 = new_keypoints[0][0],new_keypoints[0][1]
    p3,p4 = new_keypoints[1][0],new_keypoints[1][1]
    #Get the cone line intersection
    theta1,line1 = get_extended_line(p1,p2)
    theta2,line2 = get_extended_line(p3,p4)
    p5 = get_line_intersection(line1,line2)
    return p5,theta1,theta2


def set_vision_lines(posenetPred):
    '''
    This function will update the GLOBAL VARIABLE VISION_LINE
    It will remove the vision line for the last frame
    It will add vision line for this frame for all the players
    
    Parameter
    ----------
    posenetPred : The output from poenet. This should contain keypoints that are relative to
                  the big image. 
                  
    
    '''
    global VISION_LINE
    VISION_LINE = []
    for pose in posenetPred['detectionList']:
        keypoints = pose['keypoints']
        #Get cone line intersection point
        p5,theta1,theta2 = get_cone_line_intersection(keypoints)
       # get mid line 
        if p5 != False:
            theta3 = (theta1 + theta2)/2
            general_line = get_general_line(p5, theta3)
            VISION_LINE.append(general_line)
        else:
            VISION_LINE.append(False)
    print(f"Initalized {len(VISION_LINE)} player gaze lines.")


# In[9]:


def get_bbox_coords(bbox):
    '''
    This function will return all the 4 coords of the bouding boxes
    Parameter
    ---------
    bbox : [left, top ,right, bottom]
    
    Return
    --------
    list : [left_top , right_bottom , left_bottom , right_top]
    '''
    left_top,right_bottom = (bbox[0] , bbox[1]), (bbox[2] , bbox[3])
    left_bottom , right_top = (left_top[0] ,right_bottom[1]) , (right_bottom[0] , left_top[1])
    return [left_top , right_bottom , left_bottom , right_top]
    
    
    
def get_bbox_diagonals(left_top , right_bottom , left_bottom , right_top):
    '''
    This fuction finds the 2 diagonal of the bounding boxes
    Parameter
    ----------
    left_top , right_bottom , left_bottom , right_top : The 4 points of the bbox
    
    Return
    --------
    diagonal1 : A line object. It starts from left_top , right_bottom 
    diagonal2 : A line object. It starts from left_bottom , right_top
    '''
    diagonal1 = get_line_object(left_top , right_bottom)
    diagonal2 = get_line_object(left_bottom , right_top)
    return diagonal1 , diagonal2


def check_bbox_intersection(bbox , lines):
    '''
    Helper function of check_bboxs_intersection
    This function find out the number of vision line that intersect with bbox
    
    Parameter
    -----------
    bbox : A list [left,top,right,botoom] coord in order
    lines : VISION_LINE, where the vision line of the player in the bbox is set to False. 
            Because we dont want to consider it's intersection with it self
    '''
    count = 0
    bbox_points = get_bbox_coords(bbox)
    diagonal1, diagonal2 = get_bbox_diagonals(*bbox_points)
    for line in lines:
        if line != False: 
            intersection1 = get_line_intersection(diagonal1,line)
            intersection2 = get_line_intersection(diagonal2,line)
            if intersection1 or intersection2:
                count += 1
    return count


def check_bboxs_intersection(bboxs):
    '''
    This function finds the number of VISION_LINES each bboxs is intersected by 
    Parameter
    ----------
    bboxs : A list with bbox. 
            Example: [[left,top,right,botoom],.......[left,top,right,botoom],.. ] 
            
    Return
    ---------
    counter : A list with the number of VISION_LINES each bboxs is intersected by
              Example: [8,....]
    '''
    global VISION_LINE
    counter = []
    for index,bbox in enumerate(bboxs):
        VISION_LINE_COPY = VISION_LINE[:]
        VISION_LINE_COPY[index] = False
        count = check_bbox_intersection(bbox, VISION_LINE_COPY)
        counter.append(count)
    else:
        counter.append(-1)
    return counter


# #Filtering counter with action

# In[10]:


ACTION_PRIORITY = {"shoot":0,"ball in hand":1,"block":2,"pass":3,"dribble":4,"run":5,"walk":6,
                  "defense":7,"pick":8,"no_action":9,"discard":10,"unknown":11}


# In[11]:


def getIndexsOfMaxs(counter):
    '''
    Helper function of sortGazeModulePred()
    This function finds the indexs of pose who has the most line intersection
    
    Parameter
    ---------
    counter :  A list with the number of VISION_LINES each bboxs is intersected by
              Example: [8,....]
    
    Return
    -------
    indexs : A list with index 
             Example: [1,9..]
    '''
    maxIntersection = max(counter)
    indexs = [i for i,v in enumerate(counter) if v == maxIntersection]
    return indexs

def getMaxsAction(actions,maxsIndexs):
    '''
    Helper function of sortGazeModulePred()
    This function extracts the actions of bounding with max intersection
    
    Parameter
    ----------
    actions : A list with action class for all the poses in posenetPred that was passed
             Example: ['dribble','run'...]
    
    maxsIndexs :  A list with index of poses(with most visionLine intersection)
                  Example: [1,9..]
    
    Return
    --------
    maxActions : A list of action class  for all the poses with max intersection
                 Example: ['dribble','run'...]
            
    '''
    maxActions = [actions[index] for index in maxsIndexs]
    return maxActions
  
def sortGazeModulePred(actions, counter):
    '''
    Helper function of fitGazeModule()
    This function will sort the indexs of poses with max intersection based on ACTION_PRIORITY
    
    Parameter
    ---------
    actions : A list with action class for all the poses in posenetPred that was passed
             Example: ['dribble','run'...]
    
    counter :  A list with the number of VISION_LINES each bboxs is intersected by
              Example: [8,....]
    
    Return
    --------
    sortedMaxsIndex :  A list with index of poses(with most visionLine intersection) that is
                       sorted based on ACTION_PRIORITY. 
                       Example: [1,9..]
    
    '''
    global ACTION_PRIORITY
    maxsIndexs = getIndexsOfMaxs(counter)
    maxsActions = getMaxsAction(actions,maxsIndexs)
    maxsActionsPriority = [ACTION_PRIORITY[action] for action in maxsActions]
    sortedMaxsIndex = [maxIndex for _,maxIndex in sorted(zip(maxsActionsPriority,maxsIndexs))]
    return sortedMaxsIndex

def getBallLocation(pose):
    '''
    Helper function of fitGazeModule()
    This function will estimate the ball position based on the player/pose
    
    Parameter
    ----------
    pose : A dictionary containing keypoints 
    
    Return
    -------
    (x,y) : A tuple containg (width,height) or (col,row)
    '''
    #right-rist coord
    x,y= int(pose['keypoints'][10]["position"]['x']) , int(pose['keypoints'][10]["position"]['y'])
    return (x,y)

def fitGazeModule(posenetPredCombined,frame_bboxs,actions):
    '''
    This function will find ball position and index of the pose that has the ball
    
    Parameter
    ----------
    posenetPredCombined : The output from posenet server. 
    (combined means posesPred of individual cropped images with combined to 1 dict)
    
    frame_bboxs : A list with bbox. 
                Example: [[left,top,right,botoom],.......[left,top,right,botoom],.. ] 
    
    actions : A list with action class for all the poses in posenetPred that was passed
             Example: ['dribble','run'...]
             
    Return
    --------
    ball_position : A tuple of integers
                    Example:(width,height) or (col,row) or (x,y)
                    (x,y) because the cv2 draw takes (col,row)
    
    scaler :  Integer that represnets the index of the pose that has the ball in 
              posenetPredCombinede
    '''
    set_vision_lines(posenetPredCombined)
    counter = check_bboxs_intersection(frame_bboxs)
    sortedMaxsIndex = sortGazeModulePred(actions,counter)
    ball_position = getBallLocation(posenetPredCombined['detectionList'][sortedMaxsIndex[0]])
    return ball_position , sortedMaxsIndex[0]


# #Test Data

# In[12]:


b = [[522, 200, 590, 355], [660, 240, 724, 421], [286, 295, 354, 416], [103, 196, 175, 367], [422, 121, 473, 246], [43, 256, 112, 410], [185, 169, 251, 317], [717, 102, 784, 294], [0, 418, 70, 480]]


# In[13]:


p = {'detectionList': [{'score': 0.8128076037939858, 'keypoints': [{'score': 0.9568926095962524, 'part': 'nose', 'position': {'x': 572, 'y': 224}},
                                                                   {'score': 0.1137080192565918, 'part': 'leftEye', 'position': {'x': 572, 'y': 221}},
                                                                   {'score': 0.9391862750053406, 'part': 'rightEye', 'position': {'x': 571, 'y': 221}},
                                                                   {'score': 0.025502651929855347, 'part': 'leftEar', 'position': {'x': 562, 'y': 222}},
                                                                   {'score': 0.9640204310417175, 'part': 'rightEar', 'position': {'x': 565, 'y': 222}},
                                                                   {'score': 0.8656671047210693, 'part': 'leftShoulder', 'position': {'x': 556, 'y': 234}}, 
                                                                   {'score': 0.9993869066238403, 'part': 'rightShoulder', 'position': {'x': 558, 'y': 235}},
                                                                   {'score': 0.37945497035980225, 'part': 'leftElbow', 'position': {'x': 553, 'y': 265}}, 
                                                                   {'score': 0.9900776147842407, 'part': 'rightElbow', 'position': {'x': 554, 'y': 260}}, 
                                                                   {'score': 0.7229613661766052, 'part': 'leftWrist', 'position': {'x': 559, 'y': 280}},
                                                                   {'score': 0.958452582359314, 'part': 'rightWrist', 'position': {'x': 560, 'y': 282}}, {'score': 0.937991738319397, 'part': 'leftHip', 'position': {'x': 536, 'y': 267}}, {'score': 0.9963494539260864, 'part': 'rightHip', 'position': {'x': 540, 'y': 269}}, {'score': 0.9988662600517273, 'part': 'leftKnee', 'position': {'x': 541, 'y': 299}}, {'score': 0.9986031651496887, 'part': 'rightKnee', 'position': {'x': 553, 'y': 303}}, {'score': 0.9750385880470276, 'part': 'leftAnkle', 'position': {'x': 536, 'y': 325}}, {'score': 0.9955695271492004, 'part': 'rightAnkle', 'position': {'x': 548, 'y': 338}}]}, {'score': 0.7336933367392596, 'keypoints': [{'score': 0.017970144748687744, 'part': 'nose', 'position': {'x': 685, 'y': 264}}, {'score': 0.008096635341644287, 'part': 'leftEye', 'position': {'x': 683, 'y': 263}}, {'score': 0.012565791606903076, 'part': 'rightEye', 'position': {'x': 690, 'y': 264}}, {'score': 0.8024667501449585, 'part': 'leftEar', 'position': {'x': 682, 'y': 265}}, {'score': 0.822800874710083, 'part': 'rightEar', 'position': {'x': 695, 'y': 265}}, {'score': 0.9951857328414917, 'part': 'leftShoulder', 'position': {'x': 676, 'y': 282}}, {'score': 0.9927878379821777, 'part': 'rightShoulder', 'position': {'x': 702, 'y': 284}}, {'score': 0.8689776062965393, 'part': 'leftElbow', 'position': {'x': 668, 'y': 306}}, {'score': 0.9886101484298706, 'part': 'rightElbow', 'position': {'x': 707, 'y': 312}}, {'score': 0.17667239904403687, 'part': 'leftWrist', 'position': {'x': 671, 'y': 320}}, {'score': 0.9933490753173828, 'part': 'rightWrist', 'position': {'x': 706, 'y': 336}}, {'score': 0.9456639885902405, 'part': 'leftHip', 'position': {'x': 679, 'y': 328}}, {'score': 0.9749963283538818, 'part': 'rightHip', 'position': {'x': 697, 'y': 330}}, {'score': 0.9958609342575073, 'part': 'leftKnee', 'position': {'x': 680, 'y': 368}}, {'score': 0.9854234457015991, 'part': 'rightKnee', 'position': {'x': 700, 'y': 369}}, {'score': 0.9208113551139832, 'part': 'leftAnkle', 'position': {'x': 681, 'y': 403}}, {'score': 0.9705476760864258, 'part': 'rightAnkle', 'position': {'x': 701, 'y': 404}}]}, {'score': 0.7233603789525873, 'keypoints': [{'score': 0.01018369197845459, 'part': 'nose', 'position': {'x': 317, 'y': 318}}, {'score': 0.0022349953651428223, 'part': 'leftEye', 'position': {'x': 313, 'y': 315}}, {'score': 0.014854460954666138, 'part': 'rightEye', 'position': {'x': 320, 'y': 316}}, {'score': 0.9358092546463013, 'part': 'leftEar', 'position': {'x': 308, 'y': 317}}, {'score': 0.9647722244262695, 'part': 'rightEar', 'position': {'x': 319, 'y': 317}}, {'score': 0.9997965693473816, 'part': 'leftShoulder', 'position': {'x': 299, 'y': 333}}, {'score': 0.9996219873428345, 'part': 'rightShoulder', 'position': {'x': 326, 'y': 333}}, {'score': 0.9916865229606628, 'part': 'leftElbow', 'position': {'x': 296, 'y': 359}}, {'score': 0.9974299669265747, 'part': 'rightElbow', 'position': {'x': 332, 'y': 357}}, {'score': 0.9664741158485413, 'part': 'leftWrist', 'position': {'x': 300, 'y': 377}}, {'score': 0.9982190132141113, 'part': 'rightWrist', 'position': {'x': 338, 'y': 378}}, {'score': 0.9899277091026306, 'part': 'leftHip', 'position': {'x': 306, 'y': 376}}, {'score': 0.9903024435043335, 'part': 'rightHip', 'position': {'x': 323, 'y': 376}}, {'score': 0.9591392874717712, 'part': 'leftKnee', 'position': {'x': 310, 'y': 406}}, {'score': 0.9796425104141235, 'part': 'rightKnee', 'position': {'x': 326, 'y': 405}}, {'score': 0.07908296585083008, 'part': 'leftAnkle', 'position': {'x': 309, 'y': 423}}, {'score': 0.41794872283935547, 'part': 'rightAnkle', 'position': {'x': 326, 'y': 424}}]}, {'score': 0.786461270907346, 'keypoints': [{'score': 0.5958542227745056, 'part': 'nose', 'position': {'x': 148, 'y': 223}}, {'score': 0.009891480207443237, 'part': 'leftEye', 'position': {'x': 146, 'y': 221}}, {'score': 0.49513232707977295, 'part': 'rightEye', 'position': {'x': 145, 'y': 221}}, {'score': 0.17179739475250244, 'part': 'leftEar', 'position': {'x': 131, 'y': 224}}, {'score': 0.9379146099090576, 'part': 'rightEar', 'position': {'x': 138, 'y': 224}}, {'score': 0.975072979927063, 'part': 'leftShoulder', 'position': {'x': 125, 'y': 240}}, {'score': 0.9905099272727966, 'part': 'rightShoulder', 'position': {'x': 144, 'y': 238}}, {'score': 0.7258508205413818, 'part': 'leftElbow', 'position': {'x': 121, 'y': 262}}, {'score': 0.9442204236984253, 'part': 'rightElbow', 'position': {'x': 146, 'y': 263}}, {'score': 0.5896404981613159, 'part': 'leftWrist', 'position': {'x': 128, 'y': 271}}, {'score': 0.9617345333099365, 'part': 'rightWrist', 'position': {'x': 150, 'y': 279}}, {'score': 0.9959503412246704, 'part': 'leftHip', 'position': {'x': 129, 'y': 278}}, {'score': 0.9956531524658203, 'part': 'rightHip', 'position': {'x': 143, 'y': 275}}, {'score': 0.9998238682746887, 'part': 'leftKnee', 'position': {'x': 128, 'y': 310}}, {'score': 0.9984769821166992, 'part': 'rightKnee', 'position': {'x': 149, 'y': 309}}, {'score': 0.9932644963264465, 'part': 'leftAnkle', 'position': {'x': 120, 'y': 341}}, {'score': 0.9890535473823547, 'part': 'rightAnkle', 'position': {'x': 149, 'y': 345}}]}, {'score': 0.7328198429416207, 'keypoints': [{'score': 0.1354440450668335, 'part': 'nose', 'position': {'x': 464, 'y': 146}}, {'score': 0.002656102180480957, 'part': 'leftEye', 'position': {'x': 464, 'y': 144}}, {'score': 0.4120482802391052, 'part': 'rightEye', 'position': {'x': 465, 'y': 144}}, {'score': 0.1845640242099762, 'part': 'leftEar', 'position': {'x': 454, 'y': 142}}, {'score': 0.9192991256713867, 'part': 'rightEar', 'position': {'x': 461, 'y': 144}}, {'score': 0.9387127161026001, 'part': 'leftShoulder', 'position': {'x': 444, 'y': 149}}, {'score': 0.9962905645370483, 'part': 'rightShoulder', 'position': {'x': 450, 'y': 152}}, {'score': 0.41550958156585693, 'part': 'leftElbow', 'position': {'x': 447, 'y': 174}}, {'score': 0.9962441921234131, 'part': 'rightElbow', 'position': {'x': 439, 'y': 173}}, {'score': 0.8206527233123779, 'part': 'leftWrist', 'position': {'x': 451, 'y': 187}}, {'score': 0.9859794974327087, 'part': 'rightWrist', 'position': {'x': 440, 'y': 190}}, {'score': 0.9732809066772461, 'part': 'leftHip', 'position': {'x': 436, 'y': 179}}, {'score': 0.9876495599746704, 'part': 'rightHip', 'position': {'x': 438, 'y': 180}}, {'score': 0.9847617149353027, 'part': 'leftKnee', 'position': {'x': 439, 'y': 202}}, {'score': 0.9893618226051331, 'part': 'rightKnee', 'position': {'x': 447, 'y': 205}}, {'score': 0.7181053161621094, 'part': 'leftAnkle', 'position': {'x': 434, 'y': 222}}, {'score': 0.9973771572113037, 'part': 'rightAnkle', 'position': {'x': 444, 'y': 231}}]}, {'score': 0.7578465833383448, 'keypoints': [{'score': 0.43723151087760925, 'part': 'nose', 'position': {'x': 78, 'y': 275}}, {'score': 0.010233044624328613, 'part': 'leftEye', 'position': {'x': 78, 'y': 272}}, {'score': 0.7422469258308411, 'part': 'rightEye', 'position': {'x': 77, 'y': 273}}, {'score': 0.1277085840702057, 'part': 'leftEar', 'position': {'x': 67, 'y': 273}}, {'score': 0.9521619081497192, 'part': 'rightEar', 'position': {'x': 73, 'y': 274}}, {'score': 0.9967963695526123, 'part': 'leftShoulder', 'position': {'x': 54, 'y': 288}}, {'score': 0.9993197321891785, 'part': 'rightShoulder', 'position': {'x': 75, 'y': 292}}, {'score': 0.4407055377960205, 'part': 'leftElbow', 'position': {'x': 50, 'y': 306}}, {'score': 0.9932873249053955, 'part': 'rightElbow', 'position': {'x': 78, 'y': 317}}, {'score': 0.3313325047492981, 'part': 'leftWrist', 'position': {'x': 64, 'y': 322}}, {'score': 0.9975097179412842, 'part': 'rightWrist', 'position': {'x': 90, 'y': 334}}, {'score': 0.9901885986328125, 'part': 'leftHip', 'position': {'x': 51, 'y': 327}}, {'score': 0.9929696321487427, 'part': 'rightHip', 'position': {'x': 64, 'y': 329}}, {'score': 0.9699243307113647, 'part': 'leftKnee', 'position': {'x': 59, 'y': 363}}, {'score': 0.997429609298706, 'part': 'rightKnee', 'position': {'x': 64, 'y': 364}}, {'score': 0.9747611284255981, 'part': 'leftAnkle', 'position': {'x': 57, 'y': 389}}, {'score': 0.9295854568481445, 'part': 'rightAnkle', 'position': {'x': 53, 'y': 391}}]}, {'score': 0.9003641570315641, 'keypoints': [{'score': 0.9966267943382263, 'part': 'nose', 'position': {'x': 218, 'y': 192}}, {'score': 0.9904731512069702, 'part': 'leftEye', 'position': {'x': 221, 'y': 190}}, {'score': 0.9927913546562195, 'part': 'rightEye', 'position': {'x': 216, 'y': 189}}, {'score': 0.7534298896789551, 'part': 'leftEar', 'position': {'x': 223, 'y': 191}}, {'score': 0.9424461126327515, 'part': 'rightEar', 'position': {'x': 210, 'y': 190}}, {'score': 0.998862087726593, 'part': 'leftShoulder', 'position': {'x': 226, 'y': 203}}, {'score': 0.9966269731521606, 'part': 'rightShoulder', 'position': {'x': 203, 'y': 203}}, {'score': 0.9979666471481323, 'part': 'leftElbow', 'position': {'x': 234, 'y': 220}}, {'score': 0.9637047648429871, 'part': 'rightElbow', 'position': {'x': 194, 'y': 224}}, {'score': 0.9891492128372192, 'part': 'leftWrist', 'position': {'x': 240, 'y': 233}}, {'score': 0.6244813799858093, 'part': 'rightWrist', 'position': {'x': 196, 'y': 230}}, {'score': 0.953237771987915, 'part': 'leftHip', 'position': {'x': 216, 'y': 237}}, {'score': 0.9806000590324402, 'part': 'rightHip', 'position': {'x': 202, 'y': 235}}, {'score': 0.9550578594207764, 'part': 'leftKnee', 'position': {'x': 208, 'y': 268}}, {'score': 0.8570579290390015, 'part': 'rightKnee', 'position': {'x': 203, 'y': 269}}, {'score': 0.8889736533164978, 'part': 'leftAnkle', 'position': {'x': 206, 'y': 291}}, {'score': 0.42470502853393555, 'part': 'rightAnkle', 'position': {'x': 204, 'y': 295}}]}, {'score': 0.8234139312716091, 'keypoints': [{'score': 0.8058501482009888, 'part': 'nose', 'position': {'x': 748, 'y': 156}}, {'score': 0.25563186407089233, 'part': 'leftEye', 'position': {'x': 751, 'y': 156}}, {'score': 0.46217504143714905, 'part': 'rightEye', 'position': {'x': 747, 'y': 155}}, {'score': 0.5484194159507751, 'part': 'leftEar', 'position': {'x': 755, 'y': 158}}, {'score': 0.15256261825561523, 'part': 'rightEar', 'position': {'x': 745, 'y': 155}}, {'score': 0.9785213470458984, 'part': 'leftShoulder', 'position': {'x': 762, 'y': 168}}, {'score': 0.9703998565673828, 'part': 'rightShoulder', 'position': {'x': 744, 'y': 166}}, {'score': 0.9958829283714294, 'part': 'leftElbow', 'position': {'x': 762, 'y': 172}}, {'score': 0.9740769863128662, 'part': 'rightElbow', 'position': {'x': 741, 'y': 164}}, {'score': 0.9443751573562622, 'part': 'leftWrist', 'position': {'x': 757, 'y': 159}}, {'score': 0.9271751642227173, 'part': 'rightWrist', 'position': {'x': 736, 'y': 143}}, {'score': 0.9974187016487122, 'part': 'leftHip', 'position': {'x': 756, 'y': 205}}, {'score': 0.9976546764373779, 'part': 'rightHip', 'position': {'x': 741, 'y': 203}}, {'score': 0.9981964826583862, 'part': 'leftKnee', 'position': {'x': 751, 'y': 235}}, {'score': 0.9983636140823364, 'part': 'rightKnee', 'position': {'x': 738, 'y': 233}}, {'score': 0.9950243234634399, 'part': 'leftAnkle', 'position': {'x': 753, 'y': 265}}, {'score': 0.9963085055351257, 'part': 'rightAnkle', 'position': {'x': 737, 'y': 262}}]}, {'score': 0.4068865478038788, 'keypoints': [{'score': 0.9981329441070557, 'part': 'nose', 'position': {'x': 29, 'y': 446}}, {'score': 0.9998471736907959, 'part': 'leftEye', 'position': {'x': 34, 'y': 444}}, {'score': 0.9527695178985596, 'part': 'rightEye', 'position': {'x': 28, 'y': 442}}, {'score': 0.9857003688812256, 'part': 'leftEar', 'position': {'x': 41, 'y': 446}}, {'score': 0.01755395531654358, 'part': 'rightEar', 'position': {'x': 26, 'y': 441}}, {'score': 0.9330771565437317, 'part': 'leftShoulder', 'position': {'x': 47, 'y': 460}}, {'score': 0.8623121976852417, 'part': 'rightShoulder', 'position': {'x': 16, 'y': 458}}, {'score': 0.12711289525032043, 'part': 'leftElbow', 'position': {'x': 50, 'y': 477}}, {'score': 0.1893087923526764, 'part': 'rightElbow', 'position': {'x': 9, 'y': 476}}, {'score': 0.1686551570892334, 'part': 'leftWrist', 'position': {'x': 46, 'y': 477}}, {'score': 0.046192944049835205, 'part': 'rightWrist', 'position': {'x': 18, 'y': 482}}, {'score': 0.35544249415397644, 'part': 'leftHip', 'position': {'x': 39, 'y': 483}}, {'score': 0.2762615382671356, 'part': 'rightHip', 'position': {'x': 22, 'y': 486}}, {'score': 0.0028268098831176758, 'part': 'leftKnee', 'position': {'x': 40, 'y': 480}}, {'score': 0.0010389089584350586, 'part': 'rightKnee', 'position': {'x': 23, 'y': 484}}, {'score': 0.0007581710815429688, 'part': 'leftAnkle', 'position': {'x': 41, 'y': 485}}, {'score': 8.028745651245117e-05, 'part': 'rightAnkle', 'position': {'x': 23, 'y': 485}}]}]}


# In[14]:


a = ['dribble', 'pass', 'no_action', 'defense', 'defense', 'dribble', 'run', 'block', 'walk']


# In[15]:


p = {'detectionList': [{'score': 0.7833564474302179, 'keypoints': [{'score': 0.8145382404327393, 'part': 'nose', 'position': {'x': 572, 'y': 225}}, {'score': 0.0926288366317749, 'part': 'leftEye', 'position': {'x': 572, 'y': 222}}, {'score': 0.7632265686988831, 'part': 'rightEye', 'position': {'x': 571, 'y': 222}}, {'score': 0.03102251887321472, 'part': 'leftEar', 'position': {'x': 561, 'y': 223}}, {'score': 0.8718037605285645, 'part': 'rightEar', 'position': {'x': 565, 'y': 222}}, {'score': 0.9094659090042114, 'part': 'leftShoulder', 'position': {'x': 559, 'y': 235}}, {'score': 0.9937941431999207, 'part': 'rightShoulder', 'position': {'x': 556, 'y': 234}}, {'score': 0.6076040863990784, 'part': 'leftElbow', 'position': {'x': 556, 'y': 260}}, {'score': 0.9070342779159546, 'part': 'rightElbow', 'position': {'x': 552, 'y': 258}}, {'score': 0.7013438940048218, 'part': 'leftWrist', 'position': {'x': 558, 'y': 274}}, {'score': 0.6796921491622925, 'part': 'rightWrist', 'position': {'x': 557, 'y': 275}}, {'score': 0.9837402105331421, 'part': 'leftHip', 'position': {'x': 539, 'y': 268}}, {'score': 0.9933797121047974, 'part': 'rightHip', 'position': {'x': 538, 'y': 268}}, {'score': 0.9966810941696167, 'part': 'leftKnee', 'position': {'x': 542, 'y': 296}}, {'score': 0.998536229133606, 'part': 'rightKnee', 'position': {'x': 554, 'y': 302}}, {'score': 0.9763523936271667, 'part': 'leftAnkle', 'position': {'x': 536, 'y': 325}}, {'score': 0.9962155818939209, 'part': 'rightAnkle', 'position': {'x': 548, 'y': 337}}]}, {'score': 0.7479874000829809, 'keypoints': [{'score': 0.010827720165252686, 'part': 'nose', 'position': {'x': 686, 'y': 264}}, {'score': 0.005038142204284668, 'part': 'leftEye', 'position': {'x': 684, 'y': 262}}, {'score': 0.009267568588256836, 'part': 'rightEye', 'position': {'x': 689, 'y': 263}}, {'score': 0.735933780670166, 'part': 'leftEar', 'position': {'x': 683, 'y': 264}}, {'score': 0.7378385663032532, 'part': 'rightEar', 'position': {'x': 695, 'y': 265}}, {'score': 0.9950188398361206, 'part': 'leftShoulder', 'position': {'x': 675, 'y': 282}}, {'score': 0.9854587912559509, 'part': 'rightShoulder', 'position': {'x': 701, 'y': 283}}, {'score': 0.9366270303726196, 'part': 'leftElbow', 'position': {'x': 668, 'y': 307}}, {'score': 0.9394189119338989, 'part': 'rightElbow', 'position': {'x': 705, 'y': 310}}, {'score': 0.5084448456764221, 'part': 'leftWrist', 'position': {'x': 671, 'y': 319}}, {'score': 0.9748278856277466, 'part': 'rightWrist', 'position': {'x': 706, 'y': 336}}, {'score': 0.9587023854255676, 'part': 'leftHip', 'position': {'x': 680, 'y': 327}}, {'score': 0.9719808101654053, 'part': 'rightHip', 'position': {'x': 697, 'y': 327}}, {'score': 0.9969696402549744, 'part': 'leftKnee', 'position': {'x': 682, 'y': 367}}, {'score': 0.9938398599624634, 'part': 'rightKnee', 'position': {'x': 701, 'y': 367}}, {'score': 0.9837883710861206, 'part': 'leftAnkle', 'position': {'x': 682, 'y': 397}}, {'score': 0.9718026518821716, 'part': 'rightAnkle', 'position': {'x': 701, 'y': 401}}]}, {'score': 0.667756925610935, 'keypoints': [{'score': 0.011418372392654419, 'part': 'nose', 'position': {'x': 318, 'y': 320}}, {'score': 0.0025731921195983887, 'part': 'leftEye', 'position': {'x': 319, 'y': 319}}, {'score': 0.020744770765304565, 'part': 'rightEye', 'position': {'x': 320, 'y': 318}}, {'score': 0.906981885433197, 'part': 'leftEar', 'position': {'x': 308, 'y': 317}}, {'score': 0.9274094104766846, 'part': 'rightEar', 'position': {'x': 320, 'y': 318}}, {'score': 0.9994644522666931, 'part': 'leftShoulder', 'position': {'x': 299, 'y': 333}}, {'score': 0.9984629154205322, 'part': 'rightShoulder', 'position': {'x': 326, 'y': 334}}, {'score': 0.9468675851821899, 'part': 'leftElbow', 'position': {'x': 296, 'y': 358}}, {'score': 0.9836893677711487, 'part': 'rightElbow', 'position': {'x': 333, 'y': 358}}, {'score': 0.8349813222885132, 'part': 'leftWrist', 'position': {'x': 299, 'y': 375}}, {'score': 0.9913501739501953, 'part': 'rightWrist', 'position': {'x': 339, 'y': 378}}, {'score': 0.9890222549438477, 'part': 'leftHip', 'position': {'x': 305, 'y': 379}}, {'score': 0.9770079851150513, 'part': 'rightHip', 'position': {'x': 324, 'y': 379}}, {'score': 0.7337028980255127, 'part': 'leftKnee', 'position': {'x': 309, 'y': 409}}, {'score': 0.9025148749351501, 'part': 'rightKnee', 'position': {'x': 326, 'y': 407}}, {'score': 0.023043423891067505, 'part': 'leftAnkle', 'position': {'x': 308, 'y': 424}}, {'score': 0.10263285040855408, 'part': 'rightAnkle', 'position': {'x': 326, 'y': 424}}]}, {'score': 0.690222350990071, 'keypoints': [{'score': 0.3313853144645691, 'part': 'nose', 'position': {'x': 133, 'y': 232}}, {'score': 0.12640085816383362, 'part': 'leftEye', 'position': {'x': 133, 'y': 230}}, {'score': 0.0969846248626709, 'part': 'rightEye', 'position': {'x': 133, 'y': 230}}, {'score': 0.45396551489830017, 'part': 'leftEar', 'position': {'x': 129, 'y': 230}}, {'score': 0.6423051357269287, 'part': 'rightEar', 'position': {'x': 137, 'y': 231}}, {'score': 0.9870813488960266, 'part': 'leftShoulder', 'position': {'x': 124, 'y': 244}}, {'score': 0.9726269245147705, 'part': 'rightShoulder', 'position': {'x': 147, 'y': 241}}, {'score': 0.7856081128120422, 'part': 'leftElbow', 'position': {'x': 118, 'y': 261}}, {'score': 0.8086168169975281, 'part': 'rightElbow', 'position': {'x': 149, 'y': 256}}, {'score': 0.2625052332878113, 'part': 'leftWrist', 'position': {'x': 124, 'y': 270}}, {'score': 0.3280481696128845, 'part': 'rightWrist', 'position': {'x': 148, 'y': 276}}, {'score': 0.9941620230674744, 'part': 'leftHip', 'position': {'x': 131, 'y': 280}}, {'score': 0.9862638711929321, 'part': 'rightHip', 'position': {'x': 144, 'y': 282}}, {'score': 0.9995994567871094, 'part': 'leftKnee', 'position': {'x': 127, 'y': 311}}, {'score': 0.9987181425094604, 'part': 'rightKnee', 'position': {'x': 150, 'y': 309}}, {'score': 0.9868148565292358, 'part': 'leftAnkle', 'position': {'x': 120, 'y': 341}}, {'score': 0.9726935625076294, 'part': 'rightAnkle', 'position': {'x': 149, 'y': 344}}]}, {'score': 0.747470448998844, 'keypoints': [{'score': 0.09487265348434448, 'part': 'nose', 'position': {'x': 461, 'y': 145}}, {'score': 0.047620415687561035, 'part': 'leftEye', 'position': {'x': 460, 'y': 144}}, {'score': 0.10721024870872498, 'part': 'rightEye', 'position': {'x': 461, 'y': 143}}, {'score': 0.15316346287727356, 'part': 'leftEar', 'position': {'x': 453, 'y': 142}}, {'score': 0.6769036650657654, 'part': 'rightEar', 'position': {'x': 459, 'y': 145}}, {'score': 0.9524451494216919, 'part': 'leftShoulder', 'position': {'x': 446, 'y': 152}}, {'score': 0.9923552870750427, 'part': 'rightShoulder', 'position': {'x': 447, 'y': 152}}, {'score': 0.8989572525024414, 'part': 'leftElbow', 'position': {'x': 449, 'y': 173}}, {'score': 0.997251033782959, 'part': 'rightElbow', 'position': {'x': 438, 'y': 173}}, {'score': 0.9670958518981934, 'part': 'leftWrist', 'position': {'x': 450, 'y': 187}}, {'score': 0.9952622652053833, 'part': 'rightWrist', 'position': {'x': 438, 'y': 191}}, {'score': 0.9634613990783691, 'part': 'leftHip', 'position': {'x': 441, 'y': 181}}, {'score': 0.9816059470176697, 'part': 'rightHip', 'position': {'x': 438, 'y': 182}}, {'score': 0.9952489137649536, 'part': 'leftKnee', 'position': {'x': 441, 'y': 203}}, {'score': 0.9948863387107849, 'part': 'rightKnee', 'position': {'x': 446, 'y': 204}}, {'score': 0.8975281119346619, 'part': 'leftAnkle', 'position': {'x': 435, 'y': 223}}, {'score': 0.9911296367645264, 'part': 'rightAnkle', 'position': {'x': 444, 'y': 230}}]}, {'score': 0.6807764400454128, 'keypoints': [{'score': 0.09946867823600769, 'part': 'nose', 'position': {'x': 73, 'y': 282}}, {'score': 0.08977577090263367, 'part': 'leftEye', 'position': {'x': 69, 'y': 280}}, {'score': 0.04911220073699951, 'part': 'rightEye', 'position': {'x': 73, 'y': 280}}, {'score': 0.8674992322921753, 'part': 'leftEar', 'position': {'x': 64, 'y': 280}}, {'score': 0.48962777853012085, 'part': 'rightEar', 'position': {'x': 77, 'y': 280}}, {'score': 0.9920362234115601, 'part': 'leftShoulder', 'position': {'x': 54, 'y': 288}}, {'score': 0.9908226728439331, 'part': 'rightShoulder', 'position': {'x': 76, 'y': 296}}, {'score': 0.18416059017181396, 'part': 'leftElbow', 'position': {'x': 48, 'y': 303}}, {'score': 0.9947155714035034, 'part': 'rightElbow', 'position': {'x': 78, 'y': 317}}, {'score': 0.043832093477249146, 'part': 'leftWrist', 'position': {'x': 55, 'y': 304}}, {'score': 0.9960497617721558, 'part': 'rightWrist', 'position': {'x': 90, 'y': 333}}, {'score': 0.984390377998352, 'part': 'leftHip', 'position': {'x': 50, 'y': 326}}, {'score': 0.9901418685913086, 'part': 'rightHip', 'position': {'x': 65, 'y': 327}}, {'score': 0.9429605603218079, 'part': 'leftKnee', 'position': {'x': 60, 'y': 361}}, {'score': 0.9988808631896973, 'part': 'rightKnee', 'position': {'x': 64, 'y': 361}}, {'score': 0.9668096303939819, 'part': 'leftAnkle', 'position': {'x': 56, 'y': 391}}, {'score': 0.8929156064987183, 'part': 'rightAnkle', 'position': {'x': 55, 'y': 391}}]}, {'score': 0.8333122537416571, 'keypoints': [{'score': 0.9808187484741211, 'part': 'nose', 'position': {'x': 219, 'y': 192}}, {'score': 0.8793118000030518, 'part': 'leftEye', 'position': {'x': 221, 'y': 189}}, {'score': 0.9634954333305359, 'part': 'rightEye', 'position': {'x': 216, 'y': 189}}, {'score': 0.4793465733528137, 'part': 'leftEar', 'position': {'x': 223, 'y': 191}}, {'score': 0.899986743927002, 'part': 'rightEar', 'position': {'x': 211, 'y': 190}}, {'score': 0.997833251953125, 'part': 'leftShoulder', 'position': {'x': 225, 'y': 203}}, {'score': 0.9941227436065674, 'part': 'rightShoulder', 'position': {'x': 204, 'y': 205}}, {'score': 0.9704650640487671, 'part': 'leftElbow', 'position': {'x': 235, 'y': 220}}, {'score': 0.3094051480293274, 'part': 'rightElbow', 'position': {'x': 203, 'y': 225}}, {'score': 0.8092747330665588, 'part': 'leftWrist', 'position': {'x': 224, 'y': 231}}, {'score': 0.4567853510379791, 'part': 'rightWrist', 'position': {'x': 199, 'y': 236}}, {'score': 0.9939448833465576, 'part': 'leftHip', 'position': {'x': 211, 'y': 239}}, {'score': 0.9704053401947021, 'part': 'rightHip', 'position': {'x': 197, 'y': 238}}, {'score': 0.984355092048645, 'part': 'leftKnee', 'position': {'x': 209, 'y': 263}}, {'score': 0.7979574799537659, 'part': 'rightKnee', 'position': {'x': 202, 'y': 263}}, {'score': 0.8682509064674377, 'part': 'leftAnkle', 'position': {'x': 201, 'y': 283}}, {'score': 0.8105490207672119, 'part': 'rightAnkle', 'position': {'x': 200, 'y': 284}}]}, {'score': 0.16497299075126648, 'keypoints': [{'score': 0.008961617946624756, 'part': 'nose', 'position': {'x': 755, 'y': 148}}, {'score': 0.015253216028213501, 'part': 'leftEye', 'position': {'x': 755, 'y': 145}}, {'score': 0.018510431051254272, 'part': 'rightEye', 'position': {'x': 753, 'y': 146}}, {'score': 0.005624234676361084, 'part': 'leftEar', 'position': {'x': 756, 'y': 146}}, {'score': 0.006182491779327393, 'part': 'rightEar', 'position': {'x': 751, 'y': 146}}, {'score': 0.06223544478416443, 'part': 'leftShoulder', 'position': {'x': 756, 'y': 162}}, {'score': 0.0909428596496582, 'part': 'rightShoulder', 'position': {'x': 756, 'y': 162}}, {'score': 0.05070573091506958, 'part': 'leftElbow', 'position': {'x': 765, 'y': 173}}, {'score': 0.02157267928123474, 'part': 'rightElbow', 'position': {'x': 758, 'y': 173}}, {'score': 0.05463290214538574, 'part': 'leftWrist', 'position': {'x': 761, 'y': 196}}, {'score': 0.03564557433128357, 'part': 'rightWrist', 'position': {'x': 756, 'y': 194}}, {'score': 0.37425994873046875, 'part': 'leftHip', 'position': {'x': 755, 'y': 203}}, {'score': 0.3569902777671814, 'part': 'rightHip', 'position': {'x': 749, 'y': 203}}, {'score': 0.5903506875038147, 'part': 'leftKnee', 'position': {'x': 755, 'y': 226}}, {'score': 0.4620838165283203, 'part': 'rightKnee', 'position': {'x': 740, 'y': 227}}, {'score': 0.2564328908920288, 'part': 'leftAnkle', 'position': {'x': 752, 'y': 257}}, {'score': 0.3941560387611389, 'part': 'rightAnkle', 'position': {'x': 739, 'y': 255}}]}]}


# In[16]:


b = [[522, 200, 590, 355], [660, 240, 724, 421], [286, 295, 354, 416], [103, 196, 175, 367], [422, 121, 473, 246], [43, 256, 112, 410], [185, 169, 251, 317], [717, 102, 784, 294]]


# In[17]:


a = ['dribble', 'dribble', 'no_action', 'run', 'walk', 'walk', 'no_action', 'pick']


# #Test

# In[18]:


TESTARGS = {
            'debugger' : False,
            'testImage1' : './dataandmodles/data/pz3Pointer.png',
            'testResultPath' : './dataandmodles/data/weightedDistanceAndTeamDetection'
           }
import DrawPose


# In[19]:


def load_image():
    img = cv2.imread(TESTARGS['testImage1'])
    return img, img.shape[0] , img.shape[1]

def write_image(img,fileName):
    cv2.imwrite(f'{TESTARGS["testResultPath"]}/{fileName}.png' , img)


# In[65]:


def visualDebugger():
    '''
    This function finds the number of VISION_LINES each bboxs is intersected by 
    Parameter
    ----------
    bboxs : A list with bbox. 
            Example: [[left,top,right,botoom],.......[left,top,right,botoom],.. ] 
            
    Return
    ---------
    counter : A list with the number of VISION_LINES each bboxs is intersected by
              Example: [8,....]
    '''
    global VISION_LINE , b,p,a
    counter = []
    bboxs = b
    posenetPred = p
    actions = a
    img,h,w = load_image()
    set_vision_lines(p)
    for index,bbox in enumerate(bboxs):
        #Draw the vision line
        if VISION_LINE[index] != False:
            draw_line(img,VISION_LINE[index] ,(255,0,0))
            VISION_LINE_COPY = VISION_LINE[:]
            VISION_LINE_COPY[index] = False
            count = check_bbox_intersection(bbox, VISION_LINE_COPY)
            counter.append(count)
        else:
            counter.append(-1)
    #Drawing ball
    sortedMaxsIndex = sortGazeModulePred(actions,counter)
    ballPosition = getBallLocation(posenetPred['detectionList'][sortedMaxsIndex[0]])
    DrawPose.drawBall(img,w,h,(735,149),(0,0,255),20)
    
    #Draw the bboxs 
    argmax = sortedMaxsIndex[0]
    for index,bbox in enumerate(bboxs):
        if index != 7:
            color = (255,255,0)
        else:
            color = (255,0,255)
        print(color)
        DrawPose.drawPred(img, "Lines",counter[index],bbox[0],bbox[1], bbox[2], bbox[3],color,yolo=True,roi=None,mask=None)
    img = DrawPose.draw_skel_and_kp(p,img,(255,0,255))            
    
    write_image(img,'GazeModuleTest3ConeAndVisionLine_Bbox')


# In[ ]:





# In[66]:


#visualDebugger()


# In[22]:



#bp, i = fitGazeModule(p,b,a)


# In[23]:


#bp


# In[24]:


#len(p['detectionList'])


# In[ ]:




