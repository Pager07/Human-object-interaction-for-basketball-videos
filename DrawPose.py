#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2 
import numpy as np


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


# In[15]:


def draw_skel_and_kp(poses , img, color,classes=None):
    '''
    This function draws the keypoints and skeleton
    
    Parameter
    -------------
    poses : That raw output from posenet
    
    img : The image that was sent to posenet for pose prediction
    
    Return
    -------------
    out_img : Image of same size as img. It has the skeleton and keypoints drawn in it
    
    '''
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    #For every pose of the player
    for i,pose in enumerate(poses['detectionList']):
        keypoints = pose['keypoints']
        new_keypoint = get_adjacent_keypoints(keypoints)
        adjacent_keypoints.extend(new_keypoint)
        
        for keypoint in keypoints:
            x,y,score = round(keypoint['position']['x']) ,round(keypoint['position']['y']),keypoint['score']
            cv_keypoints.append(cv2.KeyPoint(x,y , 10. * score))
        if classes != None:
            cv2.putText(out_img, classes[i][0], (x,y), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    
    out_img = cv2.drawKeypoints(
        img, cv_keypoints , outImage=np.array([]), color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
    out_img = cv2.polylines(out_img , adjacent_keypoints , isClosed=False, color=list(color))
    return out_img


# In[2]:


def draw_skelton(poses,img):
    '''
    This function draws the skeleton

    Parameter
    -------------
    poses : That raw output from posenet

    img : The image that was sent to posenet for pose prediction

    Return
    -------------
    out_img : Image of same size as img. It has the skeleton keypoints drawn in it
    
    '''
    adjacent_keypoints = []
    out_img = img
    for pose in poses['detectionList']:
        keypoints  = pose['keypoints']
        new_keypoints = get_adjacent_keypoints(keypoints)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img , adjacent_keypoints, isClosed=False, color=(255,255,0))
    return out_img   


# In[1]:


def draw_keypoints(poses,img):
    '''
    This function draws the keypoints
    
    Parameter
    -------------
    poses : That raw output from posenet
    
    img : The image that was sent to posenet for pose prediction
    
    Return
    -------------
    out_img : Image of same size as img. It has the keypoints drawn in it
    '''
    cv_keypoints= []
    for pose in poses['detectionList']:
        keypoints = pose['keypoints']
        for keypoint in keypoints:
            x,y,score = round(keypoint['position']['x']) ,round(keypoint['position']['y']),keypoint['score']
            cv_keypoints.append(cv2.KeyPoint(x,y , 10. *score))
    out_img = cv2.drawKeypoints(img, cv_keypoints , outImage=np.array([]) ) 


# In[11]:


def putLabel(image,left,top,label,colour,customLabelSize=False):
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    if customLabelSize:
        labelSize = customLabelSize #(116,12)
        
    top = max(top, labelSize[1])
    pt1 = (left, top - round(1.5*labelSize[1])) 
    pt2 = (left + round(1.5*labelSize[0]), top + baseLine)
    cv2.rectangle(image,pt1 , pt2 , colour, cv2.FILLED)
    
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    

def drawPred(image,actionClass, confidence, left, top, right, bottom, colour,yolo=False,roi=None,mask=None):
    
    if yolo == False:
        blended = ((0.4 * np.array(colour)) + (0.6 * roi)).astype("uint8")
        image[top:bottom, left:right][mask] = blended
        
    
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 1)

    # construct label
    label = '%s:%.2f' % (actionClass, confidence)
    
    putLabel(image,left,top,label,colour,customLabelSize=(116,12))
    
    


# In[9]:


def drawBall(img,imgW,imgH,center,color,bboxSize):
    '''
    Draw a cricle
    
    Parameter 
    ---------
    img : Image to draw the cricle on
    center : (x,y) or (width,height) or (col,row)
    '''
    n = bboxSize
    left_top = (max(center[0]-n,0) ,max(center[1]-n,0))
    right_bottom = (min(center[0]+n,imgW) ,min(center[1]+n,imgH))
    cv2.rectangle(img,left_top,right_bottom,color,1)
    label = 'Ball'
    putLabel(img,left_top[0], left_top[1], label,color)
    
    
    

