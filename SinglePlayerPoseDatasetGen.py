#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
import pickle
import json
import os
from pathlib import Path


# #Data Generation

# In[2]:


def load_data():
    with open('/Users/sandeep/Downloads/dataset/annotation_dict.json') as f:
      data = json.load(f)
    labels = {0 : "block", 1 : "pass", 2 : "run", 
              3: "dribble",4: "shoot",5 : "ball in hand", 
              6 : "defense", 7: "pick" , 8 : "no_action" ,
              9: "walk" ,10: "discard"}
    return data , labels
    


# In[3]:


raw =  {10:'left_ankel' , 13:'right_ankel' , 9:'left_knee',12:'right_knee',8:'left_hip',
        11:'right_hip', 4:'left_wrist',7:'right_wrist', 3:'left_elbow',6:'right_elbow',
        2:'left_shoulder',5:'right_shoulder',1:'nose',17:'right_ear',15:'right_eye',
        14:'left_eye' , 16:'left_ear', 0:'unknown'}
sorted_keys = np.sort((np.array(list(raw.keys()))))
PART_IDS = {raw[i]:i for i in sorted_keys}

CONNECTED_PART_NAMES = [
    ("left_hip", "left_shoulder"), ("left_elbow", "left_shoulder"),
    ("left_elbow", "left_wrist"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankel"), ("right_hip", "right_shoulder"),
    ("right_elbow", "right_shoulder"), ("right_elbow", "right_wrist"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankel"),
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")
]
CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES];
KEY_COLORS = {
  'unknown':(255,255,255),
  'nose': (255,255,0),
  'left_eye': (255,0,0),
  'right_eye': (0,0,255),
  'left_ear': (0,0,0),
  'right_ear': (0,255,0),
  'left_shoulder': (159, 95, 95),
  'right_shoulder': (227, 164, 43),
  'left_elbow': (35, 175, 164),
  'right_elbow': (226, 41, 226),
  'left_wrist': (205, 205, 0),
  'right_wrist': (255, 165, 0),
  'left_hip': (177, 152, 112),
  'right_hip': (128, 0, 0),
  'left_knee': (0, 128, 128),
  'right_knee': (255, 20, 147),
  'left_ankel': (218, 112, 214),
  'right_ankel': (0, 0, 128)
}
KEY_COLORS_INDICES_MAP = {PART_IDS[key]:value for key,value in KEY_COLORS.items()}
SEGMENTCOLORS = {
  'left_elbow|left_shoulder': (226, 115, 78), 
  'left_elbow|left_wrist': (191, 255, 0), 
  'left_hip|left_knee': (128, 128, 0), 
  'left_knee|left_ankel': (136, 45, 23),
  'right_hip|right_shoulder': (255, 0, 255), 
  'right_elbow|right_shoulder': (106, 90, 205),
  'right_elbow|right_wrist': (255, 99, 71),
  'right_hip|right_knee': (255, 255, 0),
  'left_hip|left_shoulder': (30, 144, 255),
  'right_knee|right_ankel': (60, 179, 113),
  'left_shoulder|right_shoulder': (139, 0, 139),
  'left_hip|right_hip': (255, 20, 147)
};
#Make a dictionary that maps from pair to pair color (1,2) -> (color)
SEGMENTCOLORS_INDICES_MAP = ({(PART_IDS[part1], PART_IDS[part2]): SEGMENTCOLORS[f'{part1}|{part2}']
                             for part1,part2 in CONNECTED_PART_NAMES})

ARGS = {'datasetGen': False}


# In[4]:


def get_adjacent_keypoints(keypoints):
    '''
    Helper function of draw_skel_and _kp
    Returns 2 coord of 2 points where line needs to be drawn
    EXAMPLE: [[[X1,Y1],[X2,Y2]], [pair],........]
    '''
    results = []
    for left, right in CONNECTED_PART_INDICES:
        #we wont have all the points. 
        #If point is not visilbe not there.Thus,we will get key errer
        try:
           results.append({(left,right): np.array([
                                [ keypoints[left][0] , keypoints[left][1] ],
                                [ keypoints[right][0] , keypoints[right][1] ]
                                    ]
                           ).astype(np.int32)
               
                         }
                          
                         )
        except KeyError:
            continue
            
    return results


# In[5]:


def rgb_to_brg(rgb):
    '''
    Helper function of draw_skel_and_kp
    Returns brg values from rgb values 
    '''
    bgr = (rgb[2] , rgb[1] , rgb[0])
    return bgr


# In[6]:


def getScaleFactors(imgW, imgH, originalImgH , originalImgW):
   '''
      Helper function for draw_skel_and_kp
      Return
      ------
      : A tuple with given scalefactor (wdith,height). 
   '''
   return (imgW/originalImgW, imgH/originalImgH);


# In[7]:


def mapToOrginialImage(coord,scaleFactors):
    '''
      Helper function for draw_skel_and_kp
      Returns 
      -------
      : A coord (width, height)
    '''
    coordInOriginalImage = [int(coord[0] * scaleFactors[0]) , int(coord[1] * scaleFactors[1])]
    return tuple(coordInOriginalImage)


# In[8]:


def draw_skel_and_kp(poses , img,scale=False):
    '''
    Parameter
    ---------
    pose : A 1D list contianing dictionary of poses. They keys are relative to the "raw" dictionary 
    Example [{0:(x,y)....17:(x,y)}, ....]
    
    image : Image to draw the posses into. 
    
    
    scale : If True, coordinates will be mapped to a new image of size (512,512) and drawn in it.
            Flase, coordinate will be drawn the passed image. 
            In False case (Make sure the coords  posess are already relative to the passed image)
    
    Returns
    -----------
    out_img : The orignial image with poses drawn. If scale is False
              A black image of size (512x512) with poses drawn. If scale is True 
    '''
    if scale:
        out_img = np.zeros((512,512,3))
        scale = getScaleFactors(512,512,img.shape[0],img.shape[1])
    else:
        out_img = np.zeros(img.shape)
        scale = (1,1) 
    adjacent_keypoints = []
    cv_keypoints = []
    #For every pose of the player
    for pose in poses:
        keypoints = pose
        new_keypoint = get_adjacent_keypoints(keypoints)
        adjacent_keypoints.extend(new_keypoint)
        
        #Draw Lines
        for adjacent_keypoint in adjacent_keypoints:
            pair_key , coords = list(adjacent_keypoint.keys())[0] , list(adjacent_keypoint.values())[0]
            coords[0],coords[1] = mapToOrginialImage(coords[0],scale) , mapToOrginialImage(coords[1],scale) 
            color = rgb_to_brg(SEGMENTCOLORS_INDICES_MAP[pair_key])
            cv2.line(out_img, tuple(coords[0]), tuple(coords[1]),
                     color, 
                     8)
        
        #Draw Points 
        for key,keypoint in keypoints.items():
            center = (round(keypoint[0]) ,round(keypoint[1]))
            center = mapToOrginialImage(center,scale)
            color = rgb_to_brg(KEY_COLORS_INDICES_MAP[key])
            cv2.circle(out_img, center,2, color,
                       thickness=3, lineType=8, shift=0)
             
    return out_img


# In[9]:


def get_label(fileName):
    label = annotations[fileName]
    return labels[label]


# In[14]:


# ARGS['datasetGen'] = True


# In[16]:


if ARGS['datasetGen']:
    annotations , labels = load_data()
    fileName = os.listdir('/Users/sandeep/Downloads/dataset/examples')
    fileName = [name for name in fileName if name[-1] != 'y']
    for name in fileName:
        videoPath = f'/Users/sandeep/Downloads/dataset/examples/{name}'
        posePath = f'/Users/sandeep/Downloads/dataset/examples/{name[0:-4]}.npy'
        try:
            poses_dict,label = np.load(posePath,allow_pickle=True) , get_label(name[0:-4])

            cap = cv2.VideoCapture(videoPath)
            count = 0
            while cap.isOpened():
                ret,frame = cap.read()
                if ret:
                    #do stuff
                    
                    if count == 12:
                        poses = poses_dict[count]
                        frame = draw_skel_and_kp([poses],frame,[255,0,0])
                        f = '/Volumes/My Passport/FinalYearProjectData/ActionClassification5/test'
                        cv2.imwrite(f"{f}/{label}/{name[0:-4]}_{count}.jpg" , frame)

            #       End of do stuff

                    #cv2.imshow('test', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break  
                else:
                    cap.release()
                    break
                count += 1
            cap.release()
            cv2.destroyAllWindows()
        except KeyError:
            continue


# #Test

# In[ ]:




