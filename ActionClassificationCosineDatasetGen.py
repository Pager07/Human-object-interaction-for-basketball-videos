#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2 
import numpy as np 
import pandas as pd 
import os
from sklearn import preprocessing 


# In[183]:


raw =  {10:'left_ankel' , 13:'right_ankel' , 9:'left_knee',12:'right_knee',8:'left_hip',
        11:'right_hip', 4:'left_wrist',7:'right_wrist', 3:'left_elbow',6:'right_elbow',
        2:'left_shoulder',5:'right_shoulder',1:'nose',17:'right_ear',15:'right_eye',
        14:'left_eye' , 16:'left_ear', 0:'unknown'}
df_cols_to_keys = None 
sorted_keys = np.sort((np.array(list(raw.keys()))))
PART_IDS = {raw[i]:i for i in sorted_keys}

CONNECTED_PART_NAMES = [
    ("left_hip", "right_hip"), ("left_elbow", "left_shoulder"),
    ("left_elbow", "left_wrist"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankel"), ("right_hip", "right_shoulder"),
    ("right_elbow", "right_shoulder"), ("right_elbow", "right_wrist"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankel"),
    ("left_shoulder", "right_shoulder"), ("left_hip", "left_shoulder")
]
CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES];

UNCONNECTED_PART_NAMES = ["nose", "right_ear" , "left_ear", "right_eye" , "left_eye",  "unknown"]
UNCONNECTED_PART_INDICES = [PART_IDS[a] for a in UNCONNECTED_PART_NAMES];

ARGS = {'normDatasetPath':'./dataandmodles/data/singlePlayersPosesL2Normalized.csv',
        'datasetPath':'./dataandmodles/data/singlePlayersPoses.csv',
        'customDatasetPath' :  './dataandmodles/data',
        'datasetGen':False,
        'customDatasetGen': False,
        'customDatasetFrameSelectionList': [11],
        'inputImgW': 0,
        'imputImgH':0}


# In[184]:


#get_custom_dataset(norm=False)


# #Handling Main Dataset

# In[111]:


def load_data():
    with open('/Users/sandeep/Downloads/dataset/annotation_dict.json') as f:
      data = json.load(f)
    labels = {0 : "block", 1 : "pass", 2 : "run", 
              3: "dribble",4: "shoot",5 : "ball in hand", 
              6 : "defense", 7: "pick" , 8 : "no_action" ,
              9: "walk" ,10: "discard"}
    return data , labels


# In[112]:


if ARGS['datasetGen']:
    annotations , labels = load_data()
def get_label(fileName):
    label = annotations[fileName]
    return labels[label]


# In[1]:


def getScaleFactors(imgW, imgH, originalImgH , originalImgW):
   '''
      Helper function for get_data()
      Returns a tensor with given scalefactor (wdith,height). 
   '''
   scale = (imgW/originalImgW,imgH/originalImgH);
   return scale


# In[4]:


def mapToSquareImage(coord,scaleFactors):
    '''
      Helper function for get_data()
      Parameter
      ---------
      coord : a tuple  cotaining the coordinates 
      
      scaleFactors : a tuple containing scaleFactors 
      
      Return
      -------
      coord : A tuple of coordinate in integer scaled by their scaleFactors respectively
    '''
    coordInOriginalImage = [int(coord[0] * scaleFactors[0]) , int(coord[1] * scaleFactors[1])]
    return tuple(coordInOriginalImage)


# In[177]:


def get_extended_pose(poses_dict, label=None):
    '''
    Helper function for get_data() and get_posenet_extended_pose()
    
    Parameters
    ----------
    pose : A dictionary containing 1 pose. The keys refer to bodyparts relative to the raw dict.
    Example {0:(x,y), ....} 
    
    label : get_data() will send label arugment but get_posenet_extended_pose() will not
    Return
    ---------
    extended_pose : A dictionary containing x and y coord has it own key
    Example {0_x: 1 ,0_y: 7 ....}  
    '''
    extended_pose = {}
    scale = getScaleFactors(244,244,ARGS['inputImgH'],ARGS['inputImgW'])
    for key,value in poses_dict.items():
        #TO DO: scale the value before setting to the value
        value = mapToSquareImage(value,scale)
        extended_pose[f'{key}_x'] = value[0]
        extended_pose[f'{key}_y'] = value[1]
    if label != None:
        extended_pose['label'] = label 
    return extended_pose
        


# In[116]:


def filterPoseList(poseList , meidan=True):
    '''
    Helper function of get_data()
    
    '''
    if median:
        return np.array([poseList[7]])
    else:
        return poseList


# In[173]:


def get_data():
    '''
    Helper function for generate_dataset()
    
    Returns
    -------
    data : A 1D list contining dict of 16 poses gotton from 16 frames for a single player
    Example [{0_x:1 , 0_y:2 , 1_x:67........,'label':'walk'] 
    The coordinates of the poses coords for players in sqaure image of size (244, 244)
    '''
    fileName = os.listdir('/Users/sandeep/Downloads/dataset/examples')
    fileName = [name for name in fileName if name[-1] != '4']
    data = []
    for name in fileName:
        posePath = f'/Users/sandeep/Downloads/dataset/examples/{name}'
        try:
            poses_list,label = np.load(posePath,allow_pickle=True) , get_label(name[0:-4])
            for poses_dict in poses_list:
                extended_pose = get_extended_pose(pose_dict,label=label)
                data.append(extended_pose)
        except KeyError:
            continue
    return data   


# In[118]:


def get_keys(df_keys):
    '''
    Helper function for get_proxy_coord
    Returns keys that the df columns name refers too. eg. 0_x -> 0 
    '''
    global df_cols_to_keys 
    return [df_cols_to_keys[key] for key in df_keys]
    


# In[119]:


def get_connected_keys(key):
    key_pair_tuple = [t for t in CONNECTED_PART_INDICES if key in t]
    return [key1 if key1 != key else key2 for (key1,key2) in key_pair_tuple ]


# In[120]:


def filter_keys(list1 , list2):
    '''
    Helper function for get_proxy_coord
    Returns keys that are not in the list2
    '''
    return list(set(list1)-set(list2))


# In[121]:


def map_keys_to_df_key(key):
    return f'{key}_x' , f'{key}_y'


# In[122]:


def get_unconneted_proxy_coord(keys):
    '''
    Helper function for get_proxy_coord
    '''
    usable_unconnected_keys = filter_keys(UNCONNECTED_PART_INDICES , keys)
    if len(usable_unconnected_keys) != 0:
        key_x , key_y  = map_keys_to_df_key(usable_unconnected_keys[0])
        return key_x , key_y
    else:
        return get_connected_proxy_coord(2,keys)#trace starts from right shoulder


# In[123]:


def get_connected_proxy_coord(key,keys):
    '''
    Helper function for get_proxy_coord
    '''
    connected_keys_with_key = get_connected_keys(key)
    usable_connected_keys = filter_keys(connected_keys_with_key,keys)
    if  len(usable_connected_keys) != 0:#base-case, we know there is no pose with 1 points
        key_x , key_y = map_keys_to_df_key(usable_connected_keys[0])
        return key_x,key_y
    else:
       key_x, key_y = get_connected_proxy_coord(connected_keys_with_key[0],keys)
       return key_x,  key_y


# In[124]:


def get_proxy_coord(row):
    '''
    Helper function to replace the NaN values/ the poses that are not visible using poxy poses
    PARAMS key: the key that gets passed is eg.1_x
    '''
    global df_cols_to_keys 
    #df keys with nan
    df_keys = [key for key in row[row.isnull()].index.tolist() if key[-1] != 'y'] 
    keys = get_keys(df_keys)
    for key in keys:
        if key in UNCONNECTED_PART_INDICES:#if the nan key is a  Unconnected part 
            key_x , key_y = get_unconneted_proxy_coord(keys)
        else:#if the nan key is a  Connected part 
            key_x, key_y = get_connected_proxy_coord(key,keys)
                
        current_key_x , current_key_y = map_keys_to_df_key(key)
        row[current_key_x] , row[current_key_y] = row[key_x] , row[key_y]
    return row


# In[125]:


def get_df_cols():
    '''
    Helper function for generate_dataset()
    '''
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    a = []
    for col in cols:
        xCol = f'{col}_x'
        yCol = f'{col}_y'   
        a.append(xCol)
        a.append(yCol)
    a.append('label')
    b = {}
    for col in cols:
        b[f'{col}_x'] = col
        b[f'{col}_y'] = col

    return a,b


# In[126]:


def fillNaN(df):
    '''
    Helper function for generate_dataset()
    Updates the NaN cells with proxy coords. 
    '''
    return  df.apply(get_proxy_coord, axis=1)


# In[127]:


def generate_dataset(data=None, norm=True):
    '''
    1.This function will fill the missing bodypoints with proxy coords and
      generate l2 normalized dataset from raw data
    
    2.Helper function for generate_data_posenet(). If norm = Flase is passed then l2 normalization
      is not applied.
    
    Parameters
    ----------
    data : A list with extended_poses.
    If data is not passed then you are aiming to generate the main pose dataset 
    Example : [{0_x:1, 0_y:1 ,17_y:16, 17_y:17} , {0_x:1, 0_y:1 ,16_y:10, 16_y:10} , {..}]
    
    Return
    -------
    normalized_data : A 2D array with normalized data 
    [[0_x, 0_y, 1_x, 1_y, .......17_y] , [......]]
    '''
    global df_cols_to_keys
    if ARGS['datasetGen']:
        data = get_data()
        df = pd.DataFrame(data)
    else:
      df = pd.DataFrame(data)
      df['0_x'] , df['0_y'] = np.nan , np.nan
        
    df_cols,df_cols_to_keysx = get_df_cols()
    df_cols_to_keys = df_cols_to_keysx
    df = df[df_cols[:-1]] #Generates nan values
    df = fillNaN(df)
    if norm:
        normalized_data = preprocessing.normalize(df, axis=1)
        return normalized_data
    else:
        unnormalized_data = df.to_numpy()
        return unnormalized_data
    


# In[128]:


def read_dataset(name=None,norm=True):
    if name != None:
        return pd.read_csv(f"{ARGS['customDatasetPath']}/{name}.csv")
    elif norm:
        return pd.read_csv(ARGS['normDatasetPath'])
    elif norm == False:
        return pd.read_csv(ARGS['datasetPath'])


# In[129]:


def get_custom_dataset(norm=True):
    '''
    Save custom dataset. You can select the frames you want in the ARGS. 
    The ARGS['customDatasetFrameSelectionList'] should refer to frame out of the 16 frames.
    Example name of the file:
        - custom_normalized_framelist_[7, 8, 9]_total_169530.csv
    '''
    df = read_dataset(norm=norm)
    condition = True
    for i,f in enumerate(ARGS['customDatasetFrameSelectionList']):
        if i == 0:
            if f == 8:
                condition =(((df.index+1)%f == 0) & ((df.index+1)%16 != 0))
            else:
                condition =  ((df.index+1)%f == 0)
        else:
             if f == 8:
                condition =condition | (((df.index+1)%f == 0) & ((df.index+1)%16 != 0))
             else:
                condition = condition | ((df.index+1)%f == 0)
    df = df[condition]
    fname = f'custom_normalized_{norm}_framelist_{ARGS["customDatasetFrameSelectionList"]}_total_{df.shape[0]}'
    df.to_csv(f'{ARGS["customDatasetPath"]}/{fname}.csv',index=False)


# #Handeling Posenet Data

# In[130]:


def posenet_part_to_part(part):
    '''
    Helper function of get_posenet_extended_pose
    '''
    if 'Ankle' in part:
        part = part.replace('Ankle' , 'ankel')
    if 'right' in part: 
        return f'right_{part[5:].lower()}'
    elif 'left' in part:
        return f'left_{part[4:].lower()}'
    return part
           


# In[131]:


def map_part_to_key(posenetPart):
    '''
    Helper function of get_posenet_extended_pose()
    '''
    for key, part in raw.items():
        if part == posenetPart:
            return key
    raise Exception(f'{posenetPart} not in raw')


# In[132]:


def get_posenet_extended_pose(pose):
    '''
    Helper function for generate generate_dataset()
    
    Parameters
    ----------
    pose : A list containing dictionary containing 1 pose. The keys refer to bodyparts relative to the posenet pred.
    Example [{'score': 0.8984865546226501, 'part': 'nose', 'position': {'x': 741.3767104497117, 'y': 201.7590852932459}}..].
    
    Return
    ---------
    extended_pose : A dictionary containing x and y coord has it own key
    Example {0_x: 1 ,0_y: 7 ....}  
    '''
    #Change keys
    pose_dict = {}
    for part_dict in pose:
        part = posenet_part_to_part(part_dict['part']) 
        key = map_part_to_key(part)
        pose_dict[key] = (part_dict['position']['x'] , part_dict['position']['y'])
    extended_pose = get_extended_pose(pose_dict)
    return extended_pose


# In[133]:


def add_unkown_keypoint(pose):
    '''
    NOT USED ANYMORE! THIS FUNCTION DOES NOT RESPECT THE DATA PIPELINE
    Helper function of generate_data_posenet()
    This function appends a new  bodypart "unknown" object to the human keypoints list.
    This is important because all the poses in posenet pred will not contain thi body part, however,
    our original dataset contained this part.
    
    Parameter
    ---------
    pose : A 1D list with a human body parts
    Example [{score: , part:.. ,position:..}, secondBodyPart..a..]
    
    Return
    ----------
    pospose : A 1D list with a human body parts with 'unknown' body object 
    Example [{score: , part:'uknow' ,position:..} , {score: , part:.. ,position:..}, secondBodyPart..a..]
    '''
    
    unknow_obj = {'score': 0.01,
                 'part': 'unknown',
                 'position': {'x': np.nan, 'y': np.nan}}
    pose.append(unknow_obj)
    return pose


# In[150]:


def get_extended_confidences(pose):
    '''
    Helper function of generate_data_posenet()
    
    Parameter
    --------
    pose : A list containing dictionary containing 1 pose. The keys refer to bodyparts relative to the posenet pred.
    Example [{'score': 0.8984865546226501, 'part': 'nose', 'position': {'x': 741.3767104497117, 'y': 201.7590852932459}}..].
    
    Return 
    ----------
    theta : The weigth for each coordinate
    [w_0_x , w_0_y........,w_17_y]
    '''
    extended_confidence  = [0.01, 0.01] 
    for part_dict in pose:
        extended_confidence.append(part_dict['score']) #w_key_x
        extended_confidence.append(part_dict['score']) #w_key_y
    return extended_confidence


# In[2]:


def generate_data_posenet(posenetPred,norm=True):
    '''
    This functions generates l2 normalized data form non-empty(Must have atleast 1 humans)
    Posenet results
    
    Parameters
    ---------
    posenetPred : The non-empty preprocessed output of Posenet model. 
    Example {'detectionList': [{keypoints:[{score: , part:.. ,position:..}]} , human2 , ....]
    Preprocessed means the coords of the keypoints are realtive to the  image of size (244x244) that is cropped 
    to the detected huam 
    
    inputImgH : The height/rows/img.shape[0] in the input image
    inputImgW : The height/rows/img.shape[1] in the input image.
    They will be used to scale the coordinates in the posenet results to image size of (244,244)
    Because, the poses stored in the vantage tree are relative to (244,244)
    
    norm : True if you want a l2 normalized coordinates. False, if you want raw coordinate of poses 
            relative to image size (244,244)
    Return
    ------
    normalized_data : A 2D list with exte
    '''
    #need for finding the right scale factor
    #
    ARGS['inputImgH'] = 244 #img.shape[0]a.k.a rows
    ARGS['inputImgW']  = 244
    
    data = []
    weigthsParts = [] 
    for human in posenetPred['detectionList']:
        extended_pose = get_posenet_extended_pose(human['keypoints'])
        if norm == False:
            weigthsParts.append(get_extended_confidences(human['keypoints']))
        data.append(extended_pose)
    if norm:
        normalized_data = generate_dataset(data=data,norm=norm)
        return normalized_data
    else:
        unnormalized_data = generate_dataset(data=data,norm=norm)
        return unnormalized_data , np.array(weigthsParts)
    


# #Test 

# In[179]:


#d,w= generate_data_posenet(p,inputImgH=825, inputImgW=1461 , norm=False)


# In[ ]:


#d= generate_data_posenet(p,norm=True)


# In[13]:


#a = read_dataset(norm=True)


# In[14]:


#a.columns.tolist()


# In[34]:


#a.to_csv('./dataandmodles/data/singlePlayersPoses.csv', index=False)
#df_normalized_df = pd.DataFrame(df_normalized, columns=df.columns)


# In[22]:


#df_normalized_df


# In[39]:


#df_normalized_df.to_csv('./dataandmodles/data/singlePlayersPosesL2Normalized.csv', index=False)


# In[1]:


#x =np.load('/Users/sandeep/Downloads/dataset/examples/0000000_flipped.npy',allow_pickle=True) 


# In[3]:


#len(x)


# #Test data

# In[168]:


# p = {'detectionList': [{'score': 0.8020517387810875,
#    'keypoints': [{'score': 0.05392974615097046,
#      'part': 'nose',
#      'position': {'x': 163.23003359948206, 'y': 608.5078749565772}},
#     {'score': 0.02309584617614746,
#      'part': 'leftEye',
#      'position': {'x': 158.01330075281837, 'y': 603.9510282354503}},
#     {'score': 0.02208957076072693,
#      'part': 'rightEye',
#      'position': {'x': 165.339461655206, 'y': 603.2992517795265}},
#     {'score': 0.8850325345993042,
#      'part': 'leftEar',
#      'position': {'x': 148.16113053368272, 'y': 606.3292730585954}},
#     {'score': 0.8273975849151611,
#      'part': 'rightEar',
#      'position': {'x': 169.36971927939283, 'y': 606.1421976651434}},
#     {'score': 0.9995720386505127,
#      'part': 'leftShoulder',
#      'position': {'x': 136.96197301171694, 'y': 635.6809748454565}},
#     {'score': 0.9991105794906616,
#      'part': 'rightShoulder',
#      'position': {'x': 182.95736291792508, 'y': 634.3172801768139}},
#     {'score': 0.974422812461853,
#      'part': 'leftElbow',
#      'position': {'x': 128.66949637909508, 'y': 667.2687765977618}},
#     {'score': 0.996101975440979,
#      'part': 'rightElbow',
#      'position': {'x': 194.04934974556113, 'y': 663.888070537891}},
#     {'score': 0.9592186212539673,
#      'part': 'leftWrist',
#      'position': {'x': 123.35610578033362, 'y': 694.9166127780074}},
#     {'score': 0.9804818034172058,
#      'part': 'rightWrist',
#      'position': {'x': 186.84898668817814, 'y': 692.729613380994}},
#     {'score': 0.9917000532150269,
#      'part': 'leftHip',
#      'position': {'x': 140.39393137456773, 'y': 698.021448376687}},
#     {'score': 0.9993411302566528,
#      'part': 'rightHip',
#      'position': {'x': 172.00248377376727, 'y': 698.5536931780016}},
#     {'score': 0.9911080598831177,
#      'part': 'leftKnee',
#      'position': {'x': 133.2291883832953, 'y': 735.3801846347194}},
#     {'score': 0.9921324253082275,
#      'part': 'rightKnee',
#      'position': {'x': 172.5117406974571, 'y': 746.4394785287682}},
#     {'score': 0.9864200353622437,
#      'part': 'leftAnkle',
#      'position': {'x': 143.4178979959381, 'y': 772.2855890867409}},
#     {'score': 0.95372474193573,
#      'part': 'rightAnkle',
#      'position': {'x': 168.01400112480707, 'y': 790.7553921886291}}]},
#   {'score': 0.8869766435202431,
#    'keypoints': [{'score': 0.8423951864242554,
#      'part': 'nose',
#      'position': {'x': 221.9403120337354, 'y': 230.29354392759097}},
#     {'score': 0.8832824230194092,
#      'part': 'leftEye',
#      'position': {'x': 224.0734465023998, 'y': 224.80811612196976}},
#     {'score': 0.5725916624069214,
#      'part': 'rightEye',
#      'position': {'x': 221.08537565367052, 'y': 224.52157953169768}},
#     {'score': 0.9158638715744019,
#      'part': 'leftEar',
#      'position': {'x': 230.91303778007236, 'y': 226.76105242006906}},
#     {'score': 0.3342929780483246,
#      'part': 'rightEar',
#      'position': {'x': 224.24276945028413, 'y': 229.6178772395777}},
#     {'score': 0.986544132232666,
#      'part': 'leftShoulder',
#      'position': {'x': 237.28917203324565, 'y': 245.03261972176136}},
#     {'score': 0.995442271232605,
#      'part': 'rightShoulder',
#      'position': {'x': 225.79366391071218, 'y': 245.23665284201422}},
#     {'score': 0.968991219997406,
#      'part': 'leftElbow',
#      'position': {'x': 233.9817853497432, 'y': 259.16278851011594}},
#     {'score': 0.9035201668739319,
#      'part': 'rightElbow',
#      'position': {'x': 219.50578075223203, 'y': 263.8014277420176}},
#     {'score': 0.9322922825813293,
#      'part': 'leftWrist',
#      'position': {'x': 215.41493770513642, 'y': 245.94165377806956}},
#     {'score': 0.7772630453109741,
#      'part': 'rightWrist',
#      'position': {'x': 214.03382193461786, 'y': 266.8696041239403}},
#     {'score': 0.9994568228721619,
#      'part': 'leftHip',
#      'position': {'x': 236.14877241648986, 'y': 308.5849463456218}},
#     {'score': 0.9992390871047974,
#      'part': 'rightHip',
#      'position': {'x': 223.6569348203109, 'y': 309.22001482419785}},
#     {'score': 0.9966468214988708,
#      'part': 'leftKnee',
#      'position': {'x': 254.1238371388296, 'y': 363.23743641603755}},
#     {'score': 0.9974603652954102,
#      'part': 'rightKnee',
#      'position': {'x': 240.2136203385471, 'y': 364.3688832378883}},
#     {'score': 0.9928112030029297,
#      'part': 'leftAnkle',
#      'position': {'x': 253.1930084049925, 'y': 411.6310033012194}},
#     {'score': 0.9805094003677368,
#      'part': 'rightAnkle',
#      'position': {'x': 240.13132280892648, 'y': 415.4867563198096}}]},
#   {'score': 0.8388286373194527,
#    'keypoints': [{'score': 0.6557391285896301,
#      'part': 'nose',
#      'position': {'x': 598.8978732676989, 'y': 506.7789182943942}},
#     {'score': 0.6578531265258789,
#      'part': 'leftEye',
#      'position': {'x': 599.3679572509022, 'y': 502.0857118527232}},
#     {'score': 0.05247753858566284,
#      'part': 'rightEye',
#      'position': {'x': 602.1768501653207, 'y': 500.5353890771056}},
#     {'score': 0.9758005142211914,
#      'part': 'leftEar',
#      'position': {'x': 608.2421172602793, 'y': 493.849841617009}},
#     {'score': 0.7145469784736633,
#      'part': 'rightEar',
#      'position': {'x': 620.306624519691, 'y': 491.54931673516023}},
#     {'score': 0.9987959861755371,
#      'part': 'leftShoulder',
#      'position': {'x': 608.1783342504323, 'y': 520.942217609498}},
#     {'score': 0.9979566335678101,
#      'part': 'rightShoulder',
#      'position': {'x': 636.3336160415121, 'y': 512.8697870707388}},
#     {'score': 0.9972736835479736,
#      'part': 'leftElbow',
#      'position': {'x': 600.9189186810554, 'y': 562.9568884666086}},
#     {'score': 0.8372284770011902,
#      'part': 'rightElbow',
#      'position': {'x': 650.6548077801193, 'y': 546.6246073936052}},
#     {'score': 0.9979281425476074,
#      'part': 'leftWrist',
#      'position': {'x': 593.4696491595065, 'y': 591.3029705998803}},
#     {'score': 0.4135850667953491,
#      'part': 'rightWrist',
#      'position': {'x': 636.3809471121442, 'y': 570.0108607505388}},
#     {'score': 0.9993095397949219,
#      'part': 'leftHip',
#      'position': {'x': 628.4978959819351, 'y': 581.5615101521721}},
#     {'score': 0.9985918402671814,
#      'part': 'rightHip',
#      'position': {'x': 650.4556511243185, 'y': 577.9223464762524}},
#     {'score': 0.9821994304656982,
#      'part': 'leftKnee',
#      'position': {'x': 609.7089073131147, 'y': 620.8149327818608}},
#     {'score': 0.9994444847106934,
#      'part': 'rightKnee',
#      'position': {'x': 649.6841825045897, 'y': 625.2520755473621}},
#     {'score': 0.983491063117981,
#      'part': 'leftAnkle',
#      'position': {'x': 600.7801400159628, 'y': 666.844545389172}},
#     {'score': 0.9978652000427246,
#      'part': 'rightAnkle',
#      'position': {'x': 651.6086549651757, 'y': 677.071969976045}}]},
#   {'score': 0.9007229734869564,
#    'keypoints': [{'score': 0.9299418926239014,
#      'part': 'nose',
#      'position': {'x': 531.9455111794704, 'y': 333.43611599708964}},
#     {'score': 0.8541350364685059,
#      'part': 'leftEye',
#      'position': {'x': 532.8446314879571, 'y': 330.1159189214326}},
#     {'score': 0.7765036821365356,
#      'part': 'rightEye',
#      'position': {'x': 523.5307017328141, 'y': 327.3713186270649}},
#     {'score': 0.8629913330078125,
#      'part': 'leftEar',
#      'position': {'x': 562.8576965046286, 'y': 322.71127055877207}},
#     {'score': 0.7071982622146606,
#      'part': 'rightEar',
#      'position': {'x': 552.8912220108375, 'y': 331.64173144236383}},
#     {'score': 0.9797060489654541,
#      'part': 'leftShoulder',
#      'position': {'x': 549.9477634358495, 'y': 341.3249667645327}},
#     {'score': 0.9938467741012573,
#      'part': 'rightShoulder',
#      'position': {'x': 571.3982272594609, 'y': 343.6911496463227}},
#     {'score': 0.9776655435562134,
#      'part': 'leftElbow',
#      'position': {'x': 523.0600637273395, 'y': 344.2608908036759}},
#     {'score': 0.8535618782043457,
#      'part': 'rightElbow',
#      'position': {'x': 583.010996217808, 'y': 368.85026646486807}},
#     {'score': 0.9573307037353516,
#      'part': 'leftWrist',
#      'position': {'x': 513.1748874357131, 'y': 344.72253221779596}},
#     {'score': 0.5382570624351501,
#      'part': 'rightWrist',
#      'position': {'x': 576.7331447110193, 'y': 377.7758907371005}},
#     {'score': 0.9970320463180542,
#      'part': 'leftHip',
#      'position': {'x': 565.2119223883982, 'y': 392.1579105287945}},
#     {'score': 0.9955406188964844,
#      'part': 'rightHip',
#      'position': {'x': 583.4175109561352, 'y': 395.2229002121209}},
#     {'score': 0.9994198679924011,
#      'part': 'leftKnee',
#      'position': {'x': 533.1899578312363, 'y': 419.68256975666486}},
#     {'score': 0.9932212829589844,
#      'part': 'rightKnee',
#      'position': {'x': 583.2705294509952, 'y': 420.5354328114197}},
#     {'score': 0.9933654069900513,
#      'part': 'leftAnkle',
#      'position': {'x': 503.7792750547888, 'y': 461.1810554854791}},
#     {'score': 0.9025731086730957,
#      'part': 'rightAnkle',
#      'position': {'x': 606.5030665736966, 'y': 440.4511164263588}}]},
#   {'score': 0.626817021299811,
#    'keypoints': [{'score': 0.4314284026622772,
#      'part': 'nose',
#      'position': {'x': 1299.60643629367, 'y': 709.7628141185027}},
#     {'score': 0.29069116711616516,
#      'part': 'leftEye',
#      'position': {'x': 1319.5221731921706, 'y': 693.9172007471477}},
#     {'score': 0.28300121426582336,
#      'part': 'rightEye',
#      'position': {'x': 1298.8519953788443, 'y': 708.2381341721817}},
#     {'score': 0.9610827565193176,
#      'part': 'leftEar',
#      'position': {'x': 1317.9839012203145, 'y': 694.2287636249482}},
#     {'score': 0.8826542496681213,
#      'part': 'rightEar',
#      'position': {'x': 1288.703150147356, 'y': 707.4435710807813}},
#     {'score': 0.9993013143539429,
#      'part': 'leftShoulder',
#      'position': {'x': 1314.3488541220904, 'y': 726.7000980129275}},
#     {'score': 0.9953658580780029,
#      'part': 'rightShoulder',
#      'position': {'x': 1357.0013501813796, 'y': 719.4906353743899}},
#     {'score': 0.9148390293121338,
#      'part': 'leftElbow',
#      'position': {'x': 1303.4152672192577, 'y': 759.6088940215069}},
#     {'score': 0.889751672744751,
#      'part': 'rightElbow',
#      'position': {'x': 1359.4898338389307, 'y': 752.5164535776994}},
#     {'score': 0.7469696402549744,
#      'part': 'leftWrist',
#      'position': {'x': 1292.2674302501177, 'y': 778.7170160432415}},
#     {'score': 0.7793818116188049,
#      'part': 'rightWrist',
#      'position': {'x': 1362.9049662757902, 'y': 769.8108043521901}},
#     {'score': 0.9689626693725586,
#      'part': 'leftHip',
#      'position': {'x': 1316.764335560888, 'y': 782.1414413105057}},
#     {'score': 0.9731080532073975,
#      'part': 'rightHip',
#      'position': {'x': 1346.9621322154999, 'y': 782.1678903867302}},
#     {'score': 0.40984442830085754,
#      'part': 'leftKnee',
#      'position': {'x': 1265.6747587075395, 'y': 769.652980101997}},
#     {'score': 0.020849138498306274,
#      'part': 'rightKnee',
#      'position': {'x': 1358.6909180687608, 'y': 794.7339723329196}},
#     {'score': 0.10655820369720459,
#      'part': 'leftAnkle',
#      'position': {'x': 1276.9909001629005, 'y': 804.2203911627599}},
#     {'score': 0.002099752426147461,
#      'part': 'rightAnkle',
#      'position': {'x': 1375.4585873571675, 'y': 805.3495580418684}}]},
#   {'score': 0.8371166096014135,
#    'keypoints': [{'score': 0.27614492177963257,
#      'part': 'nose',
#      'position': {'x': 383.60851684313144, 'y': 556.2119824361553}},
#     {'score': 0.22982224822044373,
#      'part': 'leftEye',
#      'position': {'x': 384.0054959786519, 'y': 551.9657474564101}},
#     {'score': 0.17725297808647156,
#      'part': 'rightEye',
#      'position': {'x': 384.15528377075765, 'y': 551.6783302164987}},
#     {'score': 0.8996543884277344,
#      'part': 'leftEar',
#      'position': {'x': 377.5022451422188, 'y': 553.9159808910905}},
#     {'score': 0.8706039786338806,
#      'part': 'rightEar',
#      'position': {'x': 392.6135844005628, 'y': 553.6902578575359}},
#     {'score': 0.9932739734649658,
#      'part': 'leftShoulder',
#      'position': {'x': 366.0288457263275, 'y': 579.157855280144}},
#     {'score': 0.9985941648483276,
#      'part': 'rightShoulder',
#      'position': {'x': 410.40195340208345, 'y': 576.1001326967566}},
#     {'score': 0.9657025337219238,
#      'part': 'leftElbow',
#      'position': {'x': 355.1080439528276, 'y': 605.9863835934749}},
#     {'score': 0.9878292083740234,
#      'part': 'rightElbow',
#      'position': {'x': 431.5773532131638, 'y': 602.2752772815512}},
#     {'score': 0.9524412155151367,
#      'part': 'leftWrist',
#      'position': {'x': 348.7737713223986, 'y': 630.946604994802}},
#     {'score': 0.935217022895813,
#      'part': 'rightWrist',
#      'position': {'x': 443.84258352415395, 'y': 636.8562104912729}},
#     {'score': 0.9978746771812439,
#      'part': 'leftHip',
#      'position': {'x': 386.59041430084, 'y': 645.7433146087508}},
#     {'score': 0.9983861446380615,
#      'part': 'rightHip',
#      'position': {'x': 404.0685349618005, 'y': 641.6474805770755}},
#     {'score': 0.9957602024078369,
#      'part': 'leftKnee',
#      'position': {'x': 395.71989334299326, 'y': 699.45121700429}},
#     {'score': 0.9912469983100891,
#      'part': 'rightKnee',
#      'position': {'x': 387.9189510095432, 'y': 696.8044327801186}},
#     {'score': 0.9905133843421936,
#      'part': 'leftAnkle',
#      'position': {'x': 408.0154145172473, 'y': 749.6358494775134}},
#     {'score': 0.9706643223762512,
#      'part': 'rightAnkle',
#      'position': {'x': 362.82232833176516, 'y': 745.0365740662018}}]},
#   {'score': 0.8870415705091813,
#    'keypoints': [{'score': 0.8510965704917908,
#      'part': 'nose',
#      'position': {'x': 734.1031903827682, 'y': 262.93655321792255}},
#     {'score': 0.8677736520767212,
#      'part': 'leftEye',
#      'position': {'x': 736.7470037946094, 'y': 259.5912471850575}},
#     {'score': 0.5879550576210022,
#      'part': 'rightEye',
#      'position': {'x': 732.9761542702436, 'y': 260.35934771202255}},
#     {'score': 0.8721376061439514,
#      'part': 'leftEar',
#      'position': {'x': 743.2469600684634, 'y': 254.3228651423595}},
#     {'score': 0.2131834328174591,
#      'part': 'rightEar',
#      'position': {'x': 737.0599611475227, 'y': 258.8059565859816}},
#     {'score': 0.9708515405654907,
#      'part': 'leftShoulder',
#      'position': {'x': 755.6888343359201, 'y': 258.34611127942645}},
#     {'score': 0.9756870269775391,
#      'part': 'rightShoulder',
#      'position': {'x': 739.3911668405997, 'y': 259.8085326900515}},
#     {'score': 0.9932615160942078,
#      'part': 'leftElbow',
#      'position': {'x': 769.9314288396514, 'y': 275.286505294218}},
#     {'score': 0.9819937348365784,
#      'part': 'rightElbow',
#      'position': {'x': 740.8031947925297, 'y': 280.1308789013577}},
#     {'score': 0.9928951859474182,
#      'part': 'leftWrist',
#      'position': {'x': 760.8045877088769, 'y': 293.5800585971961}},
#     {'score': 0.9593830108642578,
#      'part': 'rightWrist',
#      'position': {'x': 734.7522400845303, 'y': 297.00639048833364}},
#     {'score': 0.9980243444442749,
#      'part': 'leftHip',
#      'position': {'x': 773.6855437603783, 'y': 288.60317832923636}},
#     {'score': 0.9969238042831421,
#      'part': 'rightHip',
#      'position': {'x': 757.1237041825212, 'y': 290.4120283374753}},
#     {'score': 0.987978458404541,
#      'part': 'leftKnee',
#      'position': {'x': 764.3031805133105, 'y': 323.1905232252653}},
#     {'score': 0.9945390224456787,
#      'part': 'rightKnee',
#      'position': {'x': 751.3878181132485, 'y': 323.40401114624126}},
#     {'score': 0.888724684715271,
#      'part': 'leftAnkle',
#      'position': {'x': 783.4895236090327, 'y': 366.3310089235289}},
#     {'score': 0.9472980499267578,
#      'part': 'rightAnkle',
#      'position': {'x': 762.8600707813149, 'y': 366.584824854621}}]},
#   {'score': 0.9427676481359145,
#    'keypoints': [{'score': 0.9149247407913208,
#      'part': 'nose',
#      'position': {'x': 375.5513109428606, 'y': 281.9225449289358}},
#     {'score': 0.8628238439559937,
#      'part': 'leftEye',
#      'position': {'x': 376.8318416652608, 'y': 278.70072825281414}},
#     {'score': 0.9300527572631836,
#      'part': 'rightEye',
#      'position': {'x': 372.49454829040985, 'y': 278.7407003482045}},
#     {'score': 0.6834704875946045,
#      'part': 'leftEar',
#      'position': {'x': 378.8217731182941, 'y': 280.2539174907955}},
#     {'score': 0.8857182264328003,
#      'part': 'rightEar',
#      'position': {'x': 368.10370174865153, 'y': 280.8183486440979}},
#     {'score': 0.99465012550354,
#      'part': 'leftShoulder',
#      'position': {'x': 378.9811210560888, 'y': 291.3590077767132}},
#     {'score': 0.9859417676925659,
#      'part': 'rightShoulder',
#      'position': {'x': 359.0225165756454, 'y': 298.4464395256968}},
#     {'score': 0.9897069931030273,
#      'part': 'leftElbow',
#      'position': {'x': 397.8577783134546, 'y': 314.1849269156233}},
#     {'score': 0.973739743232727,
#      'part': 'rightElbow',
#      'position': {'x': 357.2110724859916, 'y': 315.7311405730619}},
#     {'score': 0.9675685167312622,
#      'part': 'leftWrist',
#      'position': {'x': 405.9057171693009, 'y': 320.0255415187551}},
#     {'score': 0.9582129716873169,
#      'part': 'rightWrist',
#      'position': {'x': 392.3763132738263, 'y': 321.54360919601993}},
#     {'score': 0.9956414699554443,
#      'part': 'leftHip',
#      'position': {'x': 387.91558512855556, 'y': 342.2755210147573}},
#     {'score': 0.9678325057029724,
#      'part': 'rightHip',
#      'position': {'x': 375.8081285319525, 'y': 343.5003375833534}},
#     {'score': 0.9502295851707458,
#      'part': 'leftKnee',
#      'position': {'x': 404.43661057279354, 'y': 381.03110596377473}},
#     {'score': 0.986001193523407,
#      'part': 'rightKnee',
#      'position': {'x': 417.8707630803969, 'y': 407.60771583271185}},
#     {'score': 0.9852748513221741,
#      'part': 'leftAnkle',
#      'position': {'x': 402.1597112385968, 'y': 431.3819410416654}},
#     {'score': 0.9952602386474609,
#      'part': 'rightAnkle',
#      'position': {'x': 419.21571172042735, 'y': 452.4147852876157}}]},
#   {'score': 0.5686012979815988,
#    'keypoints': [{'score': 0.2343865931034088,
#      'part': 'nose',
#      'position': {'x': 1185.5387787229558, 'y': 695.1018555151941}},
#     {'score': 0.16997426748275757,
#      'part': 'leftEye',
#      'position': {'x': 1177.7975290277031, 'y': 694.9376848442303}},
#     {'score': 0.1556316316127777,
#      'part': 'rightEye',
#      'position': {'x': 1191.7794282213133, 'y': 692.082064391422}},
#     {'score': 0.9413723945617676,
#      'part': 'leftEar',
#      'position': {'x': 1215.353528119205, 'y': 687.2909361493856}},
#     {'score': 0.8438385725021362,
#      'part': 'rightEar',
#      'position': {'x': 1191.4044221135114, 'y': 694.1368732303639}},
#     {'score': 0.9951277375221252,
#      'part': 'leftShoulder',
#      'position': {'x': 1154.874260052313, 'y': 717.2802578621968}},
#     {'score': 0.988829493522644,
#      'part': 'rightShoulder',
#      'position': {'x': 1218.102323098129, 'y': 711.6584733817565}},
#     {'score': 0.841530442237854,
#      'part': 'leftElbow',
#      'position': {'x': 1123.3756569387315, 'y': 756.7272163032452}},
#     {'score': 0.8986760973930359,
#      'part': 'rightElbow',
#      'position': {'x': 1204.7321940825673, 'y': 751.3359444765325}},
#     {'score': 0.6037924289703369,
#      'part': 'leftWrist',
#      'position': {'x': 1109.2500073389615, 'y': 765.0169149428554}},
#     {'score': 0.5745197534561157,
#      'part': 'rightWrist',
#      'position': {'x': 1209.212806148029, 'y': 756.5251189588054}},
#     {'score': 0.9113421440124512,
#      'part': 'leftHip',
#      'position': {'x': 1145.9274648530652, 'y': 783.137546554182}},
#     {'score': 0.8819822072982788,
#      'part': 'rightHip',
#      'position': {'x': 1186.1627591343854, 'y': 781.9849693622291}},
#     {'score': 0.47143545746803284,
#      'part': 'leftKnee',
#      'position': {'x': 1083.1028352587412, 'y': 775.8372429908871}},
#     {'score': 0.033972084522247314,
#      'part': 'rightKnee',
#      'position': {'x': 1180.828851012255, 'y': 794.4969052789116}},
#     {'score': 0.10370919108390808,
#      'part': 'leftAnkle',
#      'position': {'x': 1066.3653701860806, 'y': 788.5057413573901}},
#     {'score': 0.016101568937301636,
#      'part': 'rightAnkle',
#      'position': {'x': 1191.6710920798198, 'y': 802.3288549409995}}]},
#   {'score': 0.4474380594842574,
#    'keypoints': [{'score': 0.8681598901748657,
#      'part': 'nose',
#      'position': {'x': 476.13915352517745, 'y': 327.07329704278055}},
#     {'score': 0.611355185508728,
#      'part': 'leftEye',
#      'position': {'x': 467.015848804935, 'y': 328.1423830795949}},
#     {'score': 0.9448294639587402,
#      'part': 'rightEye',
#      'position': {'x': 470.1782067799836, 'y': 328.0582653753059}},
#     {'score': 0.3498283326625824,
#      'part': 'leftEar',
#      'position': {'x': 476.4363677742776, 'y': 326.82552992161266}},
#     {'score': 0.8955947756767273,
#      'part': 'rightEar',
#      'position': {'x': 466.030011882273, 'y': 324.36412984329036}},
#     {'score': 0.9799946546554565,
#      'part': 'leftShoulder',
#      'position': {'x': 471.7577458008398, 'y': 342.11081690680834}},
#     {'score': 0.9949179887771606,
#      'part': 'rightShoulder',
#      'position': {'x': 443.7613709910532, 'y': 340.56933282524915}},
#     {'score': 0.9703690409660339,
#      'part': 'leftElbow',
#      'position': {'x': 485.0312327195643, 'y': 345.7685864818778}},
#     {'score': 0.9690847396850586,
#      'part': 'rightElbow',
#      'position': {'x': 433.90594099880605, 'y': 356.40191332161737}},
#     {'score': 0.9325779676437378,
#      'part': 'leftWrist',
#      'position': {'x': 485.3742586093002, 'y': 343.3309474098827}},
#     {'score': 0.9423339366912842,
#      'part': 'rightWrist',
#      'position': {'x': 445.4298216716181, 'y': 352.55831628454206}},
#     {'score': 0.9951775074005127,
#      'part': 'leftHip',
#      'position': {'x': 462.27811120244, 'y': 380.5048725534766}},
#     {'score': 0.9927194118499756,
#      'part': 'rightHip',
#      'position': {'x': 396.7217482538259, 'y': 354.2276014361266}},
#     {'score': 0.9994198679924011,
#      'part': 'leftKnee',
#      'position': {'x': 533.1899578312363, 'y': 419.68256975666486}},
#     {'score': 0.986001193523407,
#      'part': 'rightKnee',
#      'position': {'x': 417.8707630803969, 'y': 407.60771583271185}},
#     {'score': 0.9933654069900513,
#      'part': 'leftAnkle',
#      'position': {'x': 503.7792750547888, 'y': 461.1810554854791}},
#     {'score': 0.9952602386474609,
#      'part': 'rightAnkle',
#      'position': {'x': 419.21571172042735, 'y': 452.4147852876157}}]}]}


# In[ ]:




