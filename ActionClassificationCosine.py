#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
import pandas as pd 
import sklearn.metrics.pairwise as pairwise
import vptree
from sklearn import preprocessing 
import pickle
import time
import copy


# In[2]:


import ActionClassificationCosineDatasetGen as Dataset


# #ActionClassification Via Similarity Measurement

# In[3]:


print('Loading Poses Data....')
fname = 'custom_normalized_Flase_framelist_[8]_total_37085'
DATA_DF = Dataset.read_dataset(name=fname, norm=True)
ARGS = {'TREE':None,
        'buildTree':False,
       'X':DATA_DF.loc[:,DATA_DF.columns != 'label'],
       'Y': DATA_DF.loc[:,DATA_DF.columns == 'label'],
       'TREEPATH':'./dataandmodles/models/vptrees',
       'cosineDistanceMatching':False,
       'weightedDistanceMatching': True}
TIME_TRACKER = []
START_TIME = 0
END_TIME = 0


# In[4]:


def startTimer():
    global START_TIME
    START_TIME = time.time()


# In[5]:


def endTimer():
    global END_TIME,START_TIME
    global TIME_TRACKER
    END_TIME = time.time()
    
    timeTaken = round(END_TIME - START_TIME,3)
    TIME_TRACKER.append(timeTaken)
    
    START_TIME,END_TIME = 0,0
    


# In[6]:


def getMeanRunTime():
    global TIME_TRACKER
    mean = round(np.mean(np.array(TIME_TRACKER)) ,3)
    TIME_TRACKER = []
    return mean


# In[7]:


def weightedDistanceMatching(poseVector1,poseVector2):
    '''
    
    Parameters
    ----------
    (poseVector1 : a 2D list pose vector of a human + theta [[poseVec],[thetaVector]]
    theta : weigths for human pose vector
    Example theta = [w_0x , w_0y.........w_17x, w_17y])
    or 
    (poseVector1 : a 1D list pose vector of a human, used for building tree. 
    theta : A list with of ones. 
    Example [1,1,.....1] (17x1) )
    
    poseVector2 : pose vector that is to be compared with the human
    
    
    
    Returns
    ------------
    weigtheDistance  : 
    '''
    poseVector1 = np.array(poseVector1) 
    if poseVector1.shape[0] == 36:
        poseVector1 = poseVector1.reshape(1,-1)
        theta = np.ones_like(poseVector1)
    elif poseVector1.shape[0] == 72:
        poseVector1 = poseVector1.reshape(2,36)
        theta = poseVector1[1]
        poseVector1 = poseVector1[0]
   

    poseVector2 = np.array(poseVector2).reshape(1,-1)
    term1 = 1/ np.sum(theta)
    #Finding term 2
    distanceTranspose = np.absolute(poseVector1 - poseVector2).transpose()
    term2 = np.matmul(theta,distanceTranspose)

        
    
    return term1 * term2
    


# In[8]:


def cosineDistanceMatching(poseVector1,poseVector2):
    '''
    Returns
    -------
    distacne: the cosine similarity as a distance function between the two L2 normalized vectors.
    The distance is inversely porportional to the similarity between the two vectors
    '''
    poseVector1 , poseVector2 = np.array(poseVector1).reshape(1,-1) , np.array(poseVector2).reshape(1,-1)
    cosineDistance = pairwise.cosine_distances(poseVector1 , poseVector2) 
    distance = np.sqrt(cosineDistance * 2)
    return distance

    


# In[9]:


def getPoseData():
    '''
    Helper function of buildVPTree()
    '''
    return ARGS['X'].values.tolist()


# In[10]:


def buildVPTree():
    '''
    This fuction will build the vptree for the given dataset
    '''
    poseData = getPoseData()
    if ARGS['cosineDistanceMatching']:
        ARGS['TREE'] = vptree.VPTree(poseData,cosineDistanceMatching)
    else:
        ARGS['TREE'] = vptree.VPTree(poseData,weightedDistanceMatching)
        


# In[11]:


def findMatch(pose):
    '''
    Helper fuction for getPred()
    Will get the closest vectors to pose using vantage point tree
    
    Parameter
    ---------
    pose : A list with coordinates of 1 human keyspoints.
    If consine distance tree is used 
    Example : [0_x, 0_y, 1_x,......17_y]
    If weighted distance tree is used, then it also contains theta/weigths. 
    The weigths for x and y for key points are equal.
    Example : [0_x, 0_y, 1_x,......17_y , w_0_x, w_0_y.........w_17_y]
    
    
    Returns
    ---------
    similarPoses : a list of tuples. 1 tuplexand vectors. (0.5 , [..])
    '''
    startTimer()
    similarPoses= ARGS['TREE'].get_nearest_neighbor(pose)
    endTimer()
    
    return similarPoses


# In[12]:


def getSimilarPosesClassesIndex(similarPoses):
    '''
    Helper function of getPred()
    
    Parameters
    --------------
    similarPoses : a list of tuples. 1 tuple contains distacne and vectors. [(0.5 , [..]) , ...] 
    
    Returns 
    -------------
    index : a list of index in the ARGS['X'] dataframe of the vectors found by vptrees
    '''
    index = []
    for d,p in similarPoses:
        poseIndexList = ARGS['X'][ARGS['X'] == p].dropna().index.tolist()
        index.append(poseIndexList[0])
    return list(index)


# In[13]:


def getSimilarPosesClasses(indexs):
    '''
    Helper function of getPred()
    
    Parameters
    ----------
    index : a list of index in the ARGS['X'] dataframe of the vectors found by vptrees
    
    Returns
    --------
    list : A 2D list with classes of each pose. [[similarVectorClass1],  [....]]
    '''
    return ARGS['Y'].iloc[indexs].values


# In[14]:


def getPred(poses):
    '''
    This function finds the classes of pose in poses using VPTREES
    
    Parameters
    --------- 
    poses : a 2D list that contains poses for all the humans detected in the frame. 
    Example [[0_x, 0_y.....] , [0_x, 0_y.....] , ....... ]
    The poses must contain coordinates must be relative to image of size (244 x 244)

    Returns
    ---------
    poesesClasses : a 2D list that contains poses class for each human 
    Example [['walk'] , ['shoot'] ,......]  
    '''
    similarPoses = []
    for pose in poses:
       similarPose = findMatch(pose)
       similarPoses.append(similarPose)
    classIndexs = getSimilarPosesClassesIndex(similarPoses)
    posesClasses = getSimilarPosesClasses(classIndexs)
    return posesClasses
    


# In[15]:


def posentPredPreprocess(posenetPred, bboxes, h,w):
    '''
    Parameters
    -----------
    posenetPred : The raw output of posenet for keypoints
    
    bboxes : The raw output of pose for bounding boxes
             Example: 
    
    h : The height/img.shape[0]/row of the image that was passed to the posenet model
    
    w : The width/img.shape[1]/cols of the image that was passed to the posenet model
    
    Return
    ---------
    posenetPred : The raw output of posenet for keypoints.
    Where the coordinates relative to cropped and scaled image of (244x244) 
    
    '''
    bboxes = getBoxCoord(bboxes,h,w)
    
    for i,box in enumerate(bboxes):
        left,top  = box[0][0] , box[0][1]
        right,bottom = box[1][0] , box[1][1]
        roiH,roiW = abs(bottom-top) , abs(right-left)
        for keypoint in posenetPred['detectionList'][i]['keypoints']:
            keyPointPosition = mapCoordToCroppedImage(keypoint['position'],left,top)
            keyPointPosition = mapCoordToSqaureImage(keyPointPosition ,roiH ,roiW)
            keypoint['position'] = keyPointPosition
    return posenetPred

def getStackedPoses(posenetPred,bboxes,inputImgH,inputImgW,cosine=False):
    '''
    Helper function of fit
    This function generate posenetPred to contain keypoints coords then weigths
    
    Parameter
    ---------
    posenetPred : The non-empty output of Posenet model
    Example {'detectionList': [{keypoints:[{score: , part:.. ,position:..}]} , human2 , ....]
    
    bboxes : The bboxes from posenet. 
             Example : {'bbox':[[box1] , [box2],.......]}
    
    inputImgH : The height/rows/img.shape[0] in the input image
    
    inputImgW : The height/rows/img.shape[1] in the input image.
    They will be used to scale the coordinates in the posenet results to image size of (244,244)
    Because, the poses stored in the vantage tree are relative to (244,244)
    
    cosine : True cosine distacne is used to find the most similar vectors. False weighted distance
    
    
    Return
    -------
    stackedPoses : 1D list containing pose and weights [0_x.......17_w_y]
    poses : 1D list with only pose [0_x.....17_y]
    '''
    posenetPred = posentPredPreprocess(posenetPred,bboxes,inputImgH,inputImgW)
    if cosine:
        poses = Dataset.generate_data_posenet(posenetPred, norm=True)
        return None,poses
    else:
        poses, theta = Dataset.generate_data_posenet(posenetPred , norm=False)
        stackedPoses = np.hstack((poses,theta))
        return stackedPoses,poses


# In[16]:


def fit(posenetPred,bboxes,inputImgH,inputImgW, cosine = False):
    '''
    This function take posenet results, use VPTREE that was loaded and find 
    classes for each poses.
    It also uses Tracker class to update the predicted class.
    
    Parameter
    ---------
    posenetPred : The non-empty output of Posenet model
    Example {'detectionList': [{keypoints:[{score: , part:.. ,position:..}]} , human2 , ....]
    
    bboxes : The bboxes from posenet. 
             Example : {'bbox':[[box1] , [box2],.......]}
    
    inputImgH : The height/rows/img.shape[0] in the input image
    
    inputImgW : The height/rows/img.shape[1] in the input image.
    They will be used to scale the coordinates in the posenet results to image size of (244,244)
    Because, the poses stored in the vantage tree are relative to (244,244)
    
    cosine : True cosine distacne is used to find the most similar vectors. False weighted distance
    
    Return
    ------
    posesClasses : a 2D list that contains poses class for each human 
    Example [['walk'] , ['shoot'] ,......]  
    '''  
    stackedPoses,_ = getStackedPoses(posenetPred,bboxes,inputImgH,inputImgW,cosine=False)
    
    bboxs = getBoxCoord(bboxes,inputImgH,inputImgW)
    posesClasses = getPred(stackedPoses)
    
#     if len(list(Tracker.previousFramePose.keys())) != 0 :
#         posesClasses = updatePosesClasses(stackedPoses,posesClasses,bboxs)
    
#     #For testing
#     if len(list(previousFramePose.keys())) != 0 :
#         prevFrameMidpoints = list(Tracker.track(bboxs).keys())
#     else:
#          prevFrameMidpoints = [(0,0)]
    
#     Tracker.setTracker(bboxs,posesClasses)
#     setPreviousFramePoses(bboxs,poses)
    
    return posesClasses


# #Test

# In[17]:


import DrawPose
previousFramePose = {}


# In[18]:


# videoPath ='./dataandmodles/data/1v1.mp4'
# cap = cv2.VideoCapture(videoPath)
# count = 0
# while cap.isOpened():
#     ret,frame = cap.read()
#     if ret :
#         #do stuff
#         start = time.time()
#         poses,bboxes = getPoses(frame)
#         end = time.time()
#         print(f'Posenet Time:{round(end-start,3)}')
        
#         if len(bboxes['bbox']) != 0:
#             posesCopy = copy.deepcopy(poses)
#             #frame = draw_visibilty_cone(poses,frame)
#             classes,prevFrameMidpoints = fit(poses,
#                                             bboxes,
#                                             inputImgH=frame.shape[0],
#                                             inputImgW=frame.shape[1])
#             frame = DrawPose.draw_skel_and_kp(posesCopy,frame,[255,0,0],list(classes))
#             bboxs = getBoxCoord(bboxes,frame.shape[0],frame.shape[1])
#             midPoints = Tracker.convertBboxsToMidPoint(bboxs)
#             for i in range(len(bboxs)):
#                 left, top, right, bottom = bboxs[i][0][0],bboxs[i][0][1],bboxs[i][1][0],bboxs[i][1][0]
#                 actionClass = classes[i][0]
#                 DrawPose.drawPred(frame,actionClass, left, top, right, bottom,(255,0,255),confidence=0,yolo=True)
#                 m1 = (int(midPoints[i][0]),int(midPoints[i][1]) )
#                 try:
#                     m2 = (int(prevFrameMidpoints[i][0]) , int(prevFrameMidpoints[i][1]))
#                     cv2.line(frame,m2 , m1 ,(255,0,255),  1)
#                 except:
#                     continue
# #         End of do stuff
#         cv2.imshow('test', frame)
#         #cv2.imwrite(f"./dataandmodles/data/coneVision/{count}.png" , frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break  
            
#     else:
#         cap.release()
#         break
#     count += 1
    
# cap.release()
# cv2.destroyAllWindows()


# #Saving and loading trees

# In[19]:


def saveTree(fileName):
    '''
    Save the tree to the path defined at  ARGS['TREEPATH']
    Parameters
    -----------
    fileName : The file name of the plk file
    '''
    
    file_vptree = f"{ARGS['TREEPATH']}/{fileName}.pkl"
    with open(file_vptree, 'wb') as output:
        pickle.dump(ARGS['TREE'], output, pickle.HIGHEST_PROTOCOL)


# In[20]:


def loadTree():
    '''
    Loads the tree based on the switch defined at ARGS. Sw
    to the path defined at  ARGS['TREEPATH']
    Parameters
    -----------
    fileName : The file name of the plk file
    '''
    if ARGS['cosineDistanceMatching']:
        fileName = 'vptreeCosineDistance'
    elif ARGS['weightedDistanceMatching']:
        fileName = f'vptreeWeightedDistance_{fname}'
        
    file_vptree = f"{ARGS['TREEPATH']}/{fileName}.pkl"
    with open(file_vptree, 'rb') as input:
        ARGS['TREE'] = pickle.load(input) 

        
#initailze the VPTREE

if ARGS['buildTree'] == False:
    print('Initializing VPTREE....')
    loadTree()
    print('VPTREE Ready To Use....')
else:
    print('Building VPTREE....')
    if ARGS['cosineDistanceMatching']:
        tree_name = f'vptreeCosineDistance_{fname}'
    elif ARGS['weightedDistanceMatching']:
        tree_name = f'vpTreeWeightedDistance_{fname}'
    buildVPTree()
    saveTree(tree_name)
    print('VPTREE SAVED....')


# #Visualization Debugger

# In[78]:


import SinglePlayerPoseDatasetGen as draw
import DrawPose
import requests as req
import base64
TESTARGS = {'isModelLoaded' : False,
            'debugger' : False,
            'testImage1' : './dataandmodles/data/weightedDistacne/6_resizedRoi.png.png',
            'testResultPath' : './dataandmodles/data/weightedDistacne'
           }


# In[22]:


def loadPosenetModel():
    posenet = req.get(url='http://localhost:3000/loadModel')
    TESTARGS['isModelLoaded'] = True 

    
if TESTARGS['isModelLoaded'] == False and TESTARGS['debugger'] == True:
    loadPosenetModel()


# In[23]:


def convertToBase64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    my_string = base64.b64encode(buffer)
    my_string = my_string.decode('utf-8')
    return my_string
#Sending data to server
def getPoses(image):
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image_string = convertToBase64(image)
    url2 = 'http://localhost:3000/postImage'
    data = {'imgBase64':'data:image/png;base64,'+image_string}
    r = req.post(url=url2 , data = data)
    poses = r.json()
    
    url2 = 'http://localhost:3000/getBBox'
    r = req.post(url=url2)
    bboxes = r.json()
    return poses,bboxes


# In[24]:


def load_image():
    img = cv2.imread(TESTARGS['testImage1'])
    return img, img.shape[0] , img.shape[1]

def write_image(img,fileName):
    cv2.imwrite(f'{TESTARGS["testResultPath"]}/{fileName}.png' , img)
    
def data_to_dict(data):
    '''
    Parameterse
    -----------
    data : [0_x , 0_y.....]
    
    Returns:
    -------------
    pose_dict : {0:(x,y)......}
    '''
    coords_tuples = list(zip(*[data[i::2] for i in range(2)]) )
    pose_dict = {i:t for i,t in enumerate(coords_tuples)}
    return pose_dict


# In[79]:


def visualizePoseAndSimlarPose(posenetPred,h,w):
    
    poses,theta = Dataset.generate_data_posenet(posenetPred=posenetPred,norm=False)
    poses = np.hstack((poses,theta))
    
    for i,pose in enumerate(poses): 
        pose_image = np.zeros((244,244,3))
        similarPose_image = np.ones((244,244,3))

        #Goal is to find similarpose_dict and pose_dict
        similarPose = findMatch(pose)
        similarPose_dict = data_to_dict(similarPose[1])
        pose_dict = data_to_dict(pose[0:36])
        
        #Goal is to draw in pose_image and similar pose 
        pose_image = draw.draw_skel_and_kp(poses=[pose_dict] , img=pose_image)    
        similarPose_image = draw.draw_skel_and_kp(poses=[similarPose_dict] , img=similarPose_image)
        
        combined = np.concatenate((pose_image, similarPose_image), axis=1)
        write_image(combined , f'6_resizdRoiPose_{i}')
    print('All test images are saved!')
        
        
def runTest():
    img,h,w  = load_image()
    posenetPred,bboxes = getPoses(img)
    
    posenetPred = posentPredPreprocess(posenetPred, bboxes,h,w)
    visualizePoseAndSimlarPose(posenetPred,h,w)
    
        


# In[26]:


def getBoxCoord(bboxes,imgh,imgw):
    boxes = []
    for box in bboxes['bbox']:
        points = [tuple(point.values()) for point in box ]
        n = 30
        points  = np.array(points,dtype=np.int)
        
        left,top = (points[0][0]-n ,points[0][1]-n) #left_top (w,h)/ (col,row)
        right,bottom = (points[2][0]+n ,points[2][1]+n ) #right_bottom
        #Applying boundary condition
        point1 = ( max(0,left) , max(0,top) )
        point2 = ( min(right,imgw) , min(bottom,imgh) )
    
        boxes.append([point1 , point2])
    return boxes 


# In[27]:


def cropRoi(image,top,left,bottom,right):
    h,w = image.shape[:2]
    roi = image[ top:bottom , left:right , :  ]
    return roi


# In[28]:


def mapCoordToCroppedImage(keypointPosition, left, top):
    '''
    Parameter
    ---------
    keypointPosition : A list containing dictionary of key points for a human
    Example {'x': 598, 'y': 509}}    

    left : x coord of the leftTop point / col / shape[1]. 
    
    top : y coord of the leftTop point / row /shape[0]
    
    Return
    -----------
    keypointPosition : A list containing dictionary of key points for a human.
    The coords are relatvie to the leftTop coordinates of the cropped image
    '''
    keypointPosition['x'] = abs(int(keypointPosition['x'] - left))
    keypointPosition['y'] = abs(int(keypointPosition['y'] -  top))
    
    return keypointPosition


# In[29]:


def mapCoordToSqaureImage(keypointPosition , imgH , imgW):
    '''
    Parameter
    ----------
    keypointPosition : A list containing dictionary of key points for a human 
    Example {'x': 598, 'y': 509} 
    
    imgH : Number of rows
    
    imgW : Number of cols 
    
    Return
    -----------
    keypointPosition : A list containing dictionary of key points for a human.
    The coords are relatvie to the square image of size (244*244)
    
    '''
    scaleFactor = Dataset.getScaleFactors(244,244,imgH , imgW)
    coord = Dataset.mapToSquareImage(list(keypointPosition.values()),scaleFactor)
    #coords (row,col)
    keypointPosition['x'] = coord[0]
    keypointPosition['y'] = coord[1]
    
    return keypointPosition


# In[30]:


def drawBBox():
    img,h,w  = load_image()
    posenetPred,bboxes = getPoses(img)
    bboxes = get_box_coord(bboxes)
    for box in bboxes:
        cv2.rectangle(img,box[0],box[1],(255,0,255))
    write_image(img,'bboxed')


# In[31]:


def testCroppedKeyPoints():
    img,h,w  = load_image()
    posenetPred,bboxes = getPoses(img)
    bboxes = getBoxCoord(bboxes,h,w)
    for i,box in enumerate(bboxes):
        left,top  = box[0][0] , box[0][1]
        right,bottom = box[1][0] , box[1][1]
        roi = cropRoi(img ,top,left, bottom,right)
        for keypoint in posenetPred['detectionList'][i]['keypoints']:
            keyPointPosition = mapCoordToCroppedImage(keypoint['position'],left,top)
            keyPointPosition = tuple(keyPointPosition.values())
            roi = cv2.circle(roi, keyPointPosition, 1, (255,0,255), 1)
        write_image(roi , f'keyPointTest_{i}')


# In[67]:


def testScaledCroppedKeyPoints():
    img,h,w  = load_image()
    posenetPred,bboxes = getPoses(img)
    bboxes = getBoxCoord(bboxes,h,w)
    for i,box in enumerate(bboxes):
        left,top  = box[0][0] , box[0][1]
        right,bottom = box[1][0] , box[1][1]
        roi = cropRoi(img ,top,left, bottom,right)
        roiSquare = cv2.resize(roi, (244,244),interpolation = cv2.INTER_AREA)
        for keypoint in posenetPred['detectionList'][i]['keypoints']:
            keyPointPosition = mapCoordToCroppedImage(keypoint['position'],left,top)
            keyPointPosition = mapCoordToSqaureImage(keyPointPosition ,roi.shape[0] ,roi.shape[1])
            keyPointPosition = tuple(keyPointPosition.values())
            print(keyPointPosition)
            cv2.circle(roiSquare, keyPointPosition, 1, (255,0,255), 1)
        write_image(roiSquare , f'scaled_keyPointTest_{i}')


# In[62]:


def testCrop():
    img,h,w  = load_image()
    posenetPred,bboxes = getPoses(img)
    boxes = getBoxCoord(bboxes,h,w)
    for i,box in enumerate(boxes):
        left,top  = box[0][0] , box[0][1]
        right,bottom = box[1][0] , box[1][1]
        roi = cropRoi(img ,top,left, bottom,right)
        print(roi.shape)
        write_image(roi, f'crop_{i}')
        


# In[63]:


#testCrop()


# In[64]:


#testScaledCroppedKeyPoints()


# In[66]:


#loadPosenetModel()


# In[68]:


#drawScaledCroppedSkeleton()


# In[37]:


#posentPredPreprocess(posenetPred, bboxes,h,w)


# In[38]:


#img,h,w  = load_image()
#posenetPred,bboxes = getPoses(img)


# In[73]:


#runTest()


# In[80]:


# runTest()
# img,h,w  = load_image()
# posenetPred,bboxes = getPoses(img)
# img = DrawPose.draw_skel_and_kp(posenetPred,img,(255,0,255))
# write_image(img,'8_resizdRoiPose_pose')


# #Cosine Test Data 1

# In[40]:


# d = np.array([[0.18694303, 0.11572664, 0.18694303, 0.11572664, 0.19830082,
#         0.12155902, 0.20566803, 0.13230287, 0.19983565, 0.13260983,
#         0.18264549, 0.12217295, 0.19615205, 0.10774549, 0.19185451,
#         0.10958729, 0.20198442, 0.14335369, 0.21242131, 0.16054385,
#         0.21088647, 0.17834795, 0.19031967, 0.14396762, 0.18755697,
#         0.16115779, 0.18510123, 0.17834795, 0.18663606, 0.11388483,
#         0.18540819, 0.11480574, 0.1909336 , 0.11449877, 0.18356639,
#         0.11572664],
#        [0.22652992, 0.04472884, 0.22652992, 0.04472884, 0.23198075,
#         0.04761457, 0.2332633 , 0.0522638 , 0.2332633 , 0.05899718,
#         0.22572833, 0.04841616, 0.22652992, 0.05242412, 0.22604897,
#         0.05579081, 0.23550775, 0.05547017, 0.23214107, 0.06236386,
#         0.23230139, 0.07005915, 0.23133948, 0.05579081, 0.23021725,
#         0.06252418, 0.23166011, 0.06941787, 0.22685056, 0.04408756,
#         0.2263696 , 0.04408756, 0.22797279, 0.04472884, 0.22556801,
#         0.04488916],
#        [0.1455586 , 0.14369843, 0.1455586 , 0.14369843, 0.15532451,
#         0.15811477, 0.17439129, 0.15160416, 0.17439129, 0.13300243,
#         0.13346747, 0.15811477, 0.15113912, 0.15299929, 0.17392625,
#         0.13672277, 0.15253425, 0.19950364, 0.15578955, 0.23345181,
#         0.16183512, 0.26879511, 0.13300243, 0.1990386 , 0.13067721,
#         0.23484694, 0.13067721, 0.26833007, 0.14834886, 0.14183825,
#         0.14230329, 0.14137321, 0.15578955, 0.14741877, 0.13625773,
#         0.14788382],
#        [0.18277657, 0.1276963 , 0.18277657, 0.1276963 , 0.17603205,
#         0.13376637, 0.17333424, 0.14478242, 0.17063643, 0.14950359,
#         0.19042036, 0.13444082, 0.19559116, 0.14343352, 0.19356781,
#         0.15040286, 0.17895467, 0.15332549, 0.1818773 , 0.16861307,
#         0.18614883, 0.18210212, 0.18907146, 0.15287585, 0.18907146,
#         0.16838825, 0.19424226, 0.1832262 , 0.18097803, 0.12702185,
#         0.18300139, 0.12634739, 0.1780554 , 0.12724666, 0.18435029,
#         0.12792112],
#        [0.20281541, 0.09841091, 0.20281541, 0.09841091, 0.21267583,
#         0.10131103, 0.21634933, 0.10614458, 0.21924945, 0.11001141,
#         0.20474883, 0.10111769, 0.2064889 , 0.10556455, 0.20300875,
#         0.10498453, 0.21480259, 0.11310488, 0.21364254, 0.12315864,
#         0.21731603, 0.13340575, 0.20977571, 0.11271819, 0.21035573,
#         0.12335198, 0.21074242, 0.13050563, 0.2041688 , 0.09725086,
#         0.20204204, 0.0974442 , 0.20706893, 0.09725086, 0.2018487 ,
#         0.09763754],
#        [0.22443155, 0.05285568, 0.22443155, 0.05285568, 0.22982259,
#         0.05707476, 0.23228372, 0.06293459, 0.22935381, 0.06809124,
#         0.22454875, 0.05648878, 0.22478314, 0.06094225, 0.2226736 ,
#         0.06176262, 0.22982259, 0.0662161 , 0.22736146, 0.07336509,
#         0.2297054 , 0.08180325, 0.22665829, 0.0659817 , 0.22794745,
#         0.07359948, 0.23204933, 0.08110007, 0.22536912, 0.0522697 ,
#         0.22384557, 0.05238689, 0.22806464, 0.05262129, 0.22349398,
#         0.05309007],
#        [0.22211447, 0.05895235, 0.22211447, 0.05895235, 0.22414731,
#         0.06312502, 0.22371934, 0.0684746 , 0.22136553, 0.07232629,
#         0.22821299, 0.06259006, 0.23088778, 0.06676273, 0.23227867,
#         0.07040044, 0.22478926, 0.07104239, 0.22468227, 0.07821083,
#         0.22757104, 0.08388138, 0.22853396, 0.07147036, 0.2255382 ,
#         0.07810384, 0.22778502, 0.08388138, 0.2229704 , 0.05820341,
#         0.22254244, 0.05809641, 0.22511023, 0.05905934, 0.22511023,
#         0.05863137],
#        [0.21485147, 0.08675887, 0.21485147, 0.08675887, 0.21149442,
#         0.08990611, 0.21086497, 0.09525641, 0.20960607, 0.09546623,
#         0.21946742, 0.08906684, 0.22209012, 0.0937877 , 0.22471282,
#         0.09546623, 0.21611036, 0.09745948, 0.21946742, 0.09315825,
#         0.22009687, 0.09085028, 0.22009687, 0.09714476, 0.22083122,
#         0.09357789, 0.22649625, 0.09703985, 0.21191405, 0.08675887,
#         0.21537601, 0.08644415, 0.21296313, 0.08686378, 0.21653   ,
#         0.08633924],
#        [0.2191449 , 0.06245351, 0.2191449 , 0.06245351, 0.22806683,
#         0.06914496, 0.22946088, 0.07639402, 0.23308542, 0.08224904,
#         0.21496275, 0.06775091, 0.21301108, 0.07527878, 0.21496275,
#         0.08085499, 0.22639397, 0.08475833, 0.22750921, 0.09702599,
#         0.22695159, 0.10873602, 0.21886609, 0.0839219 , 0.22026014,
#         0.09842004, 0.22137539, 0.10706316, 0.22026014, 0.06133827,
#         0.21830847, 0.06133827, 0.22276944, 0.0621747 , 0.21802966,
#         0.06245351],
#        [0.17711887, 0.12412839, 0.17711887, 0.12412839, 0.19889578,
#         0.13501684, 0.20542885, 0.15171248, 0.18510374, 0.15824555,
#         0.16114914, 0.13792043, 0.15243837, 0.15461606, 0.16695631,
#         0.16187503, 0.18002246, 0.16550452, 0.17131169, 0.18147425,
#         0.20688065, 0.21341372, 0.16260093, 0.1705858 , 0.17857066,
#         0.19018502, 0.1466312 , 0.20833244, 0.17494118, 0.11832121,
#         0.17421528, 0.11977301, 0.17929656, 0.11904711, 0.1698599 ,
#         0.1204989 ]])


# #weigthed distance data 

# In[41]:


d = [[ 27., 179.,  27., 179.,  22., 188.,  21., 197.,  20., 205.,  30.,
       187.,  32., 196.,  31., 204.,  23., 206.,  22., 217.,  23., 228.,
        28., 206.,  28., 220.,  28., 233.,  26., 178.,  27., 178.,  24.,
       179.,  28., 179.],
      [ 37.,  68.,  37.,  68.,  39.,  72.,  39.,  76.,  35.,  72.,  37.,
        72.,  36.,  78.,  35.,  78.,  39.,  91.,  42., 107.,  42., 121.,
        37.,  91.,  40., 107.,  40., 122.,  37.,  66.,  36.,  66.,  38.,
        67.,  37.,  67.],
      [100., 149., 100., 149., 101., 154., 100., 166.,  99., 174., 106.,
       151., 108., 161., 106., 168., 104., 172., 101., 183., 100., 197.,
       108., 170., 108., 184., 108., 200., 100., 148., 100., 148., 101.,
       146., 103., 145.],
      [ 88.,  98.,  88.,  98.,  91., 100.,  87., 101.,  85., 101.,  95.,
       101.,  97., 109.,  96., 111.,  94., 115.,  89., 124.,  84., 136.,
        97., 116.,  97., 124., 101., 130.,  88.,  97.,  87.,  96.,  94.,
        95.,  92.,  98.],
      [217., 209., 217., 209., 219., 214., 217., 224., 215., 230., 226.,
       212., 227., 222., 227., 227., 219., 231., 211., 227., 213., 237.,
       224., 231., 226., 235., 229., 238., 220., 205., 216., 209., 220.,
       205., 215., 209.],
      [ 64., 164.,  64., 164.,  61., 171.,  59., 179.,  58., 186.,  68.,
       170.,  72., 178.,  74., 188.,  64., 190.,  66., 206.,  68., 221.,
        67., 189.,  64., 206.,  60., 220.,  64., 163.,  64., 163.,  63.,
       163.,  65., 163.],
      [122.,  77., 122.,  77., 126.,  76., 128.,  81., 127.,  86., 123.,
        76., 123.,  82., 122.,  87., 129.,  85., 127.,  95., 130., 108.,
       126.,  85., 125.,  95., 127., 108., 123.,  76., 122.,  77., 124.,
        75., 123.,  76.],
      [ 62.,  83.,  62.,  83.,  63.,  86.,  66.,  92.,  67.,  94.,  59.,
        88.,  59.,  93.,  65.,  95.,  64., 101.,  67., 112.,  67., 127.,
        62., 101.,  69., 120.,  70., 133.,  62.,  82.,  62.,  82.,  63.,
        82.,  61.,  83.],
      [197., 205., 197., 205., 192., 212., 187., 223., 185., 226., 203.,
       210., 201., 222., 201., 223., 191., 231., 180., 229., 178., 233.,
       198., 231., 197., 234., 199., 237., 196., 205., 199., 204., 202.,
       203., 198., 205.],
      [ 79.,  96.,  79.,  96.,  78., 101.,  81., 102.,  81., 101.,  74.,
       100.,  72., 105.,  74., 104.,  77., 112.,  89., 124.,  84., 136.,
        66., 104.,  69., 120.,  70., 133.,  77.,  97.,  78.,  97.,  79.,
        96.,  77.,  95.]]


# In[42]:


w = [[0.01      , 0.01      , 0.05392975, 0.05392975, 0.02309585,
        0.02309585, 0.02208957, 0.02208957, 0.88503253, 0.88503253,
        0.82739758, 0.82739758, 0.99957204, 0.99957204, 0.99911058,
        0.99911058, 0.97442281, 0.97442281, 0.99610198, 0.99610198,
        0.95921862, 0.95921862, 0.9804818 , 0.9804818 , 0.99170005,
        0.99170005, 0.99934113, 0.99934113, 0.99110806, 0.99110806,
        0.99213243, 0.99213243, 0.98642004, 0.98642004, 0.95372474,
        0.95372474],
       [0.01      , 0.01      , 0.84239519, 0.84239519, 0.88328242,
        0.88328242, 0.57259166, 0.57259166, 0.91586387, 0.91586387,
        0.33429298, 0.33429298, 0.98654413, 0.98654413, 0.99544227,
        0.99544227, 0.96899122, 0.96899122, 0.90352017, 0.90352017,
        0.93229228, 0.93229228, 0.77726305, 0.77726305, 0.99945682,
        0.99945682, 0.99923909, 0.99923909, 0.99664682, 0.99664682,
        0.99746037, 0.99746037, 0.9928112 , 0.9928112 , 0.9805094 ,
        0.9805094 ],
       [0.01      , 0.01      , 0.65573913, 0.65573913, 0.65785313,
        0.65785313, 0.05247754, 0.05247754, 0.97580051, 0.97580051,
        0.71454698, 0.71454698, 0.99879599, 0.99879599, 0.99795663,
        0.99795663, 0.99727368, 0.99727368, 0.83722848, 0.83722848,
        0.99792814, 0.99792814, 0.41358507, 0.41358507, 0.99930954,
        0.99930954, 0.99859184, 0.99859184, 0.98219943, 0.98219943,
        0.99944448, 0.99944448, 0.98349106, 0.98349106, 0.9978652 ,
        0.9978652 ],
       [0.01      , 0.01      , 0.92994189, 0.92994189, 0.85413504,
        0.85413504, 0.77650368, 0.77650368, 0.86299133, 0.86299133,
        0.70719826, 0.70719826, 0.97970605, 0.97970605, 0.99384677,
        0.99384677, 0.97766554, 0.97766554, 0.85356188, 0.85356188,
        0.9573307 , 0.9573307 , 0.53825706, 0.53825706, 0.99703205,
        0.99703205, 0.99554062, 0.99554062, 0.99941987, 0.99941987,
        0.99322128, 0.99322128, 0.99336541, 0.99336541, 0.90257311,
        0.90257311],
       [0.01      , 0.01      , 0.4314284 , 0.4314284 , 0.29069117,
        0.29069117, 0.28300121, 0.28300121, 0.96108276, 0.96108276,
        0.88265425, 0.88265425, 0.99930131, 0.99930131, 0.99536586,
        0.99536586, 0.91483903, 0.91483903, 0.88975167, 0.88975167,
        0.74696964, 0.74696964, 0.77938181, 0.77938181, 0.96896267,
        0.96896267, 0.97310805, 0.97310805, 0.40984443, 0.40984443,
        0.02084914, 0.02084914, 0.1065582 , 0.1065582 , 0.00209975,
        0.00209975],
       [0.01      , 0.01      , 0.27614492, 0.27614492, 0.22982225,
        0.22982225, 0.17725298, 0.17725298, 0.89965439, 0.89965439,
        0.87060398, 0.87060398, 0.99327397, 0.99327397, 0.99859416,
        0.99859416, 0.96570253, 0.96570253, 0.98782921, 0.98782921,
        0.95244122, 0.95244122, 0.93521702, 0.93521702, 0.99787468,
        0.99787468, 0.99838614, 0.99838614, 0.9957602 , 0.9957602 ,
        0.991247  , 0.991247  , 0.99051338, 0.99051338, 0.97066432,
        0.97066432],
       [0.01      , 0.01      , 0.85109657, 0.85109657, 0.86777365,
        0.86777365, 0.58795506, 0.58795506, 0.87213761, 0.87213761,
        0.21318343, 0.21318343, 0.97085154, 0.97085154, 0.97568703,
        0.97568703, 0.99326152, 0.99326152, 0.98199373, 0.98199373,
        0.99289519, 0.99289519, 0.95938301, 0.95938301, 0.99802434,
        0.99802434, 0.9969238 , 0.9969238 , 0.98797846, 0.98797846,
        0.99453902, 0.99453902, 0.88872468, 0.88872468, 0.94729805,
        0.94729805],
       [0.01      , 0.01      , 0.91492474, 0.91492474, 0.86282384,
        0.86282384, 0.93005276, 0.93005276, 0.68347049, 0.68347049,
        0.88571823, 0.88571823, 0.99465013, 0.99465013, 0.98594177,
        0.98594177, 0.98970699, 0.98970699, 0.97373974, 0.97373974,
        0.96756852, 0.96756852, 0.95821297, 0.95821297, 0.99564147,
        0.99564147, 0.96783251, 0.96783251, 0.95022959, 0.95022959,
        0.98600119, 0.98600119, 0.98527485, 0.98527485, 0.99526024,
        0.99526024],
       [0.01      , 0.01      , 0.23438659, 0.23438659, 0.16997427,
        0.16997427, 0.15563163, 0.15563163, 0.94137239, 0.94137239,
        0.84383857, 0.84383857, 0.99512774, 0.99512774, 0.98882949,
        0.98882949, 0.84153044, 0.84153044, 0.8986761 , 0.8986761 ,
        0.60379243, 0.60379243, 0.57451975, 0.57451975, 0.91134214,
        0.91134214, 0.88198221, 0.88198221, 0.47143546, 0.47143546,
        0.03397208, 0.03397208, 0.10370919, 0.10370919, 0.01610157,
        0.01610157],
       [0.01      , 0.01      , 0.86815989, 0.86815989, 0.61135519,
        0.61135519, 0.94482946, 0.94482946, 0.34982833, 0.34982833,
        0.89559478, 0.89559478, 0.97999465, 0.97999465, 0.99491799,
        0.99491799, 0.97036904, 0.97036904, 0.96908474, 0.96908474,
        0.93257797, 0.93257797, 0.94233394, 0.94233394, 0.99517751,
        0.99517751, 0.99271941, 0.99271941, 0.99941987, 0.99941987,
        0.98600119, 0.98600119, 0.99336541, 0.99336541, 0.99526024,
        0.99526024]]


# In[43]:


d = {'detectionList': [{'score': 0.8020517387810875,
   'keypoints': [{'score': 0.05392974615097046,
     'part': 'nose',
     'position': {'x': 163.23003359948206, 'y': 608.5078749565772}},
    {'score': 0.02309584617614746,
     'part': 'leftEye',
     'position': {'x': 158.01330075281837, 'y': 603.9510282354503}},
    {'score': 0.02208957076072693,
     'part': 'rightEye',
     'position': {'x': 165.339461655206, 'y': 603.2992517795265}},
    {'score': 0.8850325345993042,
     'part': 'leftEar',
     'position': {'x': 148.16113053368272, 'y': 606.3292730585954}},
    {'score': 0.8273975849151611,
     'part': 'rightEar',
     'position': {'x': 169.36971927939283, 'y': 606.1421976651434}},
    {'score': 0.9995720386505127,
     'part': 'leftShoulder',
     'position': {'x': 136.96197301171694, 'y': 635.6809748454565}},
    {'score': 0.9991105794906616,
     'part': 'rightShoulder',
     'position': {'x': 182.95736291792508, 'y': 634.3172801768139}},
    {'score': 0.974422812461853,
     'part': 'leftElbow',
     'position': {'x': 128.66949637909508, 'y': 667.2687765977618}},
    {'score': 0.996101975440979,
     'part': 'rightElbow',
     'position': {'x': 194.04934974556113, 'y': 663.888070537891}},
    {'score': 0.9592186212539673,
     'part': 'leftWrist',
     'position': {'x': 123.35610578033362, 'y': 694.9166127780074}},
    {'score': 0.9804818034172058,
     'part': 'rightWrist',
     'position': {'x': 186.84898668817814, 'y': 692.729613380994}},
    {'score': 0.9917000532150269,
     'part': 'leftHip',
     'position': {'x': 140.39393137456773, 'y': 698.021448376687}},
    {'score': 0.9993411302566528,
     'part': 'rightHip',
     'position': {'x': 172.00248377376727, 'y': 698.5536931780016}},
    {'score': 0.9911080598831177,
     'part': 'leftKnee',
     'position': {'x': 133.2291883832953, 'y': 735.3801846347194}},
    {'score': 0.9921324253082275,
     'part': 'rightKnee',
     'position': {'x': 172.5117406974571, 'y': 746.4394785287682}},
    {'score': 0.9864200353622437,
     'part': 'leftAnkle',
     'position': {'x': 143.4178979959381, 'y': 772.2855890867409}},
    {'score': 0.95372474193573,
     'part': 'rightAnkle',
     'position': {'x': 168.01400112480707, 'y': 790.7553921886291}}]},
  {'score': 0.8869766435202431,
   'keypoints': [{'score': 0.8423951864242554,
     'part': 'nose',
     'position': {'x': 221.9403120337354, 'y': 230.29354392759097}},
    {'score': 0.8832824230194092,
     'part': 'leftEye',
     'position': {'x': 224.0734465023998, 'y': 224.80811612196976}},
    {'score': 0.5725916624069214,
     'part': 'rightEye',
     'position': {'x': 221.08537565367052, 'y': 224.52157953169768}},
    {'score': 0.9158638715744019,
     'part': 'leftEar',
     'position': {'x': 230.91303778007236, 'y': 226.76105242006906}},
    {'score': 0.3342929780483246,
     'part': 'rightEar',
     'position': {'x': 224.24276945028413, 'y': 229.6178772395777}},
    {'score': 0.986544132232666,
     'part': 'leftShoulder',
     'position': {'x': 237.28917203324565, 'y': 245.03261972176136}},
    {'score': 0.995442271232605,
     'part': 'rightShoulder',
     'position': {'x': 225.79366391071218, 'y': 245.23665284201422}},
    {'score': 0.968991219997406,
     'part': 'leftElbow',
     'position': {'x': 233.9817853497432, 'y': 259.16278851011594}},
    {'score': 0.9035201668739319,
     'part': 'rightElbow',
     'position': {'x': 219.50578075223203, 'y': 263.8014277420176}},
    {'score': 0.9322922825813293,
     'part': 'leftWrist',
     'position': {'x': 215.41493770513642, 'y': 245.94165377806956}},
    {'score': 0.7772630453109741,
     'part': 'rightWrist',
     'position': {'x': 214.03382193461786, 'y': 266.8696041239403}},
    {'score': 0.9994568228721619,
     'part': 'leftHip',
     'position': {'x': 236.14877241648986, 'y': 308.5849463456218}},
    {'score': 0.9992390871047974,
     'part': 'rightHip',
     'position': {'x': 223.6569348203109, 'y': 309.22001482419785}},
    {'score': 0.9966468214988708,
     'part': 'leftKnee',
     'position': {'x': 254.1238371388296, 'y': 363.23743641603755}},
    {'score': 0.9974603652954102,
     'part': 'rightKnee',
     'position': {'x': 240.2136203385471, 'y': 364.3688832378883}},
    {'score': 0.9928112030029297,
     'part': 'leftAnkle',
     'position': {'x': 253.1930084049925, 'y': 411.6310033012194}},
    {'score': 0.9805094003677368,
     'part': 'rightAnkle',
     'position': {'x': 240.13132280892648, 'y': 415.4867563198096}}]},
  {'score': 0.8388286373194527,
   'keypoints': [{'score': 0.6557391285896301,
     'part': 'nose',
     'position': {'x': 598.8978732676989, 'y': 506.7789182943942}},
    {'score': 0.6578531265258789,
     'part': 'leftEye',
     'position': {'x': 599.3679572509022, 'y': 502.0857118527232}},
    {'score': 0.05247753858566284,
     'part': 'rightEye',
     'position': {'x': 602.1768501653207, 'y': 500.5353890771056}},
    {'score': 0.9758005142211914,
     'part': 'leftEar',
     'position': {'x': 608.2421172602793, 'y': 493.849841617009}},
    {'score': 0.7145469784736633,
     'part': 'rightEar',
     'position': {'x': 620.306624519691, 'y': 491.54931673516023}},
    {'score': 0.9987959861755371,
     'part': 'leftShoulder',
     'position': {'x': 608.1783342504323, 'y': 520.942217609498}},
    {'score': 0.9979566335678101,
     'part': 'rightShoulder',
     'position': {'x': 636.3336160415121, 'y': 512.8697870707388}},
    {'score': 0.9972736835479736,
     'part': 'leftElbow',
     'position': {'x': 600.9189186810554, 'y': 562.9568884666086}},
    {'score': 0.8372284770011902,
     'part': 'rightElbow',
     'position': {'x': 650.6548077801193, 'y': 546.6246073936052}},
    {'score': 0.9979281425476074,
     'part': 'leftWrist',
     'position': {'x': 593.4696491595065, 'y': 591.3029705998803}},
    {'score': 0.4135850667953491,
     'part': 'rightWrist',
     'position': {'x': 636.3809471121442, 'y': 570.0108607505388}},
    {'score': 0.9993095397949219,
     'part': 'leftHip',
     'position': {'x': 628.4978959819351, 'y': 581.5615101521721}},
    {'score': 0.9985918402671814,
     'part': 'rightHip',
     'position': {'x': 650.4556511243185, 'y': 577.9223464762524}},
    {'score': 0.9821994304656982,
     'part': 'leftKnee',
     'position': {'x': 609.7089073131147, 'y': 620.8149327818608}},
    {'score': 0.9994444847106934,
     'part': 'rightKnee',
     'position': {'x': 649.6841825045897, 'y': 625.2520755473621}},
    {'score': 0.983491063117981,
     'part': 'leftAnkle',
     'position': {'x': 600.7801400159628, 'y': 666.844545389172}},
    {'score': 0.9978652000427246,
     'part': 'rightAnkle',
     'position': {'x': 651.6086549651757, 'y': 677.071969976045}}]},
  {'score': 0.9007229734869564,
   'keypoints': [{'score': 0.9299418926239014,
     'part': 'nose',
     'position': {'x': 531.9455111794704, 'y': 333.43611599708964}},
    {'score': 0.8541350364685059,
     'part': 'leftEye',
     'position': {'x': 532.8446314879571, 'y': 330.1159189214326}},
    {'score': 0.7765036821365356,
     'part': 'rightEye',
     'position': {'x': 523.5307017328141, 'y': 327.3713186270649}},
    {'score': 0.8629913330078125,
     'part': 'leftEar',
     'position': {'x': 562.8576965046286, 'y': 322.71127055877207}},
    {'score': 0.7071982622146606,
     'part': 'rightEar',
     'position': {'x': 552.8912220108375, 'y': 331.64173144236383}},
    {'score': 0.9797060489654541,
     'part': 'leftShoulder',
     'position': {'x': 549.9477634358495, 'y': 341.3249667645327}},
    {'score': 0.9938467741012573,
     'part': 'rightShoulder',
     'position': {'x': 571.3982272594609, 'y': 343.6911496463227}},
    {'score': 0.9776655435562134,
     'part': 'leftElbow',
     'position': {'x': 523.0600637273395, 'y': 344.2608908036759}},
    {'score': 0.8535618782043457,
     'part': 'rightElbow',
     'position': {'x': 583.010996217808, 'y': 368.85026646486807}},
    {'score': 0.9573307037353516,
     'part': 'leftWrist',
     'position': {'x': 513.1748874357131, 'y': 344.72253221779596}},
    {'score': 0.5382570624351501,
     'part': 'rightWrist',
     'position': {'x': 576.7331447110193, 'y': 377.7758907371005}},
    {'score': 0.9970320463180542,
     'part': 'leftHip',
     'position': {'x': 565.2119223883982, 'y': 392.1579105287945}},
    {'score': 0.9955406188964844,
     'part': 'rightHip',
     'position': {'x': 583.4175109561352, 'y': 395.2229002121209}},
    {'score': 0.9994198679924011,
     'part': 'leftKnee',
     'position': {'x': 533.1899578312363, 'y': 419.68256975666486}},
    {'score': 0.9932212829589844,
     'part': 'rightKnee',
     'position': {'x': 583.2705294509952, 'y': 420.5354328114197}},
    {'score': 0.9933654069900513,
     'part': 'leftAnkle',
     'position': {'x': 503.7792750547888, 'y': 461.1810554854791}},
    {'score': 0.9025731086730957,
     'part': 'rightAnkle',
     'position': {'x': 606.5030665736966, 'y': 440.4511164263588}}]},
  {'score': 0.626817021299811,
   'keypoints': [{'score': 0.4314284026622772,
     'part': 'nose',
     'position': {'x': 1299.60643629367, 'y': 709.7628141185027}},
    {'score': 0.29069116711616516,
     'part': 'leftEye',
     'position': {'x': 1319.5221731921706, 'y': 693.9172007471477}},
    {'score': 0.28300121426582336,
     'part': 'rightEye',
     'position': {'x': 1298.8519953788443, 'y': 708.2381341721817}},
    {'score': 0.9610827565193176,
     'part': 'leftEar',
     'position': {'x': 1317.9839012203145, 'y': 694.2287636249482}},
    {'score': 0.8826542496681213,
     'part': 'rightEar',
     'position': {'x': 1288.703150147356, 'y': 707.4435710807813}},
    {'score': 0.9993013143539429,
     'part': 'leftShoulder',
     'position': {'x': 1314.3488541220904, 'y': 726.7000980129275}},
    {'score': 0.9953658580780029,
     'part': 'rightShoulder',
     'position': {'x': 1357.0013501813796, 'y': 719.4906353743899}},
    {'score': 0.9148390293121338,
     'part': 'leftElbow',
     'position': {'x': 1303.4152672192577, 'y': 759.6088940215069}},
    {'score': 0.889751672744751,
     'part': 'rightElbow',
     'position': {'x': 1359.4898338389307, 'y': 752.5164535776994}},
    {'score': 0.7469696402549744,
     'part': 'leftWrist',
     'position': {'x': 1292.2674302501177, 'y': 778.7170160432415}},
    {'score': 0.7793818116188049,
     'part': 'rightWrist',
     'position': {'x': 1362.9049662757902, 'y': 769.8108043521901}},
    {'score': 0.9689626693725586,
     'part': 'leftHip',
     'position': {'x': 1316.764335560888, 'y': 782.1414413105057}},
    {'score': 0.9731080532073975,
     'part': 'rightHip',
     'position': {'x': 1346.9621322154999, 'y': 782.1678903867302}},
    {'score': 0.40984442830085754,
     'part': 'leftKnee',
     'position': {'x': 1265.6747587075395, 'y': 769.652980101997}},
    {'score': 0.020849138498306274,
     'part': 'rightKnee',
     'position': {'x': 1358.6909180687608, 'y': 794.7339723329196}},
    {'score': 0.10655820369720459,
     'part': 'leftAnkle',
     'position': {'x': 1276.9909001629005, 'y': 804.2203911627599}},
    {'score': 0.002099752426147461,
     'part': 'rightAnkle',
     'position': {'x': 1375.4585873571675, 'y': 805.3495580418684}}]},
  {'score': 0.8371166096014135,
   'keypoints': [{'score': 0.27614492177963257,
     'part': 'nose',
     'position': {'x': 383.60851684313144, 'y': 556.2119824361553}},
    {'score': 0.22982224822044373,
     'part': 'leftEye',
     'position': {'x': 384.0054959786519, 'y': 551.9657474564101}},
    {'score': 0.17725297808647156,
     'part': 'rightEye',
     'position': {'x': 384.15528377075765, 'y': 551.6783302164987}},
    {'score': 0.8996543884277344,
     'part': 'leftEar',
     'position': {'x': 377.5022451422188, 'y': 553.9159808910905}},
    {'score': 0.8706039786338806,
     'part': 'rightEar',
     'position': {'x': 392.6135844005628, 'y': 553.6902578575359}},
    {'score': 0.9932739734649658,
     'part': 'leftShoulder',
     'position': {'x': 366.0288457263275, 'y': 579.157855280144}},
    {'score': 0.9985941648483276,
     'part': 'rightShoulder',
     'position': {'x': 410.40195340208345, 'y': 576.1001326967566}},
    {'score': 0.9657025337219238,
     'part': 'leftElbow',
     'position': {'x': 355.1080439528276, 'y': 605.9863835934749}},
    {'score': 0.9878292083740234,
     'part': 'rightElbow',
     'position': {'x': 431.5773532131638, 'y': 602.2752772815512}},
    {'score': 0.9524412155151367,
     'part': 'leftWrist',
     'position': {'x': 348.7737713223986, 'y': 630.946604994802}},
    {'score': 0.935217022895813,
     'part': 'rightWrist',
     'position': {'x': 443.84258352415395, 'y': 636.8562104912729}},
    {'score': 0.9978746771812439,
     'part': 'leftHip',
     'position': {'x': 386.59041430084, 'y': 645.7433146087508}},
    {'score': 0.9983861446380615,
     'part': 'rightHip',
     'position': {'x': 404.0685349618005, 'y': 641.6474805770755}},
    {'score': 0.9957602024078369,
     'part': 'leftKnee',
     'position': {'x': 395.71989334299326, 'y': 699.45121700429}},
    {'score': 0.9912469983100891,
     'part': 'rightKnee',
     'position': {'x': 387.9189510095432, 'y': 696.8044327801186}},
    {'score': 0.9905133843421936,
     'part': 'leftAnkle',
     'position': {'x': 408.0154145172473, 'y': 749.6358494775134}},
    {'score': 0.9706643223762512,
     'part': 'rightAnkle',
     'position': {'x': 362.82232833176516, 'y': 745.0365740662018}}]},
  {'score': 0.8870415705091813,
   'keypoints': [{'score': 0.8510965704917908,
     'part': 'nose',
     'position': {'x': 734.1031903827682, 'y': 262.93655321792255}},
    {'score': 0.8677736520767212,
     'part': 'leftEye',
     'position': {'x': 736.7470037946094, 'y': 259.5912471850575}},
    {'score': 0.5879550576210022,
     'part': 'rightEye',
     'position': {'x': 732.9761542702436, 'y': 260.35934771202255}},
    {'score': 0.8721376061439514,
     'part': 'leftEar',
     'position': {'x': 743.2469600684634, 'y': 254.3228651423595}},
    {'score': 0.2131834328174591,
     'part': 'rightEar',
     'position': {'x': 737.0599611475227, 'y': 258.8059565859816}},
    {'score': 0.9708515405654907,
     'part': 'leftShoulder',
     'position': {'x': 755.6888343359201, 'y': 258.34611127942645}},
    {'score': 0.9756870269775391,
     'part': 'rightShoulder',
     'position': {'x': 739.3911668405997, 'y': 259.8085326900515}},
    {'score': 0.9932615160942078,
     'part': 'leftElbow',
     'position': {'x': 769.9314288396514, 'y': 275.286505294218}},
    {'score': 0.9819937348365784,
     'part': 'rightElbow',
     'position': {'x': 740.8031947925297, 'y': 280.1308789013577}},
    {'score': 0.9928951859474182,
     'part': 'leftWrist',
     'position': {'x': 760.8045877088769, 'y': 293.5800585971961}},
    {'score': 0.9593830108642578,
     'part': 'rightWrist',
     'position': {'x': 734.7522400845303, 'y': 297.00639048833364}},
    {'score': 0.9980243444442749,
     'part': 'leftHip',
     'position': {'x': 773.6855437603783, 'y': 288.60317832923636}},
    {'score': 0.9969238042831421,
     'part': 'rightHip',
     'position': {'x': 757.1237041825212, 'y': 290.4120283374753}},
    {'score': 0.987978458404541,
     'part': 'leftKnee',
     'position': {'x': 764.3031805133105, 'y': 323.1905232252653}},
    {'score': 0.9945390224456787,
     'part': 'rightKnee',
     'position': {'x': 751.3878181132485, 'y': 323.40401114624126}},
    {'score': 0.888724684715271,
     'part': 'leftAnkle',
     'position': {'x': 783.4895236090327, 'y': 366.3310089235289}},
    {'score': 0.9472980499267578,
     'part': 'rightAnkle',
     'position': {'x': 762.8600707813149, 'y': 366.584824854621}}]},
  {'score': 0.9427676481359145,
   'keypoints': [{'score': 0.9149247407913208,
     'part': 'nose',
     'position': {'x': 375.5513109428606, 'y': 281.9225449289358}},
    {'score': 0.8628238439559937,
     'part': 'leftEye',
     'position': {'x': 376.8318416652608, 'y': 278.70072825281414}},
    {'score': 0.9300527572631836,
     'part': 'rightEye',
     'position': {'x': 372.49454829040985, 'y': 278.7407003482045}},
    {'score': 0.6834704875946045,
     'part': 'leftEar',
     'position': {'x': 378.8217731182941, 'y': 280.2539174907955}},
    {'score': 0.8857182264328003,
     'part': 'rightEar',
     'position': {'x': 368.10370174865153, 'y': 280.8183486440979}},
    {'score': 0.99465012550354,
     'part': 'leftShoulder',
     'position': {'x': 378.9811210560888, 'y': 291.3590077767132}},
    {'score': 0.9859417676925659,
     'part': 'rightShoulder',
     'position': {'x': 359.0225165756454, 'y': 298.4464395256968}},
    {'score': 0.9897069931030273,
     'part': 'leftElbow',
     'position': {'x': 397.8577783134546, 'y': 314.1849269156233}},
    {'score': 0.973739743232727,
     'part': 'rightElbow',
     'position': {'x': 357.2110724859916, 'y': 315.7311405730619}},
    {'score': 0.9675685167312622,
     'part': 'leftWrist',
     'position': {'x': 405.9057171693009, 'y': 320.0255415187551}},
    {'score': 0.9582129716873169,
     'part': 'rightWrist',
     'position': {'x': 392.3763132738263, 'y': 321.54360919601993}},
    {'score': 0.9956414699554443,
     'part': 'leftHip',
     'position': {'x': 387.91558512855556, 'y': 342.2755210147573}},
    {'score': 0.9678325057029724,
     'part': 'rightHip',
     'position': {'x': 375.8081285319525, 'y': 343.5003375833534}},
    {'score': 0.9502295851707458,
     'part': 'leftKnee',
     'position': {'x': 404.43661057279354, 'y': 381.03110596377473}},
    {'score': 0.986001193523407,
     'part': 'rightKnee',
     'position': {'x': 417.8707630803969, 'y': 407.60771583271185}},
    {'score': 0.9852748513221741,
     'part': 'leftAnkle',
     'position': {'x': 402.1597112385968, 'y': 431.3819410416654}},
    {'score': 0.9952602386474609,
     'part': 'rightAnkle',
     'position': {'x': 419.21571172042735, 'y': 452.4147852876157}}]},
  {'score': 0.5686012979815988,
   'keypoints': [{'score': 0.2343865931034088,
     'part': 'nose',
     'position': {'x': 1185.5387787229558, 'y': 695.1018555151941}},
    {'score': 0.16997426748275757,
     'part': 'leftEye',
     'position': {'x': 1177.7975290277031, 'y': 694.9376848442303}},
    {'score': 0.1556316316127777,
     'part': 'rightEye',
     'position': {'x': 1191.7794282213133, 'y': 692.082064391422}},
    {'score': 0.9413723945617676,
     'part': 'leftEar',
     'position': {'x': 1215.353528119205, 'y': 687.2909361493856}},
    {'score': 0.8438385725021362,
     'part': 'rightEar',
     'position': {'x': 1191.4044221135114, 'y': 694.1368732303639}},
    {'score': 0.9951277375221252,
     'part': 'leftShoulder',
     'position': {'x': 1154.874260052313, 'y': 717.2802578621968}},
    {'score': 0.988829493522644,
     'part': 'rightShoulder',
     'position': {'x': 1218.102323098129, 'y': 711.6584733817565}},
    {'score': 0.841530442237854,
     'part': 'leftElbow',
     'position': {'x': 1123.3756569387315, 'y': 756.7272163032452}},
    {'score': 0.8986760973930359,
     'part': 'rightElbow',
     'position': {'x': 1204.7321940825673, 'y': 751.3359444765325}},
    {'score': 0.6037924289703369,
     'part': 'leftWrist',
     'position': {'x': 1109.2500073389615, 'y': 765.0169149428554}},
    {'score': 0.5745197534561157,
     'part': 'rightWrist',
     'position': {'x': 1209.212806148029, 'y': 756.5251189588054}},
    {'score': 0.9113421440124512,
     'part': 'leftHip',
     'position': {'x': 1145.9274648530652, 'y': 783.137546554182}},
    {'score': 0.8819822072982788,
     'part': 'rightHip',
     'position': {'x': 1186.1627591343854, 'y': 781.9849693622291}},
    {'score': 0.47143545746803284,
     'part': 'leftKnee',
     'position': {'x': 1083.1028352587412, 'y': 775.8372429908871}},
    {'score': 0.033972084522247314,
     'part': 'rightKnee',
     'position': {'x': 1180.828851012255, 'y': 794.4969052789116}},
    {'score': 0.10370919108390808,
     'part': 'leftAnkle',
     'position': {'x': 1066.3653701860806, 'y': 788.5057413573901}},
    {'score': 0.016101568937301636,
     'part': 'rightAnkle',
     'position': {'x': 1191.6710920798198, 'y': 802.3288549409995}}]},
  {'score': 0.4474380594842574,
   'keypoints': [{'score': 0.8681598901748657,
     'part': 'nose',
     'position': {'x': 476.13915352517745, 'y': 327.07329704278055}},
    {'score': 0.611355185508728,
     'part': 'leftEye',
     'position': {'x': 467.015848804935, 'y': 328.1423830795949}},
    {'score': 0.9448294639587402,
     'part': 'rightEye',
     'position': {'x': 470.1782067799836, 'y': 328.0582653753059}},
    {'score': 0.3498283326625824,
     'part': 'leftEar',
     'position': {'x': 476.4363677742776, 'y': 326.82552992161266}},
    {'score': 0.8955947756767273,
     'part': 'rightEar',
     'position': {'x': 466.030011882273, 'y': 324.36412984329036}},
    {'score': 0.9799946546554565,
     'part': 'leftShoulder',
     'position': {'x': 471.7577458008398, 'y': 342.11081690680834}},
    {'score': 0.9949179887771606,
     'part': 'rightShoulder',
     'position': {'x': 443.7613709910532, 'y': 340.56933282524915}},
    {'score': 0.9703690409660339,
     'part': 'leftElbow',
     'position': {'x': 485.0312327195643, 'y': 345.7685864818778}},
    {'score': 0.9690847396850586,
     'part': 'rightElbow',
     'position': {'x': 433.90594099880605, 'y': 356.40191332161737}},
    {'score': 0.9325779676437378,
     'part': 'leftWrist',
     'position': {'x': 485.3742586093002, 'y': 343.3309474098827}},
    {'score': 0.9423339366912842,
     'part': 'rightWrist',
     'position': {'x': 445.4298216716181, 'y': 352.55831628454206}},
    {'score': 0.9951775074005127,
     'part': 'leftHip',
     'position': {'x': 462.27811120244, 'y': 380.5048725534766}},
    {'score': 0.9927194118499756,
     'part': 'rightHip',
     'position': {'x': 396.7217482538259, 'y': 354.2276014361266}},
    {'score': 0.9994198679924011,
     'part': 'leftKnee',
     'position': {'x': 533.1899578312363, 'y': 419.68256975666486}},
    {'score': 0.986001193523407,
     'part': 'rightKnee',
     'position': {'x': 417.8707630803969, 'y': 407.60771583271185}},
    {'score': 0.9933654069900513,
     'part': 'leftAnkle',
     'position': {'x': 503.7792750547888, 'y': 461.1810554854791}},
    {'score': 0.9952602386474609,
     'part': 'rightAnkle',
     'position': {'x': 419.21571172042735, 'y': 452.4147852876157}}]}]}


# #Cosine Test

# In[44]:


#c = getPred(d)


# In[45]:


#getMeanRunTime()


# In[46]:


#np.mean(TIME_TRACKER)


# In[47]:


#ARGS['X'].shape


# #Weighted Distacne Test
# 

# In[48]:


# w = np.array(w)
# d = np.array(d)


# In[49]:


# sp = findMatch(d[1])


# In[50]:


# indexs = getSimilarPosesClassesIndex(sp)


# In[51]:


# getSimilarPosesClasses(indexs)


# In[52]:


# c = fit(d,inputImgH=825, inputImgW=1461)


# In[53]:


# c


# In[54]:


# for i in v11:
#     center = tuple(np.array(i,dtype=np.int))
#     cv2.circle(empty_image, center,2, (255,255,255),
#                thickness=3, lineType=8, shift=0)


# In[55]:


# cv2.imwrite('./dataandmodles/data/v22.png', empty_image)

