#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# #Importing ActionClassification



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

import import_ipynb
import ActionClassificationCosine


# #Tracker



prevFrameMidPointsToActionDict = None
prevFrameMidPoints = None
threshold = 30
previousFramePose = {}




def getMidPoint(bbox):
    '''
    Helper function of conertBboxsToMidPoint()
    '''
    p1, p2 = bbox[0] , bbox[1]
    rowMid = (p1[0] + p2[0])/2
    widthMid = (p1[1] + p2[1])/2
    return (rowMid , widthMid)

def convertBboxsToMidPoint(bboxs):
    '''
     Parameter
     ----------
     bboxs : A 2D list containing the two coordinates of box, p1(left,top) and p2(right,bottom)
     [ [p1 , p2] , ... ] (from getBoxCoord)
     
     Return
     -------------
     midPointList : A List containig tuple of mid-point coord for each box
     Example : [(mX,mY), ....]
    '''
    midPointList =  list(map(getMidPoint , bboxs))
    return midPointList




def getDistanceMidToMid(midPoint1, midPoint2):
    '''
    Helper function of getDistacneMidToMids()
    '''
    dist = np.sqrt((midPoint1[0] - midPoint2[0])**2 + (midPoint1[1] - midPoint2[1])**2)
    return dist

def getDistanceMidToMids(midPoint1,midPoints):
    '''
    This function compares the distacne of 1 mid point to list of midpoints
    
    Parameter 
    -----------
    midPoint1 : A list containing midPoint coordintes [mx, my]
    
    Return
    -------
    distances : Distance of midPoint1 to all other midpoints in midPoints
    Example : [3 ,4 ,5 ....]
    
    '''
    distances = [getDistanceMidToMid(midPoint1,midPoint2) for midPoint2 in midPoints]
    return distances




def getAllMidPointsDistance(midPointList):
    '''
    This function , for each midpoint in midPointList, it find the distance to other midpoints in the 
    previous frame
    
    Parameter 
    ---------
    midPointList  :  A list contianing midPoints
    Example : [(mx,my)....(mx,my)]
    
    Return 
    ----------
    midPointsDistances : A 2D list containing distance to all the other midpoints 
    Example : [ [1,2,3] , [4,5,9] , ... ]
    
    '''
    global prevFrameMidPoints
    midPointsDistances = [getDistanceMidToMids(midPoint1 , prevFrameMidPoints) for midPoint1 in midPointList]
    return midPointsDistances




def getClosestMidPoint(midPointDistance):
    '''
    Helper function of track()
    '''
    global prevFrameMidPoints
    closestMidPoint = prevFrameMidPoints[np.argmin(midPointDistance)]
    distanceToClosestMidPoint = np.min(midPointDistance)
    return closestMidPoint , distanceToClosestMidPoint


def track(bboxs):
    '''
    This function will return action for each detectect pose/bbox.
    It will do this by finding the closest bounding box in the last frame.
    
    Parameter
    ----------
    bboxs  : A list containing bboxs. Where each box contain 2 points of the box
    Example : [[p1,p2] , ....[]..]
    
    Return 
    -----------
    bboxsAction : A dict containing estimated action class for each pose/box in bboxs 
                None is returned if no closest midpoint is found
    Example : [(midpoint): 'Shoot',(midPoint) : None .....] (order is kept)
    '''
    
    global prevFrameMidPointsToActionDict
    bboxsAction = {}
    
    midPointsList = convertBboxsToMidPoint(bboxs)
    midPointsDistances = getAllMidPointsDistance(midPointsList)
    for midPointDistance in midPointsDistances:
        closestMidPoint , distanceToClosestMidPoint = getClosestMidPoint(midPointDistance)
        if distanceToClosestMidPoint < threshold:
            action =  prevFrameMidPointsToActionDict[closestMidPoint]
        else:
            action = None
        bboxsAction[closestMidPoint] = action
    
    
    return bboxsAction









def getBoxCoord(bboxes,imgh,imgw):
    '''
    Helper function of setPreviousFramePoses
    This function finds the left_top and right_bottom coordinates of larger bounding box
    Parameter
    ---------
    bboxes : The posenet box output.
             Example: {'bbox':[[box1] , [box2]....]}
             (box1 and box2 should be relative to the whole image)
    imgh : The image(main/big/not ROI) height
    
    imgw : The image(main/big/not ROI) width 
    
    Return 
    -------
    boxes : A list with list of coords. 
            Example : [[p1,p2] , [box2],.......]]. Where p1 and p2 is a list of points
    '''
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

def setTracker(bboxs,classes):
    '''
    This function sets the list of midpoints and the dict; that contains midPoints and classes.
    
    Parameter
    ----------
    bboxs  : A list containing bboxs. Where each box contain 2 points of the box
    Example : [[p1,p2] , ....[]..]
    
    classes :  A list of action classes for the bboxes. It should not contain any None Value
    Example : ['Shoot',.......]
    '''
    global prevFrameMidPoints , prevFrameMidPointsToActionDict
    #Setting midpoints list 
    prevFrameMidPoints = convertBboxsToMidPoint(bboxs)
    #Setting dict
    prevFrameMidPointsToActionDict = {midPoint : classes[i][0] for i,midPoint in enumerate(prevFrameMidPoints)}
    
    
def setPreviousFramePoses(bboxs,poses,imgh,imgw):
    '''
    This function saves this frame for history. 
    Parameter
    ----------
     bboxs  : A list containing bboxs. Where each box contain 2 points of the box
             Example : [[p1,p2] , ....[]..]
    
    poses : Processed Posenet prediction
            Example : [[x_1,y_1......x_17,y_17] , [..]]. 
            (See:ActionClassificationCosine.getStackedPoses)
            
    '''
    global previousFramePose
    previousFramePose = {m:poses[i] for i,m in enumerate(convertBboxsToMidPoint(bboxs))}

def setPrevFrameWrapper(bboxes,poses,actionClasses,imgh, imgw):
    '''
    This function will set this frames: 
    poses,midpoints and a dict(mapping from midpoint to classes)
    
    Parameter
    -----------
    bboxs : The bboxes from posenet. 
             Example : {'bbox':[[{x:..y:},....] , [box2],.......]}
    
    poses : Posenet prediction.
            This contains keypoints that are relative the the main big image

    
    actionClasses :  A list of action classes for the bboxes. It should not contain any None Value
                    Example : ['Shoot',.......]
        
    imgh : Height of the main image
    
    imgw : Wdith of the main image
    
    '''
    _, poses = ActionClassificationCosine.getStackedPoses(poses,
                                                          bboxes,
                                                          imgh,
                                                          imgw,
                                                          cosine=False)
    bboxs = ActionClassificationCosine.getBoxCoord(bboxes,imgh,imgw)
    setTracker(bboxs,actionClasses)
    setPreviousFramePoses(bboxs,poses,imgh,imgw)
    


# In[9]:


def isPrevAndCurrentPoseSame(currentPoses,closestPose):
    '''
    Helper function of updatePoseesClasses
    This function finds the distacne between the  current pose and the pose of the last frame
    
    Parameter
    ----------
    currentPose : 1D list containing pose and weights [0_x.......17_w_y]
    
    Return 
    ---------
    bool : True, (if distance is less than x) then the 2 poses are same 
           False, (distance more than x) then the 2 posese are very different
    '''
    distance = ActionClassificationCosine.weightedDistanceMatching(currentPoses,closestPose)
    if distance < 10:
        return True 
    else:
        return False

def updatePosesClasses(poses,posesClasses,bboxs,imgh,imgw):
    '''
    This function will update the action class based on the previous frame 
    
    Parameter
    ----------
    poses : Posenet prediction.
            This contains keypoints that are relative the the main big image
    
    poesesClasses : A list of action classes for the bboxes. It should not contain any None Value
                    Example : ['Shoot',.......]
                
    bboxes : The bboxes from posenet. 
             Example : {'bbox':[[{x:..y:},....] , [box2],.......]}
    
    Return
    -------
    poesesClasses : a 2D list that contains poses class for each human 
                    Example [['walk'] , ['shoot'] ,......]  
    
    '''
    global previousFramePose
    stackedPoses,_ = ActionClassificationCosine.getStackedPoses(poses,
                                                            bboxs,
                                                            imgh,
                                                            imgw,
                                                          cosine=False)
    bboxs = ActionClassificationCosine.getBoxCoord(bboxs,imgh,imgw)
    midPointToAction = track(bboxs)
    

    count = 0
    for prevMidpoint,action in midPointToAction.items():
        if action != None:
            currentPose = stackedPoses[count]
            closestPose = previousFramePose[prevMidpoint]
            if isPrevAndCurrentPoseSame(currentPose,closestPose):
                posesClasses[count] = action
        count += 1   
    return posesClasses







