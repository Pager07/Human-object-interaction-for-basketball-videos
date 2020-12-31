#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
import time
import math
from scipy.spatial import distance
from sklearn.cluster import KMeans
import import_ipynb
import imutils 
import copy
import GazeModule


# File info:
# 

# #Action Classification Similarity Function

# In[2]:


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



import ActionClassificationCosine
import Tracker


# #Loading and initialising yolov3 from opencv

# In[3]:


args = {
          "confThreshold": 0.88,
          "nmsThreshold":0.4,
          "inpWidth":416,
          "inpHeight":416,
          "bboxAreaToImageArea":0.15,
          "knnLearner" : None,
          "team0":'MIL',
          "team1":'CAVS',
          "colorBoundaries":[
                        ([ 56, -7 ,186], [196, 133, 266]), #white/team0/HSV/ FOR BASIC
                        ([160-70,170-80, 60-30], [160+70,170+80, 60+30]) #red/team1/HSV
                        ],
          "team0HSV":[135,50,215], #white/team0/HSV/For KMEANS
          "team1HSV":[170,50,70], #red/team1/HSV
          "yolo": False,
          "mask-rcnn" : False,
          "net" : None,
          "output_layer_names" : None
        }



# In[4]:


with open("./dataandmodles/models/teamDetection/coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


# In[5]:


# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object
def getOutputsNames(net):
    if args['yolo']:
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    else:
        return ["detection_out_final", "detection_masks"]


# In[6]:


def loadYolo():
    rootDir = './dataandmodles/models/teamDetection'
    args['net'] = cv2.dnn.readNet(rootDir+"/yolov3.weights",rootDir+"/yolov3.cfg")
    net = args['net']
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    # change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    args['output_layer_names'] = getOutputsNames(args['net'])


# #Loading Mask-RCNN

# In[7]:


def loadMaskRcnn():
    #Load colors 
    RED_COLOR = np.array([255, 0, 0]) 
    BLACK_COLOR = np.array([255, 255, 255]) 
    # Load classes
    classes_file = "./dataandmodles/models/teamDetection/Mask_RCNN/mscoco_labels.names"
    text_graph = './dataandmodles/models/teamDetection/Mask_RCNN/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
    model_weights = './dataandmodles/models/teamDetection/Mask_RCNN/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'

    # load our Mask R-CNN trained on the COCO dataset (90 classes) from disk
    print("[INFO] loading Mask R-CNN from disk...")
    args['net'] = cv2.dnn.readNetFromTensorflow(model_weights, text_graph)
    args['output_layer_names'] = getOutputsNames(args['net'])


# #Opencv setup

# In[8]:


# dummy on trackbar callback function
def on_trackbar(val):
    return


# In[9]:


def openDisplayWindow():
    windowName = 'YOLOv3 Team detection'
    cv2.namedWindow(windowName , cv2.WINDOW_NORMAL) 
    trackbarName = 'reporting confidence > (x 0.01)'
    cv2.createTrackbar(trackbarName,windowName,70,100, on_trackbar)
    return windowName , trackbarName


# #HelperFunction:Drawing Prediction

# In[10]:


def drawPred(image,actionClass,team, confidence, left, top, right, bottom, colour,roi=None,mask=None):
    
    if args['yolo'] == False:
        blended = ((0.4 * np.array(colour)) + (0.6 * roi)).astype("uint8")
        image[top:bottom, left:right][mask] = blended
        
    
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 1)

    # construct label
    label = '%s:%.2f' % (actionClass, confidence)
    
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    labelSize = (116,12)
    top = max(top, labelSize[1])
    pt1 = (left, top - round(1.5*labelSize[1])) 
    pt2 = (left + round(1.5*labelSize[0]), top + baseLine)
    cv2.rectangle(image,pt1 , pt2 , colour, cv2.FILLED)
    
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


# #HelperFunction:Post Procces

# In[11]:


def postprocess(image, results, threshold_confidence, threshold_nms, yolo=True, raw_masks=None):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    masks = []
    if yolo:
        for result in results:
            for detection in result:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > threshold_confidence:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        for i in range(0, results.shape[2]):
            classId = int(results[0, 0, i, 1])
            confidence = results[0, 0, i, 2]
            if confidence > threshold_confidence:
                box = results[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                (left, top, right, bottom) = box.astype("int")
                boxW = right - left
                boxH = bottom - top
                mask = raw_masks[i, classId]
                mask = cv2.resize(mask, (boxW, boxH), interpolation = cv2.INTER_NEAREST)
                mask = (mask > 0.3)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([int(left), int(top), int(boxW), int(boxH)])
                masks.append(mask)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []
    masks_nms = []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])
        if yolo == False:
            masks_nms.append(masks[i])
            
    # return post processed lists of classIds, confidences and bounding boxes
    if yolo:
        return (classIds_nms, confidences_nms, boxes_nms)
    else:
        return (classIds_nms, confidences_nms, boxes_nms , masks_nms)


# #Filtering the preditction

# In[12]:


def check_bbox_size(bboxW,bboxH,imgW,imgH):
    bboxToImg = (bboxW*bboxH) / (imgW * imgH)
    return bboxToImg <= args['bboxAreaToImageArea']


# In[13]:


def check_label(label_id):
    return label_id == 0


# #ROI color detection

# In[14]:


def getRoi(frame, left,top,right,bottom,mask):
    '''
    Helper function for detect_teams
    Returns ROI(region of interest) 
    '''
    if args['yolo']:
        roi = frame[top:bottom , left:right,:]
    else:
        roi = frame[top:bottom , left:right,:][mask]
        
    return roi


# In[15]:


def countNonBalckPix(roiMasked):
    '''
    Helper function for findColorRatio
    Returns the number of non black pixels in the roi
    '''
    return roiMasked.any(axis = -1).sum()


# In[16]:


def getColorRatio(roi,show=False):
    '''
    Helper function for detect teams
    Returns a list, that contains percentage of the pixel that have the team %colors
    Example: [0.9 , 0.1]. 90% of the pixels are of team 1
    '''
    ratioList = []
    
    for teamColorLower,teamColorUpper in args['colorBoundaries']:
        mask = cv2.inRange(roi , np.array(teamColorLower) , np.array(teamColorUpper))
        roiMasked = cv2.bitwise_and(roi,roi,mask=mask)
        totalColorPix = countNonBalckPix(roiMasked)
        totalPix = countNonBalckPix(roi)
        colorPixRatio = totalColorPix / totalPix
        ratioList.append(colorPixRatio)
        #print(f'totalColrPix:{totalColorPix} , totalPx:{totalPix}')
        if show == True:
            cv2.imshow("images", np.hstack([roi,roiMasked]))
            if cv2.waitKey(0) & 0xFF == ord('q'):
              cv2.destroyAllWindows() 

    return np.array(ratioList)    


# In[17]:


# img = cv2.imread('./dataandmodles/data/cavs.JPG')
# roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# getColorRatio(roi, show=True)


# In[18]:


def compareRatio(ratioList):
    '''
    Helper function for detectTeam
    Finds the team with highest color ratio.
    Returns string team names or "Uncertain" if not sure
    '''
    maxRatio = max(ratioList)
    if maxRatio < 0.1:
        return 'Uncertain'
    else:      
        if ratioList[1] > ratioList[0]:
            return 'team1'
        elif ratioList[1] <= ratioList[0]:
            return'team0'
            
    


# In[19]:


def detectTeamBasic(img,left,top,right,bottom,mask=None):
    '''
    Given an image(BGR) and the location of ROI
    Finds the team based on ROI color
    '''
    roi = getRoi(img,left,top,right,bottom,mask)
    if args['yolo']:
        roiHSV = np.array(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    else:#For mask rcnn, roi will contain list of pixles in mask region
        roi_reshaped = np.reshape(roi,(1,roi.shape[0],3))
        roiHSV = np.array(cv2.cvtColor(roi_reshaped, cv2.COLOR_BGR2HSV))

    ratioList = getColorRatio(roiHSV)
    team = compareRatio(ratioList)
    
    if args['yolo']:
        return team
    else:
        return team , roi


# In[20]:


def getTeamInfo(team):
    if team == 'Uncertain':
        return (0,0,0) , 'Uncertain'
    elif team == 'team0': 
        return (0,213,255) , args[team]
    else:
        return (36,36,158) , args[team]


# #Team using K-means

# In[21]:


def findHistogram(learner):
    '''
    Helper function for detectTeamsKmeans
    Returns a histrogam object for an ROI
    '''
    numLabels = np.arange(0, len(np.unique(learner.labels_)) + 1)
    (hist, _) = np.histogram(learner.labels_, bins=numLabels)
    
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


# In[22]:


def resizeForKMeans(roi):
    '''
    Helper function for detectTeamsKmenas
    Given an roi in HSV space
    Returns reshaped roi of (NumberofPixles x channels)
    '''
    return roi.reshape((roi.shape[0] * roi.shape[1],3))


# In[23]:


def getClustersAndPercatage(hist , learner):
    '''
    Helper function for detectTeansKmeans
    Returns a dict with cluster object {'c1':[h,s,v,percentage] , c2:[..]}
    '''
    clusters = {}
    for index,(percent, color) in enumerate(zip(hist, learner.cluster_centers_)):
        colorList = color.astype("uint8").tolist()
        cluster = f'c{index}'
        clusters[cluster] = [colorList[0], colorList[1], colorList[2], int(percent*100)]
    return clusters
    


# In[24]:


def getLargestCluster(clusters):
    '''
    Helper function for detectTemsKmeans
    Returns the name/key of the largest cluster in the clusters dict
    '''
    percentages = np.array([clusters[cluster][3]for cluster in clusters])
    max_index = np.argmax(percentages)
    return list(clusters.keys())[max_index]


# In[25]:


def getEuclidianDistance(hsv1,hsv2):
    return distance.euclidean(hsv1,hsv2)


# In[26]:


def getLearner(nClusters):
    '''
     Returns a KMeans Learner object
    '''
    learner = KMeans(n_clusters=nClusters) #cluster number
    return learner


# In[72]:


def detectTeamKmeans(learner,img,left,top,right,bottom,mask=None):
    '''
    Given an image(BGR) and the location of ROI
    Returns the team using K-Means clustering
    '''

    roi = getRoi(img,left,top,right,bottom,mask)
    
    if args['yolo']:
        roiHSV = np.array(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    else:
        roiReshaped = np.reshape(roi,(1,roi.shape[0],3))
        roiHSV = np.array(cv2.cvtColor(roiReshaped, cv2.COLOR_BGR2HSV))


    
    roiHSV = resizeForKMeans(roiHSV) #represent as (row*column,channel number) eg. [[0,0,255],[..]]
    learner.fit(roiHSV)
    hist = findHistogram(learner)
    clusters = getClustersAndPercatage(hist,learner) #clusters is a dict {c1:[h,s,v,perc],c2:[]}
 
    hsv = clusters[getLargestCluster(clusters)][:-1] # is a list [h,s,v]

    teamIndex = np.argmin(np.array([
                                getEuclidianDistance(hsv, args['team0HSV']),
                                 getEuclidianDistance(hsv, args['team1HSV'])
                     ]))
    team = f'team{teamIndex}'
    #hsv is returned for testing purposes. To check the k-means cluster mean
    return team,hsv,roi

    


# #Controller to switch between basic and Kmeans

# In[28]:


def detectTeam(img,left,top,right,bottom, algo='basic', mask=None, learner=None):
    if algo == 'basic':
        if args['yolo']: #yolo
            return detectTeamBasic(img,left,top,right,bottom)
        else: #mask rcnn
            return detectTeamBasic(img,left,top,right,bottom,mask)
    else:
        if args['yolo']:#yolo with k-means
            return detectTeamKmeans(learner,img,left,top,right,bottom)
        else: #mask-rcnn with k-means
            return detectTeamKmeans(learner,img,left,top,right,bottom,mask)
            


# #Object detection boiler template (YOLO) Basic or K-Means

# In[29]:


if args['yolo']:
    videoPath ='./dataandmodles/data/3-Pointer2.mov'
    cap = cv2.VideoCapture(videoPath)
    frameCount = 0 
    rawFrame=[]
    learner = getLearner(3)
    net = args['net']
    output_layer_names = args['output_layer_names']
    while cap.isOpened():
        ret,frame = cap.read()
        frameCopy = frame[:]
        if ret:
            start_t = cv2.getTickCount()

            #do stuff
            # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
            tensor = (cv2.dnn.blobFromImage(frame , 1/255 , (args["inpWidth"], args["inpHeight"]) , [0,0,0] , 1, 
                                            crop=False))
            # set the input to the CNN network
            net.setInput(tensor)
            results = net.forward(output_layer_names)

            args['confThreshold'] = cv2.getTrackbarPos(trackbarName,windowName) / 100
            classIDs, confidences, boxes = (postprocess(frame, results, args["confThreshold"], 
                                                        args["nmsThreshold"]))
            for detected_object in range(0, len(boxes)):

                box = boxes[detected_object]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]

                bboxFit = check_bbox_size(width,height, *frame.shape[0:-1])
                labelFit = check_label(classes[classIDs[detected_object]])
                if bboxFit and labelFit and left>0:
                    team,hue= detectTeam(frameCopy, left ,top, left+width,top+height , algo='kmeans' ,
                                     learner=learner)

                    teamColor, teamName = getTeamInfo(team)
                    teamName = f'{team},{hue}'
                    (drawPred(frame,teamName,classes[classIDs[detected_object]], 
                              confidences[detected_object], 
                              left, top, left + width, top + height, 
                              teamColor))

                t,_ = net.getPerfProfile()
                inference_t = (t * 1000.0 / cv2.getTickFrequency())
                label = ('Inference time: %.2f ms' % inference_t) + (' (Framerate: %.2f fps' % (1000 / inference_t)) + ')'
                cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                if frameCount == 496: 
                    print(f'{left},{top},{left + width},{top + height}')
                    rawFrame = frame[:,:,:]
                frameCount += 1
    #         End of do stuff

            cv2.imshow(windowName,frame)
            (cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_FULLSCREEN&False))    

            time_now = cv2.getTickCount()
            stop_t = ((time_now - start_t)/cv2.getTickFrequency())*1000

            #cv2.imshow("YOLO" , frame)

            key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF
            if key == ord('q'):
                break  
        else:
            cap.release()
            break
    cap.release()
    cv2.destroyAllWindows()
    


# #Object detection boiler template (MASK RCNN) Basic or Kmeans

# In[30]:


def fitMaskRcnn(frame):
    net = args['net']
    output_layer_names = args['output_layer_names']
    tensor = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(tensor)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    classIDs, confidences, boxes, masks = (postprocess(frame, boxes, args["confThreshold"], 
                                          args["nmsThreshold"],yolo=False,raw_masks=masks))
    return classIDs,confidences,boxes,masks


# In[31]:


def getBoxesAndMask(boxes,masks,detected_object_index):
    box = boxes[detected_object_index]
    mask = masks[detected_object_index] #getting the mask of a particular detected obj
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    return left,top,width,height,mask
    


# In[33]:


# #Writing Video file 
# 

# In[34]:


from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[35]:


# def detectVideos(frame):

#     learner = args['knnLearner']
#     net = args['net']
#     output_layer_names = args['output_layer_names']
    
    
#     h,w = frame.shape[0], frame.shape[1]
#     frameCopy = np.copy(frame)
    
    
#     classIDs,confidences,boxes,masks = fitMaskRcnn(frame)

   
#     for detected_object in range(0, len(boxes)):
        
#         left,top,width,height,mask = getBoxesAndMask(boxes, masks, detected_object)

#         bboxFit = check_bbox_size(width,height, *frame.shape[0:-1])
#         labelFit = check_label(classIDs[detected_object])
#         if bboxFit and labelFit and left>0:
#             #uncommet to use basic fast rcnn and remove hsv from teamName
#             #team,roi= detectTeam(frameCopy, left ,top, left+width,top+height , algo='basic',mask=mask)
#             team,hsv,roi= (detectTeam(frameCopy, left ,top, left+width,top+height , algo='kmeans',
#                                      learner=learner
#                                      ,mask=mask))
            
#             teamColor, teamName = getTeamInfo(team)
#             teamName = f'{team},{hsv}'
            
#             resizedRoi,roiCropped,leftLarger,topLarger,rightLarger,bottomLargerresizedRoi = transformRoiBoka(mask,roi,frame,h,w,left,top,left+width,top+height)
            
#             actionClass = getActionClass(resizedRoi)

#             (drawPred(frameCopy,actionClass,teamName, 
#                       confidences[detected_object], 
#                       left, top, left + width, top + height, 
#                       teamColor,roi=roi ,mask=mask))
#     return frameCopy


# In[36]:


def detectVideos(frame):
    learner = args['knnLearner']
    net = args['net']
    output_layer_names = args['output_layer_names']
    
    h,w = frame.shape[0], frame.shape[1]
    frameCopy = np.copy(frame)
    
    
    classIDs,confidences,boxes,masks = fitMaskRcnn(frame)

    #Gaze detection variable
    posenetPredCombined = {'detectionList':[]}
    frame_bboxs = []

    #Drawing variables
    frame_rois = []
    frame_masks = []
    frame_rcnnbboxes = []
    frame_teamColors = []
    frame_actions = []

    #Tracker Variables 
    frame_posenetbboxs = {'bbox':[]}
    for detected_object in range(0, len(boxes)):

        left,top,width,height,mask = getBoxesAndMask(boxes, masks, detected_object)

        bboxFit = check_bbox_size(width,height, *frame.shape[0:-1])
        labelFit = check_label(classIDs[detected_object])
        if bboxFit and labelFit and left>0:
            #uncommet to use basic fast rcnn and remove hsv from teamName
            #team,roi= detectTeam(frameCopy, left ,top, left+width,top+height , algo='basic',mask=mask)
            team,hsv,roi= (detectTeam(frameCopy, left ,top, left+width,top+height , algo='kmeans',
                                     learner=learner
                                     ,mask=mask))

#             end = time.time()
#             teamDetectionTime = end - start 
#             teamDetectionTimes = np.append(teamDetectionTimes,teamDetectionTime)


            teamColor, teamName = getTeamInfo(team)
            teamName = f'{team},{hsv}'

            resizedRoi,roiCropped,leftLarger,topLarger,rightLarger,bottomLarger = transformRoiBoka(mask,
                                                                                               roi,frame,
                                                                                               h,w,left,top,
                                                                                               left+width,
                                                                                               top+height )

            
            #resizedRoiWithPose = drawPose(resizedRoi)
            actionClass,posenetPred,posenetBoxs = getActionClass(resizedRoi)

   

            posenetPred = mapPosesToMainImageWrapper(posenetPred,
                                      topLarger, 
                                      leftLarger,resizedRoi,
                                      roiCropped)

            #GAZE DETECTION PREP(Combine posenetPred for gaze detection):
            for pose in posenetPred['detectionList']:
                posenetPredCombined['detectionList'].append(pose)
            if len(posenetPred['detectionList']) != 0:
                frame_bboxs.append([leftLarger,topLarger,rightLarger,bottomLarger])
                frame_actions.append(actionClass)
                frame_masks.append(mask)
                frame_rois.append(roi)
                frame_rcnnbboxes.append([left ,top, left+width,top+height])
                frame_teamColors.append(teamColor)
            #END GAZE DETECTION PREP:
            #Tracker Prep
            frame_posenetbboxs['bbox'].extend(posenetBoxs['bbox'])
            #End Tracker Prep


       #GAZE DETECTION tag1
    if len(frame_bboxs):
       ball_position,poseIndex = GazeModule.fitGazeModule(posenetPredCombined,frame_bboxs,frame_actions)
       drawMaskRcnnHelperFunction(frameCopy,poseIndex,frame_rcnnbboxes,
                       frame_teamColors,frame_actions,
                       frame_rois,frame_masks)
       DrawPose.drawBall(frameCopy,w,h,ball_position,(255,0,0),20)
    #END GAZE DETECTION tag1

    #Tracker

    if len(list(Tracker.previousFramePose.keys())) != 0 :
        #actions = [[f] for f in frame_actions]
        finalActionClasses = Tracker.updatePosesClasses(posenetPredCombined,
                                                  frame_actions,
                                                  frame_posenetbboxs,
                                                  frameCopy.shape[0],frameCopy.shape[1])


    Tracker.setPrevFrameWrapper(frame_posenetbboxs,posenetPredCombined,
                                                frame_actions,
                                                frameCopy.shape[0],frameCopy.shape[1])
    #End Tracker

    #Drawing boxes and ball
    if len(frame_bboxs):
       drawMaskRcnnHelperFunction(frameCopy,poseIndex,frame_rcnnbboxes,
                                   frame_teamColors,finalActionClasses,
                                   frame_rois,frame_masks)
       DrawPose.drawBall(frameCopy,w,h,ball_position,(255,0,0),20)
    
    


    return frameCopy


# In[37]:


def initializeModels():
    loadMaskRcnn()
    ActionClassificationCosine.loadPosenetModel()
    args['knnLearner'] = getLearner(3)
def proccessFrame(frame):
    global counter
    if args['net'] == None:
        initializeModels()
    if counter%1 == 0:
        outFrame = detectVideos(frame)
    counter += 1
    return outFrame


# In[ ]:





# In[38]:


# inputPath  = './dataandmodles/data/videotest/input1.mov'
# outputPath = './dataandmodles/data/videotest/input1__out_ball.mp4'
# counter = 0

# clip1 = VideoFileClip(inputPath)
# outClip1 = clip1.fl_image(proccessFrame)
# %time  outClip1.write_videofile(outputPath,audio=False)


# #Visual Debugger

# In[60]:


TESTARGS = {
            'debugger' : False,
            'testImage1' : './dataandmodles/data/1v1image.png',
            'testResultPath' : './dataandmodles/data/trackPlayers'
           }
import DrawPose


# In[40]:


def load_image():
    img = cv2.imread(TESTARGS['testImage1'])
    return img, img.shape[0] , img.shape[1]

def write_image(img,fileName):
    cv2.imwrite(f'{TESTARGS["testResultPath"]}/{fileName}.png' , img)
    


# In[41]:


def getLargerBbox(imgh,imgw,left,top,right,bottom):
    n = 10
    left,top = (left-n,top-n) #left_top (w,h)/ (col,row)
    right,bottom = (right+n ,bottom+n ) #right_bottom
    #Applying boundary condition
    left,top = max(0,left) , max(0,top) 
    right,bottom =  min(right,imgw) , min(bottom,imgh) 
    return left,top,right,bottom

    
def roiBackgroundSubtraction(mask,roi,h,w,left,top,right,bottom):
    #TEST PASSEDf
    empty_image = np.zeros((h,w,3))
    empty_image[top:bottom, left:right][mask] = roi
    return empty_image

def cropRoi(img,left,top,right,bottom):
    roiCropped = ActionClassificationCosine.cropRoi(img,top,left,bottom,right)
    return roiCropped

def resizeRoi(img):
    img = imutils.resize(img , width=244)
    return img

def transformRoi(mask,roi,img,h,w,left,top,right,bottom):
    finalRoi = cropRoi(img,left,top,right,bottom)
    finalRoi = resizeRoi(finalRoi)
    return finalRoi

def transformRoiBoka(mask,roi,img,h,w,left,top,right,bottom):
    blurredImg = np.array(cv2.bilateralFilter(img,9,75,75))
   
    blurredImg[top:bottom, left:right][mask] = roi
    left,top,right,bottom = getLargerBbox(h,w,left,top,right,bottom)
    roiCropped = cropRoi(blurredImg,left,top,right,bottom)
    roi = resizeRoi(roiCropped)
    #TODO: originally it only returned roi
    return roi,roiCropped,left,top,right,bottom


def drawPose(resizedRoi):
    poses,boxes = ActionClassificationCosine.getPoses(resizedRoi)
    resizedRoiWithPose = DrawPose.draw_skel_and_kp(poses,resizedRoi,(255,0,255))
    return resizedRoiWithPose

def getActionClass(resizedRoi):
    st = time.time()
    poses,boxes = ActionClassificationCosine.getPoses(resizedRoi)
    et = time.time()
    posenetTime = round(et-st,3)
    posesCopy = copy.deepcopy(poses)
    if len(boxes['bbox']):
        st = time.time()
        classes = ActionClassificationCosine.fit(poses,boxes,resizedRoi.shape[0] ,
                                                 resizedRoi.shape[1])
        et = time.time()
        poseMatchingTime = round(et - st,3)
    else:
        classes = [['unknown']]
    
    return classes[0][0],posesCopy,boxes


# In[42]:


#Getting Coordinates from cropped to main image 
def getScaleFactors(imgW, imgH, originalImgH , originalImgW):
    '''

       Helper function for get_data()
       Parameter
       -----------
       imgW , imgH : dimensions of the  image where the coords needs to go to
       originalImgH , originalImgW : dimensions of the  image where the coords needs to go from
       Returns a tensor with given scalefactor (wdith,height). 
    '''
    scale = (imgW/originalImgW,imgH/originalImgH);
    return scale

def mapCoord(coord,scaleFactors):
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

def mapCoordToCroppedImage(keypointPosition,scaleFactor):
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
    coord =  mapCoord(list(keypointPosition.values()),scaleFactor)
    #coords (row,col)
    keypointPosition['x'] = coord[0]
    keypointPosition['y'] = coord[1]
    
    return keypointPosition

def mapKeypointsPosition(pose,enlargedCroppedH,enlargedCroppedW,croppedH,croppedW):
    '''
    Helper function of mapPosesToCroppedImage
    '''
    scaleFactor = getScaleFactors(croppedW,croppedH,enlargedCroppedH, enlargedCroppedW)
    for keypoint in pose['keypoints']:
        keyPointPosition = mapCoordToCroppedImage(keypoint['position'],scaleFactor)
        keypoint['position'] = keyPointPosition
    return pose

def mapPosesToCroppedImage(posenetPred,enlargedCroppedH,enlargedCroppedW,croppedH,croppedW):
    '''
    This functions maps the posenetPred keypoints to the original cropped image 
    '''
    for index,pose in enumerate(posenetPred['detectionList']):
        pose = mapKeypointsPosition(pose,enlargedCroppedH,enlargedCroppedW,croppedH,croppedW)
        posenetPred['detectionList'][index] = pose
    return posenetPred


def mapKeypointsPositionToMainImage(pose,top,left):
    '''
    Helper function of mapPosesToMainImage
    '''
    for keypoint in pose['keypoints']:
        keypoint['position']['x'] =  keypoint['position']['x'] + left
        keypoint['position']['y'] =  keypoint['position']['y'] + top
    return pose
def mapPosesToMainImage(posenetPred,top,left):
    '''
    This function takes posenetPred where keypoints coords are relative to the original 
    cropped image 
    '''
    for index,pose in enumerate(posenetPred['detectionList']):
        pose = mapKeypointsPositionToMainImage(pose,top,left)
        posenetPred['detectionList'][index] = pose
    return posenetPred  

def mapPosesToMainImageWrapper(posenetPred,topLarger, leftLarger,resizedRoi,roiCropped):
    '''
    This function is a wrapper function that maps raw posenetPred for cropped images to poses 
    in the full image. 
    Used for gaze detection.
    '''
    posenetPred = mapPosesToCroppedImage(posenetPred,
                                         resizedRoi.shape[0], resizedRoi.shape[1],
                                         roiCropped.shape[0],roiCropped.shape[1])
    posenetPred = mapPosesToMainImage(posenetPred,topLarger,leftLarger)
    return posenetPred


# In[43]:


#Tracker 
def tracker(frame,poses,bboxes,classes,prevFrameMidpoints):
        
    cv2.imshow('test', frame)


# In[44]:


def drawMaskRcnnHelperFunction(frameCopy,poseIndex,frame_rcnnbboxes,
                               frame_teamColors,frame_actions,frame_rois,frame_masks,playersScores=False):
    
   
    
    for index,bbox in enumerate(frame_rcnnbboxes):
        if playersScores:
            playerScore = playersScores[index]
        else: 
            playerScore = 0
        if index != poseIndex:
            color = frame_teamColors[index]
        else:
            #color = (255,0,255)
            color =frame_teamColors[index]
        DrawPose.drawPred(frameCopy,
                          frame_actions[index],
                          playerScore,
                          bbox[0],bbox[1], bbox[2], bbox[3], 
                          color,
                          yolo=False,
                          roi=frame_rois[index],
                          mask=frame_masks[index])


# In[74]:


def visualDebugger():
    loadMaskRcnn()
    #ActionClassificationCosine.loadPosenetModel()
    rawFrame=[]
    learner = getLearner(2)
    net = args['net']
    output_layer_names = args['output_layer_names']
    
    frame,h,w = load_image()
    frameCopy = frame[:]
    frameCopy2 = frame[:]
    

    
    classIDs,confidences,boxes,masks = fitMaskRcnn(frame)
    
    #Gaze detection variable
    posenetPredCombined = {'detectionList':[]}
    frame_bboxs = []
    frame_actions = []
    
    #Drawing variables
    frame_rois = []
    frame_masks = []
    frame_rcnnbboxes = []
    frame_teamColors = []
    for detected_object in range(0, len(boxes)):
        
        left,top,width,height,mask = getBoxesAndMask(boxes, masks, detected_object)

        bboxFit = check_bbox_size(width,height, *frame.shape[0:-1])
        labelFit = check_label(classIDs[detected_object])
        if bboxFit and labelFit and left>0:
            #uncommet to use basic fast rcnn and remove hsv from teamName
            #team,roi= detectTeam(frameCopy, left ,top, left+width,top+height , algo='basic',mask=mask)
            team,hsv,roi= (detectTeam(frameCopy, left ,top, left+width,top+height , algo='kmeans',
                                     learner=learner
                                     ,mask=mask))
            
            teamColor, teamName = getTeamInfo(team)
            teamName = f'{team},{hsv}'
            
            resizedRoi,roiCropped,leftLarger,topLarger,rightLarger,bottomLarger = transformRoiBoka(mask,
                                                                                                 roi,frame,
                                                                                                  h,w,left,top,
                                                                                                  left+width,
                                                                                                  top+height )
            
            #write_image(resizedRoi,f'{detected_object}_resizedRoi.png')
            start = time.time()

            actionClass,posenetPred,_ = getActionClass(resizedRoi)
            
            
            
            #GAZE DETECTION PREP(Combine posenetPred for gaze detection):
            
            posenetPred = mapPosesToCroppedImage(posenetPred,
                                                 resizedRoi.shape[0], resizedRoi.shape[1],
                                                 roiCropped.shape[0],roiCropped.shape[1])
            

            posenetPred = mapPosesToMainImage(posenetPred,topLarger,leftLarger)
    
           
            for pose in posenetPred['detectionList']:
                posenetPredCombined['detectionList'].append(pose)
            if len(posenetPred['detectionList']) != 0:
                frame_bboxs.append([leftLarger,topLarger,rightLarger,bottomLarger])
                frame_actions.append(actionClass)
                frame_masks.append(mask)
                frame_rois.append(roi)
                frame_rcnnbboxes.append([left ,top, left+width,top+height])
                frame_teamColors.append(teamColor)
            #END GAZE DETECTION PREP:
            end = time.time()
            print(actionClass)
            print(round(end-start,3))
    
           
    #Gaze Module
    if len(frame_bboxs):
       ball_position,poseIndex = GazeModule.fitGazeModule(posenetPredCombined,frame_bboxs,frame_actions)
       drawMaskRcnnHelperFunction(frameCopy,poseIndex,frame_rcnnbboxes,
                       frame_teamColors,frame_actions,
                       frame_rois,frame_masks)
       DrawPose.drawBall(frameCopy,w,h,ball_position,(255,0,0),20)
    

    #Posenet On the whole image 
    print('All test images saved')
    
        


# In[75]:





# #Test Ball Localization

# In[46]:


import TestBallLocalization


# In[47]:


def test(frame,h,w):
    
    rawFrame=[]
    learner = getLearner(3)
    net = args['net']
    output_layer_names = args['output_layer_names']
    
    frameCopy = frame[:]
    frameCopy2 = frame[:]
    

    
    classIDs,confidences,boxes,masks = fitMaskRcnn(frame)
    
    #Gaze detection variable
    posenetPredCombined = {'detectionList':[]}
    frame_bboxs = []
    frame_actions = []
    
    #Drawing variables
    frame_rois = []
    frame_masks = []
    frame_rcnnbboxes = []
    frame_teamColors = []
    for detected_object in range(0, len(boxes)):
        
        left,top,width,height,mask = getBoxesAndMask(boxes, masks, detected_object)

        bboxFit = check_bbox_size(width,height, *frame.shape[0:-1])
        labelFit = check_label(classIDs[detected_object])
        if bboxFit and labelFit and left>0:
            #uncommet to use basic fast rcnn and remove hsv from teamName
            #team,roi= detectTeam(frameCopy, left ,top, left+width,top+height , algo='basic',mask=mask)
            team,hsv,roi= (detectTeam(frameCopy, left ,top, left+width,top+height , algo='kmeans',
                                     learner=learner
                                     ,mask=mask))
            
            teamColor, teamName = getTeamInfo(team)
            teamName = f'{team},{hsv}'
            
            resizedRoi,roiCropped,leftLarger,topLarger,rightLarger,bottomLarger = transformRoiBoka(mask,
                                                                                                 roi,frame,
                                                                                                  h,w,left,top,
                                                                                                  left+width,
                                                                                                  top+height )
            
            #write_image(resizedRoi,f'{detected_object}_resizedRoi.png')
           

            actionClass,posenetPred,_ = getActionClass(resizedRoi)
            
            
            
            #GAZE DETECTION PREP(Combine posenetPred for gaze detection):
            
            posenetPred = mapPosesToCroppedImage(posenetPred,
                                                 resizedRoi.shape[0], resizedRoi.shape[1],
                                                 roiCropped.shape[0],roiCropped.shape[1])
            

            posenetPred = mapPosesToMainImage(posenetPred,topLarger,leftLarger)
    
           
            for pose in posenetPred['detectionList']:
                posenetPredCombined['detectionList'].append(pose)
            if len(posenetPred['detectionList']) != 0:
                frame_bboxs.append([leftLarger,topLarger,rightLarger,bottomLarger])
                frame_actions.append(actionClass)
                frame_masks.append(mask)
                frame_rois.append(roi)
                frame_rcnnbboxes.append([left ,top, left+width,top+height])
                frame_teamColors.append(teamColor)
            #END GAZE DETECTION PREP:

    #Gaze Module
    start = time.time()
    if len(frame_bboxs):
       ball_position,poseIndex = GazeModule.fitGazeModule(posenetPredCombined,frame_bboxs,frame_actions)
    else:
        ball_position = (int(w/2),(h/2))
    end = time.time()
    return ball_position , round(end-start,3)


# In[48]:


#%debug


# In[49]:


def runTest():
    loadMaskRcnn()
#     global frame_rsme_collection , yhat
    #ActionClassificationCosine.loadPosenetModel()
    totalTime = []
    yhat = []
    images, labels = TestBallLocalization.imgs , TestBallLocalization.labels
    frame_rsme_collection = []
    for index,frame in enumerate(images):
        ball_pos , timeTaken = test(frame,frame.shape[0],frame.shape[1])
        ball_pos = list(ball_pos)
        print(f'{ball_pos},{labels[index]}')
        yhat.append(ball_pos)
        totalTime.append(timeTaken)
        #Drawing and saving image
        frame_rsme = TestBallLocalization.RMSE([ labels[index] ],[ball_pos])
        ball_pos = (int(ball_pos[0]), int(ball_pos[1]))
        DrawPose.drawBall(frame,frame.shape[1],frame.shape[0],tuple(ball_pos),(255,0,0),20)
        DrawPose.drawBall(frame,frame.shape[1],frame.shape[0],tuple(labels[index]),(0,0,255),20)
        write_image(frame,f'{round(frame_rsme,3)}')
        
        #Prep for min RMSE and MAX
        frame_rsme_collection.append(frame_rsme)
        
    
    #Final total RMSE and time  
    rmse = TestBallLocalization.RMSE(labels,yhat)
    totalTime = np.mean(np.array(totalTime)) 
    #Final Min and Max 
    frame_rsme_collection = np.array(frame_rsme_collection)
    minRmse = np.min(frame_rsme_collection)
    maxRmse = np.max(frame_rsme_collection)
    #Writing Results 
    content = f'RMSE:{rmse}\nMax RMSE:{maxRmse}\n{minRmse}\nMean Time:{totalTime}'
    TestBallLocalization.writeResults('GazeTestResults',content)
        


# In[50]:


#runTest()


# #Test Player Localization

# In[51]:


import TestPlayerLocalization


# In[52]:


def test(frame):
    pred_boxes={'boxes':[],'scores':[]}
    net = args['net']
    output_layer_names = args['output_layer_names']
        
    #do stuff
    # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
    tensor = (cv2.dnn.blobFromImage(frame , 1/255 , (args["inpWidth"], args["inpHeight"]) , [0,0,0] , 1, 
                                    crop=False))
    # set the input to the CNN network
    net.setInput(tensor)
    results = net.forward(output_layer_names)

    classIDs, confidences, boxes = (postprocess(frame, results, args["confThreshold"], 
                                                args["nmsThreshold"]))
    for detected_object in range(0, len(boxes)):
        box = boxes[detected_object]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        bboxFit = check_bbox_size(width,height, *frame.shape[0:-1])
        labelFit = classes[classIDs[detected_object]] == 'person'
        if bboxFit and labelFit and left>0:
            pred_boxes['boxes'].append([left,top,left+width,top+height]) 
            pred_boxes['scores'].append(confidences[detected_object])
    return pred_boxes

    #         End of do stuff

           


# In[53]:


def runYoloTest():
    imgs = TestPlayerLocalization.imgs
    labels = TestPlayerLocalization.labels
    imgNames = TestPlayerLocalization.images
    gt_boxes = {}
    yhat_boxes = {}
    
    time_collection = []
    for index,frame in enumerate(imgs):
        st = time.time()
        pred_boxes = test(frame)
        et = time.time()
        time_collection.append(et-st)
        
        #Prep variables for mAP
        yhat_boxes[imgNames[index]] = pred_boxes
        gt_boxes[imgNames[index]] = labels[index]

        #Draw pred_boxes
#         for box in pred_boxes['boxes']:
#             DrawPose.drawPred(frame,'',1,*box,(255,0,0),yolo=True)
#         #Draw Ground Truth
#         for box in labels[index]:
#             DrawPose.drawPred(frame,'',1,*box,(0,0,255),yolo=True)
#         write_image(frame,f'{imgNames[index]}')
    
    
    #Find average time taken
    time_collection = np.array(time_collection)
    totalTime  = np.mean(time_collection)
    
    #Find mAP
    result = TestPlayerLocalization.get_avg_precision_at_iou(gt_boxes,yhat_boxes,iou_thr=0.4)
    print(result)
    mAP,model_thrs = result['avg_prec']  , result['model_thrs']
    precisions, recalls  = result['precisions'], result['recalls']
    #Writing Result to the time
    content = f'Precisions:{precisions}\nRecalls:{recalls}\nmAP:{mAP}\model thres:{model_thrs}\nTime Taken:{totalTime}'
    TestPlayerLocalization.writeResults('YoloResult',content)

        


# In[54]:


# args['yolo'] = True
# loadYolo()
# runYoloTest()


# In[55]:


def testMaskRcnn(frame):
    pred_boxes={'boxes':[],'scores':[]}
    net = args['net']
    output_layer_names = args['output_layer_names']
    #do stuff
    classIDs,confidences,boxes,masks = fitMaskRcnn(frame)
    for detected_object in range(0, len(boxes)):
        left,top,width,height,mask = getBoxesAndMask(boxes, masks, detected_object)
        bboxFit = check_bbox_size(width,height, *frame.shape[0:-1])
        labelFit = check_label(classIDs[detected_object])
        if bboxFit and labelFit and left>0:
            pred_boxes['boxes'].append([left,top,left+width,top+height]) 
            pred_boxes['scores'].append(confidences[detected_object])
    return pred_boxes

  
#       End of do stuff


    


# In[56]:


def runMaskRcnnTest():
    loadMaskRcnn()
    imgs = TestPlayerLocalization.imgs
    labels = TestPlayerLocalization.labels
    imgNames = TestPlayerLocalization.images
    gt_boxes = {}
    yhat_boxes = {}
    
    time_collection = []
    for index,frame in enumerate(imgs):
        st = time.time()
        pred_boxes = testMaskRcnn(frame)
        et = time.time()
        time_collection.append(et-st)
        
        #Prep variables for mAP
        yhat_boxes[imgNames[index]] = pred_boxes
        gt_boxes[imgNames[index]] = labels[index]

        #Draw pred_boxes
#         for box in pred_boxes['boxes']:
#             DrawPose.drawPred(frame,'',1,*box,(255,0,0),yolo=True)
#         #Draw Ground Truth
#         for box in labels[index]:
#             DrawPose.drawPred(frame,'',1,*box,(0,0,255),yolo=True)
#         write_image(frame,f'{imgNames[index]}')
    
    
    #Find average time taken
    time_collection = np.array(time_collection)
    totalTime  = np.mean(time_collection)
    
    #Find mAP
    result = TestPlayerLocalization.get_avg_precision_at_iou(gt_boxes,yhat_boxes,iou_thr=0.5)
    print(result)
    mAP,model_thrs = result['avg_prec']  , result['model_thrs']
    precisions, recalls  = result['precisions'], result['recalls']
    #Writing Result to the time
    content = f'Precisions:{precisions}\nRecalls:{recalls}\nmAP:{mAP}\model thres:{model_thrs}\nTime Taken:{totalTime}'
    TestPlayerLocalization.writeResults('MaskRCNNResult',content)


# In[57]:


#runMaskRcnnTest()


# In[ ]:


if __name__ == '__main__':
    try:
        print("Sending Sever Message to load Posenet model from google")
        ActionClassificationCosine.loadPosenetModel()
        args['mask-rcnn'] = True
    except Exception:
        print('Error Occured: Please open the Javascript Server')
    if args['mask-rcnn']:
        windowName , trackbarName = openDisplayWindow()
        loadMaskRcnn()
        net = args['net']
        output_layer_names = args['output_layer_names']

        #videoPath ='./dataandmodles/data/demoVideos/3-Pointer2.mov'
        videoPath = './dataandmodles/data/demoVideos/unOccludedShooting.mov'
        cap = cv2.VideoCapture(videoPath)
        frameCount = 0 
        rawFrame=[]

        #TeamDetection Learner
        learner = getLearner(3)
        team0Score = 0
        team1Score = 0
        waitFrames0 = 0
        waitFrames1 = 0
        justShot0, justShot1 = False, False
        while cap.isOpened():
            ret,frame = cap.read()
            frameCopy = frame[:]
            h,w = frame.shape[0] , frame.shape[1]
            #Time Logs for each new frame
            actionTimes = np.array([]) 
            teamDetectionTimes = np.array([])
            playerDetectionTimes = np.array([])

            #Init player scores the new frame
            playersScores = []
            if ret:
                start_t = cv2.getTickCount()
                #do stuff
                args['confThreshold'] = cv2.getTrackbarPos(trackbarName,windowName) / 100
                start = time.time()
                classIDs,confidences,boxes,masks = fitMaskRcnn(frame)
                end = time.time()
                playerDetectionTime = end  - start 
                playerDetectionTimes = np.append(playerDetectionTimes,playerDetectionTime)

                #Gaze detection variable
                posenetPredCombined = {'detectionList':[]}
                frame_bboxs = []

                #Drawing variables
                frame_rois = []
                frame_masks = []
                frame_rcnnbboxes = []
                frame_teamColors = []
                frame_actions = []

                #Tracker Variables 
                frame_posenetbboxs = {'bbox':[]}
                for detected_object in range(0, len(boxes)):

                    left,top,width,height,mask = getBoxesAndMask(boxes, masks, detected_object)

                    bboxFit = check_bbox_size(width,height, *frame.shape[0:-1])
                    labelFit = check_label(classIDs[detected_object])
                    if bboxFit and labelFit and left>0:
                        #uncommet to use basic fast rcnn and remove hsv from teamName
                        start = time.time()
                        #team,roi= detectTeam(frameCopy, left ,top, left+width,top+height , algo='basic',mask=mask)
                        team,hsv,roi= (detectTeam(frameCopy, left ,top, left+width,top+height , algo='kmeans',
                                                learner=learner
                                                ,mask=mask))

                        end = time.time()
                        teamDetectionTime = end - start 
                        teamDetectionTimes = np.append(teamDetectionTimes,teamDetectionTime)


                        teamColor, teamName = getTeamInfo(team)
                        #teamName = f'{team},{hsv}'



                        resizedRoi,roiCropped,leftLarger,topLarger,rightLarger,bottomLarger = transformRoiBoka(mask,
                                                                                                        roi,frame,
                                                                                                        h,w,left,top,
                                                                                                        left+width,
                                                                                                        top+height )

                        start = time.time()
                        #resizedRoiWithPose = drawPose(resizedRoi)
                        actionClass,posenetPred,posenetBoxs = getActionClass(resizedRoi)

                        end = time.time()
                        actionTime = end - start 
                        actionTimes  = np.append(actionTimes, actionTime)

                        #Track player using jersey color
                        if waitFrames0 <= 0:
                            justShot0 = False
                        if waitFrames1 <= 0:
                            justShot1 == False

                        if teamName == 'MIL':
                            if actionClass == 'shoot' and justShot0 == False:
                                team0Score += 1 
                                justShot0 = True
                                waitFrames0 = 5
                            playersScores.append(team0Score)
                            waitFrames0 = waitFrames0 - 1
                        elif teamName == 'CAVS':
                            if actionClass == 'shoot' and justShot1 == False:
                                team1Score += 1
                                justShot1 = True
                                waitFrames1 = 5
                            playersScores.append(team1Score)
                            waitFrames1 = waitFrames1 - 1




                        posenetPred = mapPosesToMainImageWrapper(posenetPred,
                                                topLarger, 
                                                leftLarger,resizedRoi,
                                                roiCropped)

                        #GAZE DETECTION PREP(Combine posenetPred for gaze detection):
                        for pose in posenetPred['detectionList']:
                            posenetPredCombined['detectionList'].append(pose)
                        if len(posenetPred['detectionList']) != 0:
                            frame_bboxs.append([leftLarger,topLarger,rightLarger,bottomLarger])
                            frame_actions.append(actionClass)
                            frame_masks.append(mask)
                            frame_rois.append(roi)
                            frame_rcnnbboxes.append([left ,top, left+width,top+height])
                            frame_teamColors.append(teamColor)
                        #END GAZE DETECTION PREP:
                        #Tracker Prep
                        frame_posenetbboxs['bbox'].extend(posenetBoxs['bbox'])
                        #End Tracker Prep


                #GAZE DETECTION tag1
                if len(frame_bboxs):
                    ball_position,poseIndex = GazeModule.fitGazeModule(posenetPredCombined,frame_bboxs,frame_actions)
                    drawMaskRcnnHelperFunction(frameCopy,poseIndex,frame_rcnnbboxes,
                                    frame_teamColors,frame_actions,
                                    frame_rois,frame_masks,playersScores=playersScores)
                    DrawPose.drawBall(frameCopy,w,h,ball_position,(255,0,0),20)
                #END GAZE DETECTION tag1

                #Tracker

                if len(list(Tracker.previousFramePose.keys())) != 0 :
                    #actions = [[f] for f in frame_actions]
                    finalActionClasses = Tracker.updatePosesClasses(posenetPredCombined,
                                                            frame_actions,
                                                            frame_posenetbboxs,
                                                            frameCopy.shape[0],frameCopy.shape[1])
                else:
                    finalActionClasses = frame_actions


                Tracker.setPrevFrameWrapper(frame_posenetbboxs,posenetPredCombined,
                                                            frame_actions,
                                                            frameCopy.shape[0],frameCopy.shape[1])
                #End Tracker

                #Drawing boxes and ball
                if len(frame_bboxs):
                    drawMaskRcnnHelperFunction(frameCopy,poseIndex,frame_rcnnbboxes,
                                                frame_teamColors,finalActionClasses,
                                                frame_rois,frame_masks,playersScores=playersScores)
                    DrawPose.drawBall(frameCopy,w,h,ball_position,(255,0,0),20)

        #       End of do stuff


                #Print Time took for each part 
                timeString =   f'Time taken per frame:\nPlayer Detection:{round(np.mean(playerDetectionTimes),2)}s,Action Classification:{round(np.mean(actionTimes),2)}s,Team Detection:{round(np.mean(teamDetectionTimes),2)}s'
                print(timeString)

                # Prinint frame rate 
                t,_ = net.getPerfProfile()
                inference_t = (t * 1000.0 / cv2.getTickFrequency())
                label = ('Inference time: %.2f ms' % inference_t) + (' (Framerate: %.2f fps' % (1000 / inference_t)) + ')'
                cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                #Displaying the image
                cv2.imshow(windowName,frameCopy)
                (cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                                        cv2.WINDOW_FULLSCREEN&False))    

                time_now = cv2.getTickCount()
                stop_t = ((time_now - start_t)/cv2.getTickFrequency())*1000

                #cv2.imshow("MASK-RCNN" , frame)

                key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF
                if key == ord('q'):
                    break  
            else:
                cap.release()
                break

        cap.release()
        cv2.destroyAllWindows()



