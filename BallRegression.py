#!/usr/bin/env python
# coding: utf-8

# #Importing

# In[12]:


import matplotlib.pyplot as plt
import fastai
import cv2
from fastai.vision import *

import torchvision



# #Loading model

# In[13]:


learn = load_learner("./dataandmodles/models/ballDetection")


# #Getting prediction

# In[14]:


def unscaleCoord(imagePoint):
  '''
  Helper function for getPrediction
  Returns 
  -----------
  unscaleCoord : [Height,Width]
  '''
  s = torch.tensor([imagePoint.size[0]/2 , imagePoint.size[1]/2],dtype=torch.float32)
  scaledCoord = imagePoint.data
  unscaledCoord = (scaledCoord+1.) * (s)
  return unscaledCoord


# In[15]:


def getScaleFactors(imgW, imgH, originalImgH , originalImgW):
    '''
    Helper function for getPrediction
    
    Return
    --------
    A tensor with given scalefactor (wdith,height). 
    '''
    return tensor(originalImgW/imgW , originalImgH/imgH);


# In[16]:


def resizeFrame(frame):
    '''
    Helper function for getPrediction: that resizes the frames.
    Returns image of size (490, 360)
    '''
    resizedImg =  cv2.resize(frame[:] , (490 , 360))
    return resizedImg


# In[17]:


def mapToOrginialImage(coord,scaleFactors):
    '''
      Helper function for getPrediction
      Returns coord (width, height)
    '''
    coordInOriginalImage = coord.flip(1) * scaleFactors
    return tuple(coordInOriginalImage[0].numpy())


# In[18]:


def getPrediction(frame):
    '''
        Given an image, returns the ball coord (x,y)
    '''
    resizedImg  = resizeFrame(frame[:])
#     resizedImg  = frame
    resizedImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
    resizedImg = Image(pil2tensor(resizedImg, dtype=np.float32).div_(255))
    imagePoint = learn.predict(resizedImg)[0]
    scaleFactors = getScaleFactors(490,360,frame.shape[0],frame.shape[1])  
    unscaledCoord = unscaleCoord(imagePoint)
    coordInOriginalImage= mapToOrginialImage(unscaledCoord, scaleFactors);
    return coordInOriginalImage


# #Puting prediction into original image

# In[19]:


def drawPrediction(frame,coord):
    '''
    Given a frame and coordinate
    It draws a circle for you.
    '''
   # scaleFactors = getScaleFactors(490,360,frame.shape[0],frame.shape[1])   
    #coordInOriginalImage= mapToOrginialImage(coord, scaleFactors);
    frame = cv2.circle(frame,coord,30,(255,255,0),1);
    return frame


# #Test

# In[20]:


import import_ipynb
import TestBallLocalization
import DrawPose
import time 


# In[21]:


TESTARGS = {
            'debugger' : False,
            'testImage1' : './dataandmodles/data/cavs.png',
            'testResultPath' : './dataandmodles/data/ballTest/Regression'
           }


# In[22]:


def write_image(img,fileName):
    cv2.imwrite(f'{TESTARGS["testResultPath"]}/{fileName}.png' , img)


# In[27]:


def runTest():
    totalTime = []
    yhat = []
    images, labels = TestBallLocalization.imgs , TestBallLocalization.labels
    frame_rsme_collection = []
    for index,frame in enumerate(images):
        st = time.time()
        ball_pos = getPrediction(frame)
        et = time.time()
        timeTaken = et - st
        
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
    TestBallLocalization.writeResults('RegressionTestResults',content)
    


# In[28]:


#runTest()


# In[ ]:




