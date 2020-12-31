#!/usr/bin/env python
# coding: utf-8

# In[34]:


from sklearn import metrics
import os
import cv2


# In[6]:


def RMSE(y_true,y_pred):
    return metrics.mean_squared_error(y_true , y_pred,squared=False)


# In[38]:





# In[39]:


def getImagePath(image):
    return '/Users/sandeep/Downloads/Data5/Image' + '/' + image

def getLabel(image):
    c = image[0:-4].split(',')
    return [int(c[0]),int(c[1])]    
def loadImages():
    return [cv2.imread(getImagePath(i))for i in images]
def loadLabels():
    return [getLabel(i) for i in images]

#Loading Test Images and Lables
if __name__ == '__main__':  
    images = os.listdir('/Users/sandeep/Downloads/Data5/Image')
    imgs  = loadImages()
    labels = loadLabels()
    


# In[12]:


def writeResults(fname,content):
    basePath = './dataandmodles/data/ballTest/Regression'
    with open(f'{basePath}/{fname}.txt' , 'w') as f:
        f.write(content)


# In[ ]:




