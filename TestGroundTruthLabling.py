#!/usr/bin/env python
# coding: utf-8

# In[2]:


from lxml import etree
import os 


# In[8]:


xmlsPath = '/Users/sandeep/Downloads/Data5/label'
imagesPath = '/Users/sandeep/Downloads/Data5/Image'
images = os.listdir(imagesPath)
xmls = os.listdir(xmlsPath)


# In[9]:


xmls[0]


# In[14]:


imageLabelPath = xmlsPath +"/" +xmls[0]


# In[30]:


# path = "/annotation/descendant::submodule[text()='Advanced Databases']/ancestor::student"
def getlabel(imageLabelPath):
    imageXMLFile = etree.parse(imageLabelPath)
    x,y = 0,0
    #Get x
    path = "/annotation/object/point/x1"
    result = imageXMLFile.xpath(path)
    for r in result:
        x = r.text
    #Get y
    path = "/annotation/object/point/y1" 
    result = imageXMLFile.xpath(path)
    for r in result:
        y = r.text
    return f'{x},{y}.png'


# In[31]:


getCoord(imageLabelPath)


# In[50]:





# In[34]:


def rename(image,label):
    srcPath = imagesPath + '/' + image
    destPath = imagesPath + '/' + label
    os.rename(srcPath, destPath)


# In[51]:


for image in enumerate(images):
    imageLabel = image[:-4]+'.xml'
    if imageLabel in xmls:
        imageLabelPath = xmlsPath +"/" + imageLabel
        lable = getLabel(imageLabelPath)
        rename(image,lable)


# In[ ]:



