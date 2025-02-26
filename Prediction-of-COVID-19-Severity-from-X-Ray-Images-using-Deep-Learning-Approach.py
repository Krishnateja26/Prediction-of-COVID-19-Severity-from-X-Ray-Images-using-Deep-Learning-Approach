#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import os 
import numpy as np 


datapath1='covid-chestxray-dataset-master' 
dataset_path='dataset' 

categories=os.listdir(dataset_path) 
print(categories) 

dataset=pd.read_csv(os.path.join(datapath1,'metadata.csv')) 
findings=dataset['finding'] 
image_names=dataset['filename'] 


# In[6]:


positives_index=np.concatenate((np.where(findings=='COVID19')[0],np.where(findings=='SARS')[0])) 
positive_image_names=image_names[positives_index] 


# In[7]:


import cv2 
                                         
for positive_image_name in positive_image_names: 
    image=cv2.imread(os.path.join(datapath1,'images',positive_image_name)) 
    try: 
        cv2.imwrite(os.path.join(dataset_path,categories[1],positive_image_name),image) 
    except Exception as e: 
        print(e) 


# In[8]:


datapath2='Coronahack-Chest-XRay-Dataset' 
                                         
dataset=pd.read_csv(os.path.join(datapath2,'Chest_xray_Corona_Metadata.csv')) 
findings=dataset['Label'] 
image_names=dataset['X_ray_image_name'] 


# In[9]:


negative_index=np.where(findings=='Normal')[0]


# In[10]:


negative_image_names=image_names[negative_index]


# In[11]:


for negative_image_name in negative_image_names:
    image=cv2.imread(os.path.join(datapath2,'images',negative_image_name)) 
    try: 
        cv2.imwrite(os.path.join(dataset_path,categories[0],negative_image_name),image) 
    except Exception as e: 
        print(e) 


# In[12]:


negative_image_names.shape 


# In[13]:


#DATA PREPROCESSING 

import cv2,os 
data_path='dataset' 
categories=os.listdir(data_path) 
labels=[i for i in range(len(categories))] 
                                         
label_dict=dict(zip(categories,labels)) #empty dictionary 
                                         
print(label_dict) 
print(categories) 
print(labels)


# In[14]:


img_size=100 
data=[] 
target=[]
                                         
for category in categories: 
    folder_path=os.path.join(data_path,category) 
    img_names=os.listdir(folder_path) 
                                         
    for img_name in img_names: 
        img_path=os.path.join(folder_path,img_name) 
        img=cv2.imread(img_path) 
                                         
        try: 
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)            
            #Coverting the image into gray scale 
            resized=cv2.resize(gray,(img_size,img_size)) 
            #resizing the gray scale into 100x100, since we need a fixed common size for all the images in the dataset 
            data.append(resized) 
            target.append(label_dict[category]) 
            #appending the image and the label(categorized) into the list (dataset) 
        except Exception as e: 
            print('Exception:',e) 
            #if any exception rasied, the exception will be printed here. And pass to the next image 


# In[15]:


import numpy as np 
                                         
data=np.array(data)/255.0 
data=np.reshape(data,(data.shape[0],img_size,img_size,1)) 
target=np.array(target) 

from tensorflow.keras.utils import to_categorical  # Import to_categorical directly

new_target = to_categorical(target)


# In[16]:


np.save('data',data) 
np.save('target',target)


# In[17]:


#TRAINING CNN MODEL 
                                         
import numpy as np 
                                         
data=np.load('data.npy') 
target=np.load('target.npy')
                                         


# In[18]:


from keras.models import Sequential,Model 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D,Activation,MaxPooling2D 
from keras.utils import normalize 
from keras.layers import Concatenate 
from keras import Input 
from keras.callbacks import ModelCheckpoint 
                                         
input_shape=data.shape[1:] #50,50,1 
inp=Input(shape=input_shape) 
convs=[] 
                                         
parrallel_kernels=[3,5,7] 
                                         
for  k in range(len(parrallel_kernels)):  
    conv=Conv2D(128,parrallel_kernels[k],padding='same',activation='relu',input_shape=input_shape,strides=1)(inp) 
    convs.append(conv) 
                                         
out = Concatenate()(convs) 
conv_model = Model(inp,out) 
                                         
model = Sequential() 
model.add(conv_model) 
                                         
model.add(Conv2D(64,(3,3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 
                                         
model.add(Conv2D(32,(3,3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 
                                         
model.add(Flatten()) 
model.add(Dropout(0.5)) 
model.add(Dense(128,activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(64,activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(2,input_dim=128,activation='softmax')) 
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy']) 
                                         
model.summary() 
                                         


# In[21]:


from sklearn.model_selection import train_test_split 
                                         
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1) 
                                         


# In[22]:


checkpoint=ModelCheckpoint('model-{epoch:03d}.model.keras',monitor='val_loss',verbose=0,save_best_only=True,mode='auto') 
history=model.fit(train_data,train_target,epochs=5,callbacks=[checkpoint],validation_split=0.1) 
                                         


# In[23]:


from matplotlib import pyplot as plt 
                                         
plt.plot(history.history['loss'],'r',label='training loss') 
plt.plot(history.history['val_loss'],label='validation loss') 
plt.xlabel('# epochs') 
plt.ylabel('loss') 
plt.legend() 
plt.show() 
                                          


# In[24]:


plt.plot(history.history['accuracy'],'r',label='training accuracy') 
plt.plot(history.history['val_accuracy'],label='validation accuracy') 
plt.xlabel('# epochs')
plt.ylabel('loss') 
plt.legend() 
plt.show()


# In[25]:


print(model.evaluate(test_data,test_target))                                        

