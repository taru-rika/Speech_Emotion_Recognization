#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/blhprasanna99/speech_emotion_detection/blob/master/emotion_spectogram_CNN_2D.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 

# In[1]:


#get_ipython().system('pip install soundfile')


# In[2]:


import cv2 #to preprocess images(open cv)
import os #to access file directories
import numpy as np # to deal with multidimensional matrices
from random import shuffle 
from tqdm import tqdm
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf 
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression

'''Setting up the env'''
  

IMG_SIZE = 150
LR = 1e-3


# In[ ]:


import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split

# all emotions on RAVDESS dataset
int2emotion = {
    "01": "happy",
    "02": "sad",
    "03": "annoy",
}
#int2emotion = {
#    "01": "neutral",
#    "02": "calm",
#    "03": "happy",
#    "04": "sad",
#    "05": "angry",
#    "06": "fearful",
#    "07": "disgust",
#    "08": "surprised"
#}

# we allow only these emotions
AVAILABLE_EMOTIONS = {
    "sad",
    "happy",
    "annoy",
}
#AVAILABLE_EMOTIONS = {
#    "angry",
#    "sad",
#    "neutral",
#    "happy"
#}


# In[4]:
#print(int2emotion)

import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display

#to play audio
import IPython.display as ipd

audio_fpath = "../Desktop/研究/test_data"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))


# In[5]:


#x, sr = librosa.load('/content/drive/My Drive/wav/Actor_01/03-01-01-01-01-01-01.wav', sr=44100)
#
#print(type(x), type(sr))
#print(x.shape, sr)


## In[6]:
#
#
#plt.figure(figsize=(14, 5))
#librosa.display.waveplot(x, sr=sr)
#
#
## In[7]:
#
#
#X = librosa.stft(x)
#Xdb = librosa.amplitude_to_db(abs(X))
#plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#plt.colorbar()
#
#
## In[8]:
#
#
#plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
#plt.colorbar()
#

# In[ ]:


def label_img(emotion):
    # DIY One hot encoder
    if emotion == 'happy': return [1, 0,0,0]
    elif emotion == 'sad': return [0, 1,0,0]
    elif emotion=='annoy' : return [0,0, 1,0]
#    elif emotion=='neutral' : return [0,0,0,1]


# In[ ]:


training_data=[]
# loading the training data 
def create_train_data(): 
  try :
    for file in glob.glob("../Desktop/研究/test_data/*.png"):
          # get the base name of the audio file
          basename = os.path.basename(file)
          print(basename)
          img=file
            # get the emotion label
          emotion = int2emotion[basename.split("-")[0]]
                  # we allow only AVAILABLE_EMOTIONS we set
          #if emotion not in AVAILABLE_EMOTIONS:
          #  continue
          label = label_img(emotion) 
          print(emotion)
          #path = os.path.join(file, img)
          path = file
          #print(file)
          #print(img)
                # loading the image from the path and then converting them into 
                # greyscale for easier covnet prob 
          img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
          img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
          
                # resizing the image for processing them in the covnet 
          img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
          img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
          print(np.array(img))
                # final step-forming the training data list with numpy array of the images 
          training_data.append([np.array(img), np.array(label)]) 
          
  except :
    pass
  return training_data


# In[11]:


m=create_train_data()
print(m)


# In[12]:


#t=np.array(training_data)
#print(t)


# In[13]:


#for img in training_data[:10] :
#    plt.imshow(img[0])
#    plt.show()

# In[14]:


tf.compat.v1.reset_default_graph() 
convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.8) 
  
convnet = fully_connected(convnet, 4, activation ='softmax') 
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
      loss ='categorical_crossentropy', name ='targets') 
  
model = tflearn.DNN(convnet, tensorboard_dir ='log') 


# In[15]:


X = np.array([img[0] for img in m]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
Y = [img[1] for img in m] 
'''Fitting the data into our model'''
#model_name='cnn_model'
# epoch = 5 taken 
#model.fit({'input': np.array(X_train)}, {'targets': y_train}, n_epoch = 10,run_id=model_name) 


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)


# In[17]:


model_name='cnn_model'
# epoch = 50 taken 
model.fit({'input': X_train}, {'targets': y_train}, n_epoch = 500,snapshot_step=300,show_metric=True,run_id=model_name)  


# In[ ]:


import sklearn.metrics as  metrics
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix


# In[20]:

# CNNでの特徴量抽出
cnn_features_train = model.predict(X_train)
cnn_features_test = model.predict(X_test)
##y_pred = model.predict(X_test)
#report=metrics.classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1)) 
#print(report)
##print(accuracy_score(y_true=Y,y_pred=y_predict))
##print(confusion_matrix(y_test, y_pred) )
#matrix = metrics.confusion_matrix(np.argmax(y_test,axis=1) ,np.argmax(y_pred,axis=1))
#print(matrix)
#metrics.accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))*100 
print(cnn_features_train.shape);
#print(y_train.shape);
# SVMモデルの定義と訓練
svm_model = SVC(kernel = 'linear')
svm_model.fit(cnn_features_train, np.argmax(y_train, axis=1))
#svm_model.fit(cnn_features_train.flatten(), y_train)

# テストデータでの評価
svm_predictions = svm_model.predict(cnn_features_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), svm_predictions)
print("Accuracy :", accuracy)

#結果の可視化
print(y_test)
print(svm_predictions)
y_test_array = []
for i in range(len(y_test)):
	if((y_test[i] == np.array([1,0,0,0])).all()):
		y_test_array.append(0) #happy
	elif((y_test[i] == np.array([0,1,0,0])).all()):
                y_test_array.append(1) #sad
	else:
                y_test_array.append(2)

y_test_int = []
for i in range(len(y_test)):
        if (y_test_array[i] == 0) and (y_test_array[i] == svm_predictions[i]): #happy
                y_test_int.append(0)
        elif (y_test_array[i] == 1) and (y_test_array[i] == svm_predictions[i]): #sad
                y_test_int.append(1)
        elif (y_test_array[i] == 2) and (y_test_array[i] == svm_predictions[i]): #annoy
                y_test_int.append(2)
        elif (y_test_array[i] == 0) and (y_test_array[i] != svm_predictions[i]): #happy×
                y_test_int.append(3)
        elif (y_test_array[i] == 1) and (y_test_array[i] != svm_predictions[i]): #sad×
                y_test_int.append(4)
        else: #annoy×
                y_test_int.append(5)

#        if (y_test_array[i] == 0) and (y_test_array[i] == svm_predictions[i]): #happy
#                y_test_int.append(0)
#        elif (y_test_array[i] == 1) and (y_test_array[i] == svm_predictions[i]): #sad
#                y_test_int.append(1)
#        elif (y_test_array[i] == 0) and (y_test_array[i] != svm_predictions[i]): #happy×
#                y_test_int.append(2)
#        else: #sad×
#                y_test_int.append(3)


#print(y_test_int)

#t_SNE
print(X_test)
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(cnn_features_test)

import matplotlib.pyplot as plt
#colors = ['red', 'blue', 'green', 'orange']
colors = ['#FF4B00', '#F6AA00', '#FFF100', '#005AFF', '#03AF7A', '#4DC4FF']
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_test)):
        plt.text(
            X_tsne[i, 0],
            X_tsne[i, 1],
            str(y_test_int[i]),
            color = colors[y_test_int[i]]
            )
plt.savefig("./outputs/CNN2D_to_SVM.png")
svm_model = plt.show()

print("カーネルの種類: ガウスカーネル")
from sklearn.svm import SVC 
svm_model_gause = SVC().fit(cnn_features_train, np.argmax(y_train, axis=1)) 
svm_predictions = svm_model_gause.predict(cnn_features_test) 

# テストデータでの評価
accuracy = accuracy_score(np.argmax(y_test, axis=1), svm_predictions)
print("Accuracy :", accuracy)

#結果の可視化
print(y_test)
print(svm_predictions)
y_test_array = []
for i in range(len(y_test)):
	if((y_test[i] == np.array([1,0,0,0])).all()):
		y_test_array.append(0) #happy
	if((y_test[i] == np.array([0,1,0,0])).all()):
                y_test_array.append(1) #sad
	else:
		y_test_array.append(2) #annoy

y_test_int = []
for i in range(len(y_test)):
        if (y_test_array[i] == 0) and (y_test_array[i] == svm_predictions[i]): #happy
                y_test_int.append(0)
        elif (y_test_array[i] == 1) and (y_test_array[i] == svm_predictions[i]): #sad
                y_test_int.append(1)
        elif (y_test_array[i] == 2) and (y_test_array[i] == svm_predictions[i]): #annoy
                y_test_int.append(2)
        elif (y_test_array[i] == 0) and (y_test_array[i] != svm_predictions[i]): #happy×
                y_test_int.append(3)
        elif (y_test_array[i] == 1) and (y_test_array[i] != svm_predictions[i]): #sad×
                y_test_int.append(4)
        else: #annoy×
                y_test_int.append(5)

#        if (y_test_array[i] == 0) and (y_test_array[i] == svm_predictions[i]): #happy
#                y_test_int.append(0)
#        elif (y_test_array[i] == 1) and (y_test_array[i] == svm_predictions[i]): #sad
#                y_test_int.append(1)
#        elif (y_test_array[i] == 0) and (y_test_array[i] != svm_predictions[i]): #happy×
#                y_test_int.append(2)
#        else: #sad×
#                y_test_int.append(3)
#
#print(y_test_int)

#t_SNE
print(X_test)
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(cnn_features_test)

import matplotlib.pyplot as plt
#colors = ['red', 'blue', 'green', 'orange']
colors = ['#FF4B00', '#F6AA00', '#FFF100', '#005AFF', '#03AF7A', '#4DC4FF']
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_test)):
        plt.text(
            X_tsne[i, 0],
            X_tsne[i, 1],
            str(y_test_int[i]),
            color = colors[y_test_int[i]]
            )
plt.savefig("./outputs/CNN2D_to_SVMgause.png")
svm_model = plt.show()

# **RANDOM FOREST**
print("RANDOM FOREST")
# In[ ]:


from sklearn.ensemble import RandomForestClassifier

 # create regressor object 
classifier = RandomForestClassifier(n_estimators = 100, random_state = 0)

# fit the regressor with x and y data 
random_forest_model = classifier.fit(cnn_features_train, np.argmax(y_train,axis=1))

# テストデータでの評価
c_p = classifier.predict(cnn_features_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), c_p)
print("Accuracy :", accuracy)

#結果の可視化
print(y_test)
print(c_p)
y_test_array = []
for i in range(len(y_test)):
	if((y_test[i] == np.array([1,0,0,0])).all()):
		y_test_array.append(0) #happy
	if((y_test[i] == np.array([0,1,0,0])).all()):
                y_test_array.append(1) #sad
	else:
		y_test_array.append(2) #annoy

y_test_int = []
for i in range(len(y_test)):
        if (y_test_array[i] == 0) and (y_test_array[i] == svm_predictions[i]): #happy
                y_test_int.append(0)
        elif (y_test_array[i] == 1) and (y_test_array[i] == svm_predictions[i]): #sad
                y_test_int.append(1)
        elif (y_test_array[i] == 2) and (y_test_array[i] == svm_predictions[i]): #annoy
                y_test_int.append(2)
        elif (y_test_array[i] == 0) and (y_test_array[i] != svm_predictions[i]): #happy×
                y_test_int.append(3)
        elif (y_test_array[i] == 1) and (y_test_array[i] != svm_predictions[i]): #sad×
                y_test_int.append(4)
        else: #annoy×
                y_test_int.append(5)


#        if (y_test_array[i] == 0) and (y_test_array[i] == svm_predictions[i]): #happy
#                y_test_int.append(0)
#        elif (y_test_array[i] == 1) and (y_test_array[i] == svm_predictions[i]): #sad
#                y_test_int.append(1)
#        elif (y_test_array[i] == 0) and (y_test_array[i] != svm_predictions[i]): #happy×
#                y_test_int.append(2)
#        else: #sad×
#                y_test_int.append(3)
#
#print(y_test_int)

#t_SNE
print(X_test)
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(cnn_features_test)

import matplotlib.pyplot as plt
#colors = ['red', 'blue', 'green', 'orange']
colors = ['#FF4B00', '#F6AA00', '#FFF100', '#005AFF', '#03AF7A', '#4DC4FF']
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_test)):
        plt.text(
            X_tsne[i, 0],
            X_tsne[i, 1],
            str(y_test_int[i]),
            color = colors[y_test_int[i]]
            )
plt.savefig("./outputs/CNN2D_to_RDF.png")
svm_model = plt.show()


## SECOND
## 
#
## In[21]:
#
#
#tf.compat.v1.reset_default_graph() 
#convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
#  
#convnet = conv_2d(convnet, 32, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#  
#convnet = conv_2d(convnet, 64, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5)
#
#convnet = conv_2d(convnet, 64, 5, activation ='relu') # added_2
#convnet = max_pool_2d(convnet, 5)
#  
#convnet = conv_2d(convnet, 128, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#convnet = dropout(convnet, 0.8)    # added
#  
#convnet = conv_2d(convnet, 64, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#  
#convnet = conv_2d(convnet, 32, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#
#convnet = conv_2d(convnet, 32, 5, activation ='relu') #added_1
#convnet = max_pool_2d(convnet, 5) 
#
#convnet = fully_connected(convnet, 1024, activation ='relu') 
#convnet = dropout(convnet, 0.8) 
#  
#convnet = fully_connected(convnet, 4, activation ='softmax') 
#convnet = regression(convnet, optimizer ='rmsprop', learning_rate = LR, 
#      loss ='categorical_crossentropy', name ='targets') 
#
#m = tflearn.DNN(convnet, tensorboard_dir ='log')
#
#
## In[25]:
#
#
#model_name='cnn'
## epoch = 50 taken 
#m.fit({'input': X_train}, {'targets': y_train}, n_epoch = 500,snapshot_step=300,show_metric=True,run_id=model_name)  
#
#
## In[26]:
#
#
#y_pred = m.predict(X_test)
#report=metrics.classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1)) 
#print(report)
##print(accuracy_score(y_true=Y,y_pred=y_predict))
##print(confusion_matrix(y_test, y_pred) )
#matrix = metrics.confusion_matrix(np.argmax(y_test,axis=1) ,np.argmax(y_pred,axis=1))
#print(matrix)
#metrics.accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))*100 
#
#
## Third
#
## In[ ]:
#
#
#tf.compat.v1.reset_default_graph() 
#convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
#  
#convnet = conv_2d(convnet, 32, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#  
#convnet = conv_2d(convnet, 64, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5)
#
#convnet = conv_2d(convnet, 128, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#
#  
#convnet = conv_2d(convnet, 64, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#  
#convnet = conv_2d(convnet, 32, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#
#
#convnet = fully_connected(convnet, 1024, activation ='relu') 
#convnet = dropout(convnet, 0.8) 
#  
#convnet = fully_connected(convnet, 4, activation ='softmax') 
#convnet = regression(convnet, optimizer ='adagrad', learning_rate = LR, 
#      loss ='categorical_crossentropy', name ='targets') 
#
#n = tflearn.DNN(convnet, tensorboard_dir ='log')
#
#
## In[28]:
#
#
#model_name='c'
## epoch = 50 taken 
#n.fit({'input': X_train}, {'targets': y_train}, n_epoch = 500,snapshot_step=300,show_metric=True,run_id=model_name)  
#
#
## In[29]:
#
#
#y_pred = n.predict(X_test)
#report=metrics.classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1)) 
#print(report)
##print(accuracy_score(y_true=Y,y_pred=y_predict))
##print(confusion_matrix(y_test, y_pred) )
#matrix = metrics.confusion_matrix(np.argmax(y_test,axis=1) ,np.argmax(y_pred,axis=1))
#print(matrix)
#metrics.accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))*100 
#
#
## Fourth
#
## In[ ]:
#
#
#LR1=0.005
#tf.compat.v1.reset_default_graph() 
#convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
#  
#convnet = conv_2d(convnet, 32, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#  
#convnet = conv_2d(convnet, 64, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5)
#
#convnet = conv_2d(convnet, 128, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#
#  
#convnet = conv_2d(convnet, 64, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#  
#convnet = conv_2d(convnet, 32, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#
#
#convnet = fully_connected(convnet, 1024, activation ='relu') 
#convnet = dropout(convnet, 0.8) 
#  
#convnet = fully_connected(convnet, 4, activation ='softmax') 
#convnet = regression(convnet, optimizer ='adagrad', learning_rate = LR1, 
#      loss ='categorical_crossentropy', name ='targets') 
#
#o = tflearn.DNN(convnet, tensorboard_dir ='log')
#
#
## In[60]:
#
#
#model_name='cn'
## epoch = 50 taken 
#o.fit({'input': X_train}, {'targets': y_train}, n_epoch = 500,snapshot_step=300,show_metric=True,run_id=model_name)  
#
#
## In[61]:
#
#
#y_pred = o.predict(X_test)
#report=metrics.classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1)) 
#print(report)
##print(accuracy_score(y_true=Y,y_pred=y_predict))
##print(confusion_matrix(y_test, y_pred) )
#matrix = metrics.confusion_matrix(np.argmax(y_test,axis=1) ,np.argmax(y_pred,axis=1))
#print(matrix)
#metrics.accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))*100 
#
#
## Fifth
#
## In[ ]:
#
#
#LR1=5e-05
#tf.compat.v1.reset_default_graph() 
#convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
#  
#convnet = conv_2d(convnet, 32, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#  
#convnet = conv_2d(convnet, 64, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5)
#
#convnet = conv_2d(convnet, 128, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#
#  
#convnet = conv_2d(convnet, 64, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#  
#convnet = conv_2d(convnet, 32, 5, activation ='relu') 
#convnet = max_pool_2d(convnet, 5) 
#
#
#convnet = fully_connected(convnet, 1024, activation ='relu') 
#convnet = dropout(convnet, 0.8) 
#  
#convnet = fully_connected(convnet, 4, activation ='softmax') 
#convnet = regression(convnet, optimizer ='adam', learning_rate = LR1, 
#      loss ='categorical_crossentropy', name ='targets') 
#
#p = tflearn.DNN(convnet, tensorboard_dir ='log')
#
#
## In[88]:
#
#
#model_name='cp'
## epoch = 50 taken 
#p.fit({'input': X_train}, {'targets': y_train}, n_epoch = 500,snapshot_step=300,show_metric=True,run_id=model_name)  
#
#
## In[89]:
#
#
#y_pred = p.predict(X_test)
#report=metrics.classification_report(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1)) 
#print(report)
##print(accuracy_score(y_true=Y,y_pred=y_predict))
##print(confusion_matrix(y_test, y_pred) )
#matrix = metrics.confusion_matrix(np.argmax(y_test,axis=1) ,np.argmax(y_pred,axis=1))
#print(matrix)
#metrics.accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))*100 
#
#
## In[ ]:
#
#
#
#
