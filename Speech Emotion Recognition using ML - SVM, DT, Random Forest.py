#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/blhprasanna99/speech_emotion_detection/blob/master/speechemotion_ml_algorithms.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


#get_ipython().system('pip install soundfile')


# In[ ]:


import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

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


# In[ ]:


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X, sr = librosa.load(sound_file, dtype="float32")
        frame_length = int(sr * 0.02)
        hop_length = int(sr * 0.01)
        #X = sound_file.read(dtype="float32")
        #sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X, n_fft=frame_length, hop_length=hop_length))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sr).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


# In[ ]:


def load_data(test_size=0.2):
    X, y = [], []
    try :
      for file in glob.glob("../Desktop/研究/example-data/*.wav"):
          # get the base name of the audio file
          basename = os.path.basename(file)
          print(basename)
          # get the emotion label
          emotion = int2emotion[basename.split("-")[0]]
          # we allow only AVAILABLE_EMOTIONS we set
          #if emotion not in AVAILABLE_EMOTIONS:
              #continue
          # extract speech features
          features = extract_feature(file, mfcc=True, mel=True)
          #features = extract_feature(file, mel=True)
          #features = extract_feature(file, mfcc=True)
          #features = extract_feature(file, mfcc=True, chroma=True, mel=True)
          # add to data
          X.append(features)
          y.append(emotion)
          #print(X,y)
    except :
         pass
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

def load_data2(clf,kf,test_size=0.2):
    X, y = [], []
    try :
      for file in glob.glob("../Desktop/研究/test_data/*.wav"):
          # get the base name of the audio file
          basename = os.path.basename(file)
          print(basename)
          # get the emotion label
          emotion = int2emotion[basename.split("-")[0]]
          # we allow only AVAILABLE_EMOTIONS we set
          #if emotion not in AVAILABLE_EMOTIONS:
              #continue
          # extract speech features
          features = extract_feature(file, mfcc=True, mel=True)
          #features = extract_feature(file, mel=True)
          #features = extract_feature(file, mfcc=True)
          #features = extract_feature(file, mfcc=True, chroma=True, mel=True)
          # add to data
          X.append(features)
          y.append(emotion)
          #print(X,y)
    except :
         pass
    # split the data to training and testing and return it
    return cross_val_score(clf, np.array(X), y, cv=kf)

# In[ ]:


X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])


# **DECISION TREE**

# In[ ]:
print("DECISION TREE")

from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 

print(accuracy_score(y_true=y_test,y_pred=dtree_predictions))
print(classification_report(y_test,dtree_predictions)) 
# creating a confusion matrix 
print(confusion_matrix(y_test, dtree_predictions) )


# In[ ]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


dtree_model = DecisionTreeClassifier(max_depth = 6).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 

print(accuracy_score(y_true=y_test,y_pred=dtree_predictions))
print(classification_report(y_test,dtree_predictions)) 
# creating a confusion matrix 
print(confusion_matrix(y_test, dtree_predictions) )


# DT
print("DT")
# In[ ]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


dtree_model = DecisionTreeClassifier(max_depth = 9,random_state=0).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 

print(accuracy_score(y_true=y_test,y_pred=dtree_predictions))
print(classification_report(y_test,dtree_predictions)) 
# creating a confusion matrix 
print(confusion_matrix(y_test, dtree_predictions) )


# **SUPPORT VECTOR MACHINE**
print("SUPPORT VECTOR MACHINE")
# In[ ]:

print("カーネルの種類: 線型")
from sklearn.svm import SVC

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_model_linear = OneVsRestClassifier(svm_model_linear).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

print(accuracy_score(y_true=y_test,y_pred=svm_predictions))
print(classification_report(y_test,svm_predictions)) 
# creating a confusion matrix 
print(confusion_matrix(y_test, svm_predictions) )

print(y_test)
print(X_test.shape)
#print(y_test.shape)
print(type(X_test))
#print(type(y_test))
print("svm_predictions",svm_predictions)

#交差検証
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#KFoldの設定
kf = KFold(n_splits=5, shuffle=True, random_state=1)
#交差検証を行う
scores = load_data2(svm_model_linear, kf=kf)
#各分割におけるスコア
print('Cross-Validation scores :{}'.format(scores))
#スコアの平均値
import numpy as np
print('Average score :{}'.format(np.mean(scores)))

y_test_int = []
for i in range(len(y_test)):
        if (y_test[i] == "happy") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(0)
        elif (y_test[i] == "sad") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(1)
        elif (y_test[i] == "annoy") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(2)
        elif (y_test[i] == "happy") and (y_test[i] != svm_predictions[i]):
                y_test_int.append(3)
        elif (y_test[i] == "sad") and (y_test[i] != svm_predictions[i]):
                y_test_int.append(4)
        else:
                y_test_int.append(5)
#        if (y_test[i] == "happy") and (y_test[i] == svm_predictions[i]):
#                y_test_int.append(0)
#        elif (y_test[i] == "sad") and (y_test[i] == svm_predictions[i]):
#                y_test_int.append(1)
#        elif (y_test[i] == "happy") and (y_test[i] != svm_predictions[i]):
#                y_test_int.append(2)
#        else:
#                y_test_int.append(3)
#

print(y_test_int)

from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X_test)

#結果の可視化
#t_SNE
import matplotlib.pyplot as plt
colors = ['#FF4B00', '#F6AA00', '#FFF100', '#005AFF', '#03AF7A', '#4DC4FF']
#colors = ['red', 'blue', 'green', 'orange', 'pink', 'purple']
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_test)):
        plt.text(
            X_tsne[i, 0],
            X_tsne[i, 1],
            str(y_test_int[i]),
            color = colors[y_test_int[i]]
            )
plt.savefig("./outputs/SVM_linear_mfcc2.png")
svm_model = plt.show()

#mlxtend
#from mlxtend.plotting import plot_decision_regions
#import numpy as np
#plot_decision_regions(X_train, y_test_int, clf=SVC(), colors='red, blue')
# In[ ]:

print("カーネルの種類: ガウスカーネル")
from sklearn.svm import SVC 
svm_model_linear = SVC().fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 


print(accuracy_score(y_true=y_test,y_pred=svm_predictions))
print(classification_report(y_test,svm_predictions)) 
# creating a confusion matrix 
print(confusion_matrix(y_test, svm_predictions) )
print(y_test)
print(svm_predictions)

#交差検証
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#KFoldの設定
kf = KFold(n_splits=5, shuffle=True, random_state=1)
#交差検証を行う
scores = load_data2(svm_model_linear, kf=kf)
#各分割におけるスコア
print('Cross-Validation scores :{}'.format(scores))
#スコアの平均値
import numpy as np
print('Average score :{}'.format(np.mean(scores)))


y_test_int = []
for i in range(len(y_test)):
        if (y_test[i] == "happy") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(0)
        elif (y_test[i] == "sad") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(1)
        elif (y_test[i] == "annoy") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(2)
        elif (y_test[i] == "happy") and (y_test[i] != svm_predictions[i]):
                y_test_int.append(3)
        elif (y_test[i] == "sad") and (y_test[i] != svm_predictions[i]):
                y_test_int.append(4)
        else:
                y_test_int.append(5)

#        if (y_test[i] == "happy") and (y_test[i] == svm_predictions[i]):
#                y_test_int.append(0)
#        elif (y_test[i] == "sad") and (y_test[i] == svm_predictions[i]):
#                y_test_int.append(1)
#        elif (y_test[i] == "happy") and (y_test[i] != svm_predictions[i]):
#                y_test_int.append(2)
#        else:
#                y_test_int.append(3)

print(y_test_int)

from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X_test)

#結果の可視化
#t_SNE
import matplotlib.pyplot as plt
#colors = ['red', 'blue', 'green', 'orange']
#colors = ['red', 'blue', 'green', 'orange', 'pink', 'purple']
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
plt.savefig("./outputs/SVM_gause_mfcc2.png")
plt.show()

# **RANDOM FOREST**
print("RANDOM FOREST")
# In[ ]:


from sklearn.ensemble import RandomForestClassifier
  
 # create regressor object 
classifier = RandomForestClassifier(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
random_forest_model = classifier.fit(X_train, y_train)   

c_p = classifier.predict(X_test) 


print(accuracy_score(y_true=y_test,y_pred=c_p))
print(classification_report(y_test,c_p)) 
# creating a confusion matrix 
print(confusion_matrix(y_test,c_p) )

#交差検証
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#KFoldの設定
kf = KFold(n_splits=5, shuffle=True, random_state=1)
#交差検証を行う
scores = load_data2(random_forest_model, kf=kf)
#各分割におけるスコア
print('Cross-Validation scores :{}'.format(scores))
#スコアの平均値
import numpy as np
print('Average score :{}'.format(np.mean(scores)))


y_test_int = []
for i in range(len(y_test)):
        if (y_test[i] == "happy") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(0)
        elif (y_test[i] == "sad") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(1)
        elif (y_test[i] == "annoy") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(2)
        elif (y_test[i] == "happy") and (y_test[i] != svm_predictions[i]):
                y_test_int.append(3)
        elif (y_test[i] == "sad") and (y_test[i] != svm_predictions[i]):
                y_test_int.append(4)
        else:
                y_test_int.append(5)

#        if (y_test[i] == "happy") and (y_test[i] == svm_predictions[i]):
#                y_test_int.append(0)
#        elif (y_test[i] == "sad") and (y_test[i] == svm_predictions[i]):
#                y_test_int.append(1)
#        elif (y_test[i] == "happy") and (y_test[i] != svm_predictions[i]):
#                y_test_int.append(2)
#        else:
#                y_test_int.append(3)

print(y_test_int)

from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X_test)

#結果の可視化
#t_SNE
import matplotlib.pyplot as plt
#colors = ['red', 'blue', 'green', 'orange']
#colors = ['red', 'blue', 'green', 'orange', 'pink', 'purple']
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
plt.savefig("./outputs/randomforest_mfcc2.png")
plt.show()

from sklearn.ensemble import RandomForestClassifier
  
 # create regressor object 
classifier = RandomForestClassifier(n_estimators = 20,random_state = 0) 
  
# fit the regressor with x and y data 
random_forest_model = classifier.fit(X_train, y_train)   

c_p = classifier.predict(X_test) 


print(accuracy_score(y_true=y_test,y_pred=c_p))
print(classification_report(y_test,c_p)) 
# creating a confusion matrix 
print(confusion_matrix(y_test,c_p) )

#交差検証
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#KFoldの設定
kf = KFold(n_splits=5, shuffle=True, random_state=1)
#交差検証を行う
scores = load_data2(random_forest_model, kf=kf)
#各分割におけるスコア
print('Cross-Validation scores :{}'.format(scores))
#スコアの平均値
import numpy as np
print('Average score :{}'.format(np.mean(scores)))


y_test_int = []
for i in range(len(y_test)):
        if (y_test[i] == "happy") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(0)
        elif (y_test[i] == "sad") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(1)
        elif (y_test[i] == "annoy") and (y_test[i] == svm_predictions[i]):
                y_test_int.append(2)
        elif (y_test[i] == "happy") and (y_test[i] != svm_predictions[i]):
                y_test_int.append(3)
        elif (y_test[i] == "sad") and (y_test[i] != svm_predictions[i]):
                y_test_int.append(4)
        else:
                y_test_int.append(5)

#        if (y_test[i] == "happy") and (y_test[i] == svm_predictions[i]):
#                y_test_int.append(0)
#        elif (y_test[i] == "sad") and (y_test[i] == svm_predictions[i]):
#                y_test_int.append(1)
#        elif (y_test[i] == "happy") and (y_test[i] != svm_predictions[i]):
#                y_test_int.append(2)
#        else:
#                y_test_int.append(3)

print(y_test_int)

from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X_test)

#結果の可視化
#t_SNE
import matplotlib.pyplot as plt
#colors = ['red', 'blue', 'green', 'orange']
#colors = ['red', 'blue', 'green', 'orange', 'pink', 'purple']
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
plt.savefig("./outputs/randomforest_mfcc-2_2.png")
plt.show()

