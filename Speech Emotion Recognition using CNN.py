#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/blhprasanna99/speech_emotion_detection/blob/master/CNN_SpeechEmotion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


# get_ipython().system('pip install soundfile')


# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


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
#


# we allow only these emotions
AVAILABLE_EMOTIONS = {
    "sad",
    "happy"
}
#AVAILABLE_EMOTIONS = {
#    "angry",
#    "sad",
#    "neutral",
#    "happy"
#}
#

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
        #X = sound_file.read(dtype="float32")
        #X, sr = librosa.load(sound_file, sr=None, dtype="float32")
        X, sr = librosa.load(sound_file, dtype="float32")
        frame_length = int(sr * 0.02)
        hop_length = int(sr * 0.01)
        #sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X,n_fft=frame_length, hop_length=hop_length))
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
            #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


# In[ ]:


def load_data(test_size=0.0):
    X, y = [], []
    try :
      for file in glob.glob("../Desktop/研究/example-data/*.wav"):
      #for file in glob.glob("/content/drive/My Drive/wav/Actor_*/*.wav"):
          # get the base name of the audio file
          basename = os.path.basename(file)
          # get the emotion label
          emotion = int2emotion[basename.split("-")[0]]
          # we allow only AVAILABLE_EMOTIONS we set
          #if emotion not in AVAILABLE_EMOTIONS:
              #continue
	  #l={'happy':0.0,'sad':1.0,'neutral':3.0,'angry':4.0}
          l={"happy":0.0,"sad":1.0}
          #print("l[emotion]:",l[emotion])
          y.append(l[emotion])
          # extract speech features
          features = extract_feature(file, mfcc=True, mel=True)
          #features = extract_feature(file, mel=True)
          #features = extract_feature(file, mfcc=True)
          #features = extract_feature(file, mfcc=True, chroma=True, mel=True)
          # add to data
          X.append(features)
    except :
         pass
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=1)


# In[ ]:


X_train, X_test, y_train, y_test = load_data(test_size=0.25)

print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])


# In[ ]:


import numpy as np
X_train = np.asarray(X_train)
y_train= np.asarray(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)


# In[ ]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[ ]:


x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)


# In[ ]:


x_traincnn.shape,x_testcnn.shape


# In[ ]:


import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv1D(128, 5,padding='same',input_shape=(168,1)))        #1
#model.add(Conv1D(128, 5,padding='same',input_shape=(180,1)))        #1
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))



model.add(Conv1D(128, 5,padding='same',))                           #2
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(8))                                                 #3
model.add(Activation('softmax'))

from tensorflow.keras import optimizers
opt = optimizers.RMSprop(learning_rate=0.00005)
#opt = optimizers.rmsprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


cnnhistory=model.fit(x_traincnn, y_train, batch_size=20, epochs=500, validation_data=(x_testcnn, y_test))


# In[ ]:


#em=['happy','sad','neutral','angry']
em=['happy','sad']


# In[ ]:


predictions = (model.predict(x_testcnn)>0.5).astype("int32")
#predictions = model.predict_classes(x_testcnn)
n=predictions[1]
#print(em[int(n)])
print(n)


# In[ ]:


loss, acc = model.evaluate(x_testcnn, y_test)
#print("predictions",predictions)
#print("x_testcnn",x_testcnn)
#print("y_test",y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[ ]:
#結果の可視化
predictions_array = []
for i in range(len(predictions)):
	if((predictions[i] == np.array([1,0,0,0,0,0,0,0])).all()):
		predictions_array.append(0) #happy
	else:
		predictions_array.append(1) #sad

print(predictions_array)
print(y_test)
y_test_int = []
for i in range(len(y_test)):
        if (y_test[i] == 0) and (y_test[i] == predictions_array[i]): #happy
                y_test_int.append(0)
        elif (y_test[i] == 1) and (y_test[i] == predictions_array[i]): #sad
                y_test_int.append(1)
        elif (y_test[i] == 0) and (y_test[i] != predictions_array[i]): #happy×
                y_test_int.append(2)
        else: #sad×
                y_test_int.append(3)

#print(y_test_int)

#t_SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'blue', 'green', 'orange']
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_test)):
        plt.text(
            X_tsne[i, 0],
            X_tsne[i, 1],
            str(y_test_int[i]),
            color = colors[y_test_int[i]]
            )
plt.savefig("./outputs/CNN1D-1.png");
svm_model = plt.show()


from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
#predictions = model.predict_classes(x_testcnn)
#predictions = (model.predict(x_testcnn)>0.5).astype("int32")
#print(confusion_matrix(y_test,predictions))


# In[ ]:


##filename = "/content/drive/My Drive/wav/Actor_02/03-01-01-01-02-01-02.wav"
#filename = "../Desktop/研究/example-data/01-01-08-1.wav"
#    # record the file (start talking)
#    #record_to_file(filename)
#    # extract features and reshape it
##features = np.array(extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1))
#features = np.array(extract_feature(filename, mfcc=True).reshape(1, -1))
#    # predict
#f=np.expand_dims(features,axis=2)
#result = (model.predict(f)>0.5).astype("int32")[0]
#    # show the result !
#print("result :",result)
##print("result :",em[result])


# **Second**

# In[ ]:


import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

um = Sequential()
um.add(Conv1D(128, 5,padding='same',input_shape=(168,1)))#1
#um.add(Conv1D(128, 5,padding='same',input_shape=(40,1)))#1
#um.add(Conv1D(128, 5,padding='same',input_shape=(180,1)))#1
um.add(Activation('relu'))
um.add(Dropout(0.25))
um.add(MaxPooling1D(pool_size=(8)))

um.add(Conv1D(128, 5,padding='same',))                  #2
um.add(Activation('relu'))
um.add(MaxPooling1D(pool_size=(8)))
um.add(Dropout(0.25))

um.add(Conv1D(128, 5,padding='same',))                  #3
um.add(Activation('relu'))
um.add(Dropout(0.25))

um.add(Flatten())
um.add(Dense(8))                                        #4                      
um.add(Activation('softmax'))

from tensorflow.keras import optimizers
opt = optimizers.RMSprop(lr=0.00005)
#opt = optimizers.RMSprop(lr=0.00005,epsilon=None,rho=0.9,decay=0.0)


# In[ ]:


um.summary()


# In[ ]:


um.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


umhistory=um.fit(x_traincnn, y_train, batch_size=20, epochs=500, validation_data=(x_testcnn, y_test))


# In[ ]:


loss, acc = um.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[ ]:


from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
#predictions = (model.predict(x_testcnn)>0.5).astype("int32")
prediction_prob = um.predict(x_testcnn)
prediction = np.argmax(prediction_prob, axis=1)
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))

# In[ ]:
#結果の可視化
predictions_array = []
for i in range(len(predictions)):
	if((predictions[i] == np.array([1,0,0,0,0,0,0,0])).all()):
		predictions_array.append(0) #happy
	else:
		predictions_array.append(1) #sad

#print(predictions_array)
#print(y_test)
y_test_int = []
for i in range(len(y_test)):
        if (y_test[i] == 0) and (y_test[i] == predictions_array[i]): #happy
                y_test_int.append(0)
        elif (y_test[i] == 1) and (y_test[i] == predictions_array[i]): #sad
                y_test_int.append(1)
        elif (y_test[i] == 0) and (y_test[i] != predictions_array[i]): #happy×
                y_test_int.append(2)
        else: #sad×
                y_test_int.append(3)

#print(y_test_int)

#t_SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'blue', 'green', 'orange']
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_test)):
        plt.text(
            X_tsne[i, 0],
            X_tsne[i, 1],
            str(y_test_int[i]),
            color = colors[y_test_int[i]]
            )
plt.savefig("./outputs/CNN1D-2.png")
svm_model = plt.show()



# **Third**

# In[ ]:


import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

tm = Sequential()

tm.add(Conv1D(128, 5,padding='same',input_shape=(168,1)))#1
#tm.add(Conv1D(128, 5,padding='same',input_shape=(40,1)))#1
#tm.add(Conv1D(128, 5,padding='same',input_shape=(180,1)))#1
tm.add(Activation('relu'))
tm.add(Dropout(0.1))
tm.add(MaxPooling1D(pool_size=(8)))

tm.add(Conv1D(128, 5,padding='same',))                  #2
tm.add(Activation('relu'))
tm.add(MaxPooling1D(pool_size=(8)))
tm.add(Dropout(0.1))

tm.add(Conv1D(128, 5,padding='same',))                  #3
tm.add(Activation('relu'))
tm.add(Dropout(0.1))

tm.add(Flatten())
tm.add(Dense(8))                                        #4                      
tm.add(Activation('softmax'))

from tensorflow.keras import optimizers
opt = optimizers.RMSprop(lr=0.00005)
#opt = optimizers.RMSprop(lr=0.00005,epsilon=None,rho=0.9,decay=0.0)


# In[ ]:


tm.summary()


# In[ ]:


tm.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


tmhistory=tm.fit(x_traincnn, y_train, batch_size=20, epochs=500, validation_data=(x_testcnn, y_test))


# In[ ]:


loss, acc = tm.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[ ]:


from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
#predictions = (model.predict(x_testcnn)>0.5).astype("int32")
#predict = tm.predict_classes(x_testcnn)
predict_prob = um.predict(x_testcnn)
predict = np.argmax(predict_prob, axis=1)
print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))

# In[ ]:
#結果の可視化
predictions_array = []
for i in range(len(predictions)):
	if((predictions[i] == np.array([1,0,0,0,0,0,0,0])).all()):
		predictions_array.append(0) #happy
	else:
		predictions_array.append(1) #sad

#print(predictions_array)
#print(y_test)
y_test_int = []
for i in range(len(y_test)):
        if (y_test[i] == 0) and (y_test[i] == predictions_array[i]): #happy
                y_test_int.append(0)
        elif (y_test[i] == 1) and (y_test[i] == predictions_array[i]): #sad
                y_test_int.append(1)
        elif (y_test[i] == 0) and (y_test[i] != predictions_array[i]): #happy×
                y_test_int.append(2)
        else: #sad×
                y_test_int.append(3)

#print(y_test_int)

#t_SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'blue', 'green', 'orange']
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_test)):
        plt.text(
            X_tsne[i, 0],
            X_tsne[i, 1],
            str(y_test_int[i]),
            color = colors[y_test_int[i]]
            )
plt.savefig("./outputs/CNN1D-3.png")
svm_model = plt.show()


# **Fourth**

# In[ ]:


import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

sm = Sequential()

sm.add(Conv1D(128, 5,padding='same',input_shape=(168,1)))#1
#sm.add(Conv1D(128, 5,padding='same',input_shape=(40,1)))#1
#sm.add(Conv1D(128, 5,padding='same',input_shape=(180,1)))#1
sm.add(Activation('relu'))
sm.add(Dropout(0.1))
sm.add(MaxPooling1D(pool_size=(8)))

sm.add(Conv1D(128, 5,padding='same',))                  #2
sm.add(Activation('relu'))
sm.add(MaxPooling1D(pool_size=(8)))
sm.add(Dropout(0.1))

sm.add(Conv1D(128, 5,padding='same',))                  #3
sm.add(Activation('relu'))
sm.add(Dropout(0.1))

sm.add(Conv1D(128, 5,padding='same',))                  #4
sm.add(Activation('relu'))
sm.add(Dropout(0.1))

sm.add(Flatten())
sm.add(Dense(8))                                        #5                     
sm.add(Activation('softmax'))


from tensorflow.keras import optimizers
opt = keras.optimizers.RMSprop(lr=0.00005)
#opt = keras.optimizers.RMSprop(lr=0.00005,epsilon=None,rho=0.9,decay=0.0)


# In[ ]:


sm.summary()


# In[ ]:


sm.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


smhistory=sm.fit(x_traincnn, y_train, batch_size=20, epochs=500, validation_data=(x_testcnn, y_test))


# In[ ]:


from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
#p = (sm.predict(x_testcnn)>0.5).astype("int32")
#p = sm.predict_classes(x_testcnn)
p_prob = um.predict(x_testcnn)
p = np.argmax(p_prob, axis=1)
print(classification_report(y_test,p))
print(confusion_matrix(y_test,p))

# In[ ]:
#結果の可視化
predictions_array = []
for i in range(len(predictions)):
	if((predictions[i] == np.array([1,0,0,0,0,0,0,0])).all()):
		predictions_array.append(0) #happy
	else:
		predictions_array.append(1) #sad

print(predictions_array)
print(y_test)
y_test_int = []
for i in range(len(y_test)):
        if (y_test[i] == 0) and (y_test[i] == predictions_array[i]): #happy
                y_test_int.append(0)
        elif (y_test[i] == 1) and (y_test[i] == predictions_array[i]): #sad
                y_test_int.append(1)
        elif (y_test[i] == 0) and (y_test[i] != predictions_array[i]): #happy×
                y_test_int.append(2)
        else: #sad×
                y_test_int.append(3)

#print(y_test_int)

#t_SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'blue', 'green', 'orange']
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_test)):
        plt.text(
            X_tsne[i, 0],
            X_tsne[i, 1],
            str(y_test_int[i]),
            color = colors[y_test_int[i]]
            )
plt.savefig("./outputs/CNN1D-4.png")
svm_model = plt.show()



