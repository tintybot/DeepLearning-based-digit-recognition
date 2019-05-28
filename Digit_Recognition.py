#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
class PREPROCESSING:
  
    def __init__(self,path,height,width):
        if not path.endswith('/'):
            path           = path + "/"
        self.path      = path
        self.height    = height
        self.width     = width
        self.type      = type
        self.onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]


    def prepare_matrix(self,pathx):
        path          = self.path + pathx
        img           = cv2.imread(path)
        img           = cv2.resize(img,(self.width,self.height), interpolation = cv2.INTER_CUBIC)
        img = np.reshape(img,(1,self.width,self.height,3))
        return (img)

    def build_matrix(self):
        counter = 0
        self.labels  = np.empty([len(self.onlyfiles)],dtype="int32")
        for file in self.onlyfiles:
            q = self.prepare_matrix(file)
            if (counter != 0):
                print(str(q.shape) + str(self.global_matrix.shape))
                self.global_matrix = np.concatenate([self.global_matrix,q],axis=0)
            else:
                self.global_matrix = q
            dash          = file.rfind("_")
            dot           = file.rfind(".")
            classtype     = file[dash+1:dot]
            self.labels[counter] = classtype
            counter = counter + 1
            
    


# In[3]:


from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,LSTM
from keras.optimizers import SGD
from keras.datasets import cifar10
import time
K.set_image_dim_ordering('tf')


x=PREPROCESSING("C:\\Users\\KIIT\\Downloads\\DEEPLEARNING_PROJECT\\Digit_Recog\\training_data",50,50)
x.build_matrix()
y = np_utils.to_categorical(x.labels)
x = x.global_matrix
x = x.astype('float32')/255


x_valid=PREPROCESSING("C:\\Users\\KIIT\\Downloads\\DEEPLEARNING_PROJECT\\Digit_Recog\\validating_data",50,50)
x_valid.build_matrix()
y_valid = np_utils.to_categorical(x_valid.labels)
x_valid = x_valid.global_matrix
x_valid= x_valid.astype('float32')/255



from tensorflow.keras.callbacks import TensorBoard
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
Sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# In[ ]:


dense_layers=[3]
layer_sizes=[128]
conv_layers=[2]
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME="{}-conv-{}-nodes{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))
            tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))
            print(NAME)
            model=Sequential()
            model.add(Conv2D(layer_size,(3,3),input_shape=(50,50,3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            #...
            for l in range (conv_layer):
                model.add(Conv2D(layer_size, (3,3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                #...
            model.add(Dropout(0.25))
            model.add(Flatten())
            for l in range (dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                model.add(Dropout(0.25))
                #...
            model.add(Dense(10))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
            model.fit(x,y,epochs=80,validation_split=0.1,validation_data=(x_valid,y_valid),callbacks=[tensorboard])
model.save("digit_recog.h5")
#model is ready
#load the model and test on images

# In[ ]:




