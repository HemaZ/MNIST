from keras.layers import Conv2D,Dense,Flatten,AveragePooling2D
from keras.models import Sequential
import numpy as np 
from keras.utils import to_categorical
import matplotlib as plt
from keras.losses import categorical_crossentropy
 

imgrow=28
imgcols=28
imgclass=10

def read_data(data):
    y=data[:,0]
    yout=to_categorical(y,imgclass)
    num_iamges=data.shape[0]
    x=data[:,1:]
    xout=x.reshape(num_iamges,imgrow,imgcols,1)
    xout=xout/255
    return xout,yout    


traindata=np.loadtxt('train.csv',skiprows=1,delimiter=',')
x,y=read_data(traindata)

mymodel=Sequential()
mymodel.add(Conv2D(12,kernel_size=(3,3),activation='relu',input_shape=(img_rows,img_cols,1)))
mymodel.add(AveragePooling2D(pool_size=(1,1)))      
mymodel.add(Conv2D(12,kernel_size=(3,3),activation='relu'))
mymodel.add(AveragePooling2D(pool_size=(1,1)))
mymodel.add(Conv2D(12,kernel_size=(3,3),activation='relu'))
mymodel.add(Flatten())
mymodel.add(Dense(100,activation='relu'))
mymodel.add(Dense(num_classes,activation='softmax'))

mymodel.compile(loss = keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
mymodel.fit(x,y,batch_size=100,epochs=4,validation_split=0.2)
