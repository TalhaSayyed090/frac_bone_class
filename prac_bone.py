import os
import cv2
import tensorflow as tf
import tensorflow
from tensorflow import keras
from keras import Sequential 
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_da = keras.utils.image_dataset_from_directory(
    directory='c:\\Users\\Sayyed_Talha_bacha\\Desktop\Frac_bon_classi\\bone_DataSet\\train',
    labels='inferred',
    label_mode='int',
    batch_size=50,
    image_size=(256,256)
)



test_da = keras.utils.image_dataset_from_directory(
    directory= "c:\\Users\\Sayyed_Talha_bacha\\Desktop\\Frac_bon_classi\\bone_DataSet\\val",
    labels='inferred',
    label_mode='int',
    batch_size=50,
    image_size=(256,256)
)


def process(image, label):
    image = tf.cast(image/255. ,tf.float32)
    return image, label


train_da = train_da.map(process)
test_da = test_da.map(process)




model = Sequential()

model.add(Conv2D(19,kernel_size=(3,3),padding='valid',activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2,padding='valid'))


model.add(Conv2D(42,kernel_size=(3,3),padding='valid',activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(92,activation='relu'))
model.add(Dense(52,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_da, epochs=3, validation_data=test_da)
loss, accuracy = model.evaluate(test_da)
print('the accuracy and loss values respectivaly of the model is  : ',accuracy*100, loss)


plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'],color='blue', label='validation')
plt.legend()
plt.show()

