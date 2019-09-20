import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,  Dropout
#from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
#from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
import h5py


#defining variables
BATCH_SIZE = 128
EPOCHS = 30
NUM_CLASSES = 10

#loading and centoring the data
train_x = np.load('train_x.npy').astype('float32') / 255
train_y = np.load('train_y.npy')
train_y = to_categorical(train_y, NUM_CLASSES) 
test_x = np.load('test_x.npy').astype('float32') / 255
test_y = np.load('test_y.npy')
test_y = to_categorical(test_y, NUM_CLASSES)

#defining the network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(train_x.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# Compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_x, train_y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(test_x, test_y))

score = model.evaluate(test_x, test_y, verbose=0)
print('**********************************')
print ("Loss = " + str(score[0]))
print ("Test Accuracy = " + str(score[1]))
print('**********************************')


model.summary()
model.save('my_model.h5') 