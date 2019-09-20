import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,  Dropout
#from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
#from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import plot_model

#defining variables
BATCH_SIZE = 64
EPOCHS = 50

#loading and centoring the data
train_x = np.load('train_x').astype('float32') / 255
train_y = np.load('train_y')

test_x = np.load('test_x').astype('float32') / 255
test_y = np.load('test_y')

#defining the network
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(train_x.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# Compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=[plot_losses],
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('**********************************')
print ("Loss = " + str(score[0]))
print ("Test Accuracy = " + str(score[1]))
print('**********************************')

