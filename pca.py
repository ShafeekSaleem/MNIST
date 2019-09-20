from sklearn.decomposition import PCA
import load_data
import numpy as np 
"""Using PCA(principle component analisys) to reduce the dimensions of data"""


#loading mnist data
x_scaled, y  = load_data.fetch_data()

#separating training and testing datas
train_x = x_scaled[:60000, :]
train_y = y[:60000]
test_x = x_scaled[60000:, :]
test_y = y[60000:]

#creating a pca object
pca = PCA(.95) #0.95 percentage of information will be preserved.
pca.fit(train_x)

#resampling the data to new dimensions
train_x = pca.transform(train_x)
test_x  = pca.transform(test_x)

#saving processed data into .npy file
np.save('train_x', train_x )
np.save('train_y', train_y )
np.save('test_x', test_x)
np.save('test_y', test_y)
