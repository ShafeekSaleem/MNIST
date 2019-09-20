from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sklearn

def fetch_data():
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    #converting sklearn datas into pandas data frame
    x = pd.DataFrame(x)  #(70000, 784) dimesional
    y = pd.Series(y).astype('int').astype('category') #(70000,) dimensional

    #assigning a pixel name to each dimension of X
    n_dim = x.shape[1] #num of dimensions/features
    x.columns = ['pixel_'+str(i) for i in range(n_dim)]
    x = x.loc[:, x.columns].values
    #standerdising x to mean=0 and variance=1
    x_scaled = StandardScaler().fit_transform(x)

    return x_scaled, y







