############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: Recommender Systems

# Citation: 
# PEREIRA, V. (2022). Project: pyRecommender, GitHub repository: <https://github.com/Valdecy/pyRecommender>

############################################################################

# Installing Required Libraries
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

############################################################################

# Function: User-User / Item-Item Mean Centering
def centering(Xdata, mean_centering = "global"):
    mean = "none"
    if mean_centering == "columns":
        mean = Xdata.mean() # Missing values are discarded when calculating the mean
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean[j]          
    elif mean_centering == "rows":
        mean = Xdata.T.mean() # Missing values are discarded when calculating the mean
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean[i]  
    elif mean_centering == "global":
        mean = sum(Xdata.sum())/sum(Xdata.count()) # Missing values are discarded when calculating the mean
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean         
    return Xdata, mean, mean_centering

# Function: Decomposition
def decomposition(Xdata, mean_centering = "global", k = 2, user_in_columns = True, graph = True):
    if (k < 1):
        k =1
    if (k <= 1):
        graph = False
    if (user_in_columns == True):
        items = list(Xdata.index)
        users = list(Xdata)
        Xdata = Xdata.T
    else:
        items = list(Xdata)
        users = list(Xdata.index)
    centered_list = centering(Xdata, mean_centering = mean_centering)
    pred  = centered_list[0].copy(deep = True)
    error = Xdata.copy(deep = True) 
    svd_U, svd_sigma, svd_V_transposed = np.linalg.svd(centered_list[0], full_matrices = False) 
    svd_U, svd_sigma, svd_V_transposed = svd_U[:, :k], svd_sigma[:k], svd_V_transposed[:k, :]
    array = np.dot(svd_U, np.dot(np.diag(svd_sigma), svd_V_transposed))
    if centered_list[2] == "columns":
        for i in range(0, centered_list[0].shape[0]):
            for j in range(0, centered_list[0].shape[1]):
                pred.iloc[i, j] = array[i, j] + centered_list[1][j]
                if  (pd.isnull(Xdata.iloc[i, j]) == False):
                    error.iloc[i, j] = (Xdata.iloc[i, j] -  pred.iloc[i, j])**2
    elif centered_list[2] == "rows":
        for i in range(0, centered_list[0].shape[0]):
            for j in range(0, centered_list[0].shape[1]):
                pred.iloc[i, j] = array[i, j] + centered_list[1][i]
                if (pd.isnull(Xdata.iloc[i, j]) == False):
                    error.iloc[i, j] = (Xdata.iloc[i, j] -  pred.iloc[i, j])**2
    elif centered_list[2] == "global":
        for i in range(0, centered_list[0].shape[0]):
            for j in range(0, centered_list[0].shape[1]):
                pred.iloc[i, j] = array[i, j] + centered_list[1]
                if (pd.isnull(Xdata.iloc[i, j]) == False):
                    error.iloc[i, j] = (Xdata.iloc[i, j] -  pred.iloc[i, j])**2    
    mse  = sum(error.sum())/sum(error.count())
    rmse = (mse)**(1/2) 	
    if (graph == True):
        ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
        plt.xlabel('Feature 1', fontsize = 12)
        plt.ylabel('Feature 2', fontsize = 12)
        ax1.scatter(pd.DataFrame(svd_U).iloc[:,0], pd.DataFrame(svd_U).iloc[:,1], c = 'red',   s = 25, marker = 'o', label = 'Users')
        plt.legend(loc = 'upper right')
        for i, txt in enumerate(users):
            ax1.annotate(txt, (pd.DataFrame(svd_U).iloc[:,0][i], pd.DataFrame(svd_U).iloc[:,1][i]))
        plt.show()       
        ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
        plt.xlabel('Feature 1', fontsize = 12)
        plt.ylabel('Feature 2', fontsize = 12)
        ax1.scatter(pd.DataFrame(svd_V_transposed.T).iloc[:,0], pd.DataFrame(svd_V_transposed.T).iloc[:,1], c = 'red',   s = 25, marker = 'o', label = 'Items')
        plt.legend(loc = 'upper right')
        for i, txt in enumerate(items):
            ax1.annotate(txt, (pd.DataFrame(svd_V_transposed.T).iloc[:,0][i], pd.DataFrame(svd_V_transposed.T).iloc[:,1][i]))
        plt.show()
    if (user_in_columns == True):
        pred = pred.T
    return pred.round(2), rmse, pd.DataFrame(svd_U), pd.DataFrame(svd_sigma), pd.DataFrame(svd_V_transposed.T)

############################################################################
