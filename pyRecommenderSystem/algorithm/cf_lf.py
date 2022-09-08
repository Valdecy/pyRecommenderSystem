############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: Recommender Systems

# Citation: 
# PEREIRA, V. (2022). Project: pyRecommender, GitHub repository: <https://github.com/Valdecy/pyRecommender>

############################################################################

# Installing Required Libraries
import numpy  as np
import pandas as pd

############################################################################

# Function: User-User / Item-Item Mean Centering
def centering(Xdata, mean_centering = 'global'):
    mean = 'none'
    if (mean_centering == 'columns'):
        mean  = Xdata.mean() # Missing values are discarded when calculating the mean
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean[j]          
    elif (mean_centering == 'rows'):
        mean  = Xdata.T.mean()
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean[i]  
    elif (mean_centering == 'global'):
        mean  = sum(Xdata.sum())/sum(Xdata.count())
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean        
    return Xdata

# Function: Teta Matrix
def x_teta_matrix(Xdata, features = 1):
    x_weight    = np.random.rand(Xdata.shape[0], features + 1)/100
    teta_weight = np.random.rand(features + 1, Xdata.shape[1])/100
    return x_weight, teta_weight

# Function: RMSE
def rmse_calculator(Xdata, predictions): 
    mse = Xdata.copy(deep = True) 
    for i in range (0, Xdata.shape[0]):
        for j in range (0, Xdata.shape[1]):
            if pd.isnull(mse.iloc[i][j]) == False:
                mse.iloc[i][j] = (Xdata.iloc[i][j] - predictions[i][j])**2
    mse  = sum(mse.sum())/sum(mse.count())
    rmse = (mse)**(1/2)  
    return rmse       

############################################################################

# Function: Teta Update
def x_teta_update(Xdata, mean_centering = 'global', features = 2, iterations = 1000, alpha = 0.01): 
    X           = Xdata.copy(deep = True)
    Xdata       = centering(Xdata, mean_centering = mean_centering)
    error       = Xdata.copy(deep = True)
    weight_list = x_teta_matrix(Xdata, features)
    x_weight    = weight_list[0]
    teta_weight = weight_list[1]
    loss_graph  = np.ones(shape = (iterations + 1, 1))
    stop        = 0  
    while (stop <= iterations):      
        pred = np.dot(x_weight, teta_weight)      
        for i in range (0, Xdata.shape[0]):
            for j in range (0, Xdata.shape[1]):
                if pd.isnull(error.iloc[i][j]) == False:
                    error.iloc[i][j] = (-Xdata.iloc[i][j] + pred[i][j])
                else:
                    error.iloc[i][j] = 0                         
        for j in range (0, features + 1):
            aux_x    = error.copy(deep = True)
            aux_teta = error.copy(deep = True)
            for m in range (0, error.shape[0]):
                for L in range (0, error.shape[1]):
                    aux_x.iloc[m][L]    = aux_x.iloc[m][L]*teta_weight[j,L]
                    aux_teta.iloc[m][L] = aux_teta.iloc[m][L]*x_weight[m,j]
            for i in range (0, error.shape[0]):
                x_weight[i][j]    = x_weight[i][j] - alpha*aux_x.iloc[i,:].sum() 
            for k in range (0, error.shape[1]):
                teta_weight[j][k] = teta_weight[j][k] - alpha*aux_teta.iloc[:,k].sum()              
        rmse                = rmse_calculator(Xdata, pred)
        loss_graph[stop, 0] = rmse
        stop                = stop + 1
        print('iteration = ', stop, ' rmse = ', rmse)        
    for i in range (0, Xdata.shape[0]):
        for j in range (0, Xdata.shape[1]):
            X.iloc[i][j] = pred[i][j]        
    return X, x_weight, teta_weight.T, rmse     

############################################################################
