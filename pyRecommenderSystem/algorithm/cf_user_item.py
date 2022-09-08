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

from pyRecommenderSystem.util.gwo import grey_wolf_optimizer

############################################################################

# Function: Tranform Matrix
def tranform(original_matrix, user_in_columns = True):
    original = original_matrix.copy(deep = True)
    if (user_in_columns == False):
        original = original.T
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            if (pd.isnull(original.iloc[i, j])):               
                original.iloc[i, j] = -1000.0 
    return original

# Function: p List (User in columns)
def p_list(original):
    p_user_list = [0]*original.shape[0]
    for i in range(0, original.shape[0]):
        count = 0
        for j in range(0, original.shape[1]):
            if (original.iloc[i,j] != -1000.0):
                count = count + 1
        p_user_list[i] = count
    return p_user_list

# Function: q List (User in columns)
def q_list(original):
    q_user_list = [0]*original.shape[1]
    for j in range(0, original.shape[1]):
        count = 0
        for i in range(0, original.shape[0]):
            if (original.iloc[i,j] != -1000.0):
                count = count + 1
        q_user_list[j] = count
    return q_user_list 

# Function: Global Centering
def global_centering(Xdata_matrix, user_in_columns = True):
    Xdata = Xdata_matrix.copy(deep = True)
    if (user_in_columns == False):
        Xdata = Xdata.T
    global_mean = sum(Xdata.sum())/sum(Xdata.count()) # Missing values are discarded when calculating the mean
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (pd.isnull(Xdata.iloc[i, j])):
                Xdata.iloc[i, j] = 0.0
            elif (pd.isnull(Xdata.iloc[i, j]) == False):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - global_mean      
    return Xdata, global_mean

# Function: Bias Addition
def bias_addition(Xdata_matrix, bias_user_list, bias_item_list):
    Xdata = Xdata_matrix.copy(deep = True)
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            Xdata.iloc[i, j] = Xdata.iloc[i, j] + (-bias_user_list[j] - bias_item_list[i])      
    return Xdata

# Function: Weigth Matrix User
def weigth_matrix_calc_user(Xdata, w_list_user):
    k             = 0
    weigth_matrix = pd.DataFrame(np.zeros((Xdata.shape[1], Xdata.shape[1])))
    for i in range(0, weigth_matrix.shape[0]):
        for j in range(0, weigth_matrix.shape[1]):           
            if (i == j):
                weigth_matrix.iloc[i, j] = 0.0
            else:
                weigth_matrix.iloc[i, j] = w_list_user[k]
                k = k + 1            
    return weigth_matrix

# Function: Weigth Matrix Item
def weigth_matrix_calc_item(Xdata, w_list_item):
    k             = 0
    weigth_matrix = pd.DataFrame(np.zeros((Xdata.shape[0], Xdata.shape[0])))
    for i in range(0, weigth_matrix.shape[0]):
        for j in range(0, weigth_matrix.shape[1]):           
            if (i == j):
                weigth_matrix.iloc[i, j] = 0.0
            else:
                weigth_matrix.iloc[i, j] = w_list_item[k]
                k                       = k + 1            
    return weigth_matrix

# Function: Ratings Prediction
def ratings_prediction(original, weigth_matrix_user, weigth_matrix_item, global_mean, p_user_list, q_user_list, bias_user_list, bias_item_list):
    bias       = original.copy(deep = True)
    prediction = original.copy(deep = True)
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            bias.iloc[i, j]       = bias_user_list[j] + bias_item_list[i]
            prediction.iloc[i, j] = 0
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            for k in range(0,  weigth_matrix_user.shape[1]):
                if (original.iloc[i, k] != -1000.0 and original.iloc[i, j] != -1000.0 and k != j):
                    prediction.iloc[i, j] = prediction.iloc[i, j] + weigth_matrix_user.iloc[k,j]*(original.iloc[i, k]+ (-bias_user_list[k] - bias_item_list[i]))
            prediction.iloc[i, j] = prediction.iloc[i, j]/p_user_list[i]**(1/2)
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            for k in range(0,  weigth_matrix_item.shape[1]):
                if (original.iloc[k, j] != -1000.0 and original.iloc[i, j] != -1000.0 and k != i):
                    prediction.iloc[i, j] = prediction.iloc[i, j] + weigth_matrix_item.iloc[k,i]*(original.iloc[k, j]+ (-bias_user_list[j] - bias_item_list[k]))
            prediction.iloc[i, j] = prediction.iloc[i, j]/q_user_list[j]**(1/2)
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            prediction.iloc[i, j] = prediction.iloc[i, j] + bias.iloc[i, j] + global_mean
    return prediction

# Function: RMSE
def rmse_calculator(original, prediction):   
    mse = prediction.copy(deep = True)   
    for i in range (0, original.shape[0]):
        for j in range (0, original.shape[1]):
            if (original.iloc[i, j] != -1000):
                mse.iloc[i][j] = (original.iloc[i][j] - prediction.iloc[i][j])**2 
            else:
                mse.iloc[i][j] = 0
    rmse = sum(mse.sum())/sum(mse.count())
    rmse = (rmse)**(1/2)    
    return rmse

# Function: Separate Lists
def separate_lists(Xdata, variable_list):
    w_list_user    = [0]*(Xdata.shape[1]**2 - Xdata.shape[1])
    w_list_item    = [0]*(Xdata.shape[0]**2 - Xdata.shape[0])
    bias_user_list = [0]*Xdata.shape[1]
    bias_item_list = [0]*Xdata.shape[0]
    r              = len(w_list_user)
    s              = r + len(w_list_item)
    t              = s + len(bias_user_list)
    u              = t + len(bias_item_list)
    count_r        = 0
    count_s        = 0
    count_t        = 0
    count_u        = 0
    for i in range(0, len(variable_list)):
        if (i >= 0 and i < r):
            w_list_user[count_r] = variable_list[i]
            count_r              = count_r + 1
        elif(i >= r and i < s):
            w_list_item[count_s] = variable_list[i]
            count_s              = count_s + 1
        elif(i >= s and i < t):
            bias_user_list[count_t] = variable_list[i] 
            count_t                 = count_t + 1
        elif(i >= t and i < u):
            bias_item_list[count_u] = variable_list[i] 
            count_u                 = count_u + 1
    return w_list_user, w_list_item, bias_user_list, bias_item_list

# Function: Loss Function
def loss_function(original, variable_list, user_in_columns = True):
    Xdata, global_mean                                       = global_centering(original, user_in_columns = user_in_columns)
    original                                                 = tranform(original, user_in_columns = user_in_columns) # nan = -1000
    p_user_list                                              = p_list(original)
    q_user_list                                              = q_list(original)
    w_list_user, w_list_item, bias_user_list, bias_item_list = separate_lists(Xdata, variable_list)
    Xdata                                                    = bias_addition(Xdata, bias_user_list, bias_item_list)
    weigth_matrix_user                                       = weigth_matrix_calc_user(Xdata, w_list_user)
    weigth_matrix_item                                       = weigth_matrix_calc_item(Xdata, w_list_item)
    prediction                                               = ratings_prediction(original, weigth_matrix_user, weigth_matrix_item, global_mean, p_user_list, q_user_list, bias_user_list, bias_item_list)
    rmse                                                     = rmse_calculator(original, prediction)
    return prediction, rmse

############################################################################

# Function: User-Item Based Model
def user_item_based_model(Xdata, user_in_columns = True, pack_size = 25, iterations = 75):
    n = Xdata.shape[0]**2 + Xdata.shape[1]**2
    def solver_min (variables_values = [0]): 
        _, rmse = loss_function(Xdata, variable_list = variables_values, user_in_columns = user_in_columns)
        return rmse
    gw         = grey_wolf_optimizer(target_function = solver_min, pack_size = pack_size, min_values = [-1.5]*n, max_values = [1.5]*n, iterations = iterations)
    uibm, rmse = loss_function(Xdata, variable_list = gw[:-1], user_in_columns = user_in_columns)
    return uibm, rmse
   
############################################################################
