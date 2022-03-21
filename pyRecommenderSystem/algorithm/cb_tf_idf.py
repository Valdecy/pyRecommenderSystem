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
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

############################################################################

# Function: User-User / Item-Item Similarites
def similarities_tfidf(Xdata, graph = False):
    dtm_tf = np.zeros(shape = (Xdata.shape[0],  Xdata.shape[1])) 
    for i in range(0,  Xdata.shape[0]): 
        for j in range(0,  Xdata.shape[1]):
            if  Xdata.iloc[i,j] > 0:
                dtm_tf[i,j] =  Xdata.iloc[i,j]/ Xdata.iloc[i,].sum()        
    idf  = np.zeros(shape = (1, Xdata.shape[1]))  
    for i in range(0, Xdata.shape[1]):
        idf[0,i] = np.log10(Xdata.shape[0]/(Xdata.iloc[:,i]>0).sum())
    for i in range(0, Xdata.shape[0]): 
        for j in range(0, Xdata.shape[1]):
            Xdata.iloc[i,j] = dtm_tf[i,j]*idf[0,j]                
    row_row_cos = cosine_similarity(np.nan_to_num(Xdata))
    sim_matrix = pd.DataFrame(row_row_cos, columns = Xdata.T.dtypes.index)
    sim_matrix = sim_matrix.set_index(Xdata.T.dtypes.index)            
    if graph == True:
        f, ax = plt.subplots(figsize=(10, 10))
        cmap = sns.diverging_palette(0, 250, as_cmap =True)
        sns.heatmap(sim_matrix, cmap = cmap, square = True, linewidths = 0.5, xticklabels = sim_matrix.columns.values, yticklabels = sim_matrix.columns.values)        
    return sim_matrix

############################################################################

# Function: K-Top Users / Items. # target = User / Item target name to show the K-Tops similarities
def k_top_tfidf(Xdata, k = 5, show_all = False, graph = False, target = "none", cut_off = -0.9999):    
    rank_list = [None]*1
    sim_matrix = similarities_tfidf(Xdata, graph = graph)  
    if show_all == True:
        for j in range(0, sim_matrix.shape[0]):
            rank = sim_matrix.sort_values(sim_matrix.iloc[:,j].name, ascending = False).iloc[:,j]
            if (cut_off >= -1 and cut_off <= 1):
                if (k > rank[rank >= cut_off].count() - 1):
                    k = rank[rank >= cut_off].count() - 1
            rank = rank.iloc[0:k+1]
            if j == 0:
                rank_list[0] = rank
            else:
                rank_list.append(rank)
    elif show_all == False:       
        rank = sim_matrix.sort_values(target, ascending = False).loc[:,target]
        if (cut_off >= -1 and cut_off <= 1):
            if (k > rank[rank >= cut_off].count() - 1):
                k = rank[rank >= cut_off].count() - 1
        rank = rank.iloc[0:k+1]
        rank_list[0] = rank   
    return rank_list

############################################################################
