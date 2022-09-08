############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Lesson: Recommender Systems

# Citation: 
# PEREIRA, V. (2022). Project: pyRecommender, GitHub repository: <https://github.com/Valdecy/pyRecommender>

############################################################################

# Installing Required Libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

############################################################################

# Function: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, size = 10, ax = None, cbar_kw = {}, cbarlabel = '', **kwargs):
    if not ax:
        f, ax = plt.subplots(figsize = (size, size))
    im   = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax = ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation = -90, va = 'bottom')
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)
    ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
    plt.setp(ax.get_xticklabels(), rotation = -30, ha = 'right', rotation_mode = 'anchor')
    #ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor = True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor = True)
    ax.grid(which = 'minor', color = 'w', linestyle = '-', linewidth = 3)
    ax.tick_params(which = 'minor', bottom = False, left = False)
    return im, cbar

# Function: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, data = None, valfmt = '{x:.2f}', textcolors = ('black', 'white'), threshold = None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    kw = dict(horizontalalignment = 'center', verticalalignment = 'center')
    kw.update(textkw)
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            kw.update(color = textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return

############################################################################

# Function: User-User / Item-Item Similarites
def similarities_tfidf(Xdata, graph = False, size = 10):
    X      = Xdata.copy(deep = True)
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
            X.iloc[i,j] = dtm_tf[i,j]*idf[0,j]                
    row_row_cos = cosine_similarity(np.nan_to_num(X))
    sim_matrix  = pd.DataFrame(row_row_cos, columns = X.T.dtypes.index)
    sim_matrix  = sim_matrix.set_index(X.T.dtypes.index)   
    if (graph == True):
        fig, ax  = plt.subplots()
        im, cbar = heatmap(data = sim_matrix, row_labels = X.index.tolist(), col_labels = X.index.tolist(), size = size, ax = ax, cmap = 'magma_r', cbarlabel = '')
        annotate_heatmap(im, valfmt = '{x:.2f}', textcolors = ('black', 'white'))
        fig.tight_layout()
        plt.show()                 
    return sim_matrix

############################################################################

# Function: K-Top Users / Items. # target = User / Item target name to show the K-Tops similarities
def k_top_tfidf(Xdata, k = 5, show_all = False, graph = False, size = 10, target = 'none', cut_off = -0.9999):    
    rank_list  = [None]*1
    sim_matrix = similarities_tfidf(Xdata, graph, size)  
    if (show_all == True):
        for j in range(0, sim_matrix.shape[0]):
            rank = sim_matrix.sort_values(sim_matrix.iloc[:,j].name, ascending = False).iloc[:,j]
            if (cut_off >= -1 and cut_off <= 1):
                if (k > rank[rank >= cut_off].count() - 1):
                    k = rank[rank >= cut_off].count() - 1
            rank = rank.iloc[0:k+1]
            if (j == 0):
                rank_list[0] = rank
            else:
                rank_list.append(rank)
    elif (show_all == False and target != 'none'):       
        rank = sim_matrix.sort_values(target, ascending = False).loc[:,target]
        if (cut_off >= -1 and cut_off <= 1):
            if (k > rank[rank >= cut_off].count() - 1):
                k = rank[rank >= cut_off].count() - 1
        rank         = rank.iloc[0:k+1]
        rank_list[0] = rank   
    return rank_list

############################################################################
