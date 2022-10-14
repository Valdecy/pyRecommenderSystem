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
plt.style.use('ggplot')
import numpy  as np
import pandas as pd
import plotly.graph_objects as go 
import plotly.io as pio
import random

############################################################################

# Function: https://codereview.stackexchange.com/questions/259467/marking-duplicate-entries-in-a-numpy-array-as-true
def duplicates_set(arr):
    uniques = set()
    result  = []
    for i in range(0, arr.shape[0]):
        result.append(str(arr[i,:]) in uniques)
        uniques.add(str(arr[i,:]))
    return result

# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Plot Users or Items
def graph_interactive(data, U, V, view = 'browser', size = 10, user = True, name = [], k = 5):
    if (view == 'browser' ):
        pio.renderers.default = 'browser'
    if (user == True):
        P   = U
        n_L = data.columns.to_list()
        col = 'rgba(0, 0, 255, 0.45)'
    else:
       P   = V
       n_L = data.index.to_list()
       col = 'rgba(255, 0, 0, 0.45)'
    if (len(name) > 0):
        idx = []
        Xn  = []
        Yn  = []
        m_L = []
        dm  = build_distance_matrix(P)
        dm  = dm.astype(float)
        np.fill_diagonal(dm, float('+inf'))
        for n in name:
            if (n in n_L):
                idx.append(n_L.index(n))  
        for i in idx:
            Xn.append(P[i, 0]*1.00)
            Yn.append(P[i, 1]*1.00)
            m_L.append(n_L[i])
            rows = np.argsort(dm[i,:])
            rows = rows[:k]
            for j in range(0, rows.shape[0]):
                Xn.append(P[rows[j], 0]*1.00)
                Yn.append(P[rows[j], 1]*1.00)
                m_L.append(n_L[rows[j]])
        print(m_L)
    Xv    = []
    Yv    = []
    trace = []
    for i in range(0, P.shape[0]):
        Xv.append(P[i, 0]*1.00)
        Yv.append(P[i, 1]*1.00)
    n_trace = go.Scatter(x         = Xv,
                         y         = Yv,
                         opacity   = 1,
                         mode      = 'markers',
                         marker    = dict(symbol = 'circle-dot', size = size, color = col),
                         text      = n_L,
                         hoverinfo = 'text',
                         hovertext = n_L,
                         name      = ''
                         )
    trace.append(n_trace)
    if (len(name) > 0):
        m_trace = go.Scatter(x         = Xn,
                             y         = Yn,
                             opacity   = 1,
                             mode      = 'markers',
                             marker    = dict(symbol = 'circle-dot', size = size, color = 'rgba(0, 0, 0, 1)'),
                             text      = m_L,
                             hoverinfo = 'text',
                             hovertext = m_L,
                             name      = ''
                             )
        trace.append(m_trace)
    layout  = go.Layout(showlegend   = False,
                        hovermode    = 'closest',
                        margin       = dict(b = 10, l = 5, r = 5, t = 10),
                        plot_bgcolor = '#e0e0e0',
                        xaxis        = dict(  showgrid       = True, 
                                              gridcolor      = 'grey',
                                              zeroline       = False, 
                                              showticklabels = True, 
                                              tickmode       = 'array', 
                                           ),
                        yaxis        = dict(  showgrid       = True, 
                                              gridcolor      = 'grey',
                                              zeroline       = False, 
                                              showticklabels = True,
                                              tickmode       = 'array', 
                                            )
                        )
    fig_aut = go.Figure(data = trace, layout = layout)
    fig_aut.show() 
    return

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
        mean  = Xdata.T.mean() # Missing values are discarded when calculating the mean
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean[i]  
    elif (mean_centering == 'global'):
        mean  = sum(Xdata.sum())/sum(Xdata.count()) # Missing values are discarded when calculating the mean
        Xdata = Xdata.fillna(0)
        for i in range(0, Xdata.shape[0]):
            for j in range(0, Xdata.shape[1]):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - mean         
    return Xdata, mean, mean_centering

############################################################################

# Function: Decomposition
def decomposition(Xdata, mean_centering = 'global', k = 2, user_in_columns = True, graph = True, names = True, size_x = 15, size_y = 15):
    if (k < 1):
        k = 1
    if (k <= 1):
        graph = False
    if (user_in_columns == True):
        items = list(Xdata.index)
        users = list(Xdata)
        Xdata = Xdata.T
    else:
        items = list(Xdata)
        users = list(Xdata.index)
    centered_list                      = centering(Xdata, mean_centering = mean_centering)
    pred                               = centered_list[0].copy(deep = True)
    error                              = Xdata.copy(deep = True) 
    svd_U, svd_sigma, svd_V_transposed = np.linalg.svd(centered_list[0], full_matrices = False) 
    svd_U, svd_sigma, svd_V_transposed = svd_U[:, :k], svd_sigma[:k], svd_V_transposed[:k, :]
    array                              = np.dot(svd_U, np.dot(np.diag(svd_sigma), svd_V_transposed))
    if (centered_list[2] == 'columns'):
        for i in range(0, centered_list[0].shape[0]):
            for j in range(0, centered_list[0].shape[1]):
                pred.iloc[i, j] = array[i, j] + centered_list[1][j]
                if  (pd.isnull(Xdata.iloc[i, j]) == False):
                    error.iloc[i, j] = (Xdata.iloc[i, j] -  pred.iloc[i, j])**2
    elif (centered_list[2] == 'rows'):
        for i in range(0, centered_list[0].shape[0]):
            for j in range(0, centered_list[0].shape[1]):
                pred.iloc[i, j] = array[i, j] + centered_list[1][i]
                if (pd.isnull(Xdata.iloc[i, j]) == False):
                    error.iloc[i, j] = (Xdata.iloc[i, j] -  pred.iloc[i, j])**2
    elif (centered_list[2] == 'global'):
        for i in range(0, centered_list[0].shape[0]):
            for j in range(0, centered_list[0].shape[1]):
                pred.iloc[i, j] = array[i, j] + centered_list[1]
                if (pd.isnull(Xdata.iloc[i, j]) == False):
                    error.iloc[i, j] = (Xdata.iloc[i, j] -  pred.iloc[i, j])**2    
    mse  = sum(error.sum())/sum(error.count())
    rmse = (mse)**(1/2) 	
    if (graph == True):
        ci  = 0.050
        cf  = 0.100
        ns  = np.zeros((svd_U.shape))
        res = duplicates_set(svd_U)
        for i in range(0, len(res)):
            if (res[i] == True):
                ns[i, :] = random.uniform(ci, cf)
                ns[i, :] = ns[i, :]*svd_U[i, :]
        U_n = svd_U + ns
        U   = svd_U 
        gs  = matplotlib.gridspec.GridSpec(1, 2) 
        plt.figure(figsize = (size_x, size_y))
        ax1 = plt.subplot(gs[0,0])
        plt.xlabel('Feature 1', fontsize = 12)
        plt.ylabel('Feature 2', fontsize = 12)
        ax1.scatter(U[:,0], U[:,1], c = 'blue', s = 25, marker = 'o', alpha = 0.45)
        plt.title('Users')
        if (names == True):
            for i, txt in enumerate(users):
                ax1.annotate(txt, (U_n[:,0][i], U_n[:,1][i]))
        ax1.set_xlim(min(U_n[:,0])*1.05, max(U_n[:,0])*1.05)
        ax1.set_ylim(min(U_n[:,1])*1.05, max(U_n[:,1])*1.05)
        ax2 = plt.subplot(gs[0,1])
        plt.xlabel('Feature 1', fontsize = 12)
        plt.ylabel('Feature 2', fontsize = 12)
        ns  = np.zeros((svd_V_transposed.T.shape))
        res = duplicates_set(svd_V_transposed.T)
        for i in range(0, len(res)):
            if (res[i] == True):
                ns[i, :] = random.uniform(ci, cf)
                ns[i, :] = ns[i, :]*svd_V_transposed.T[i, :]
        V_n = svd_V_transposed.T + ns
        V   = svd_V_transposed.T
        ax2.scatter(V[:,0], V[:,1], c = 'red', s = 25, marker = 'o', alpha = 0.45)
        plt.title('Items')
        if (names == True):
            for i, txt in enumerate(items):
                ax2.annotate(txt, (V_n[:,0][i], V_n[:,1][i]))
        ax2.set_xlim(min(V_n[:,0])*1.05, max(V_n[:,0])*1.05)
        ax2.set_ylim(min(V_n[:,1])*1.05, max(V_n[:,1])*1.05)
        plt.show()
    if (user_in_columns == True):
        pred = pred.T
    return pred.round(2), rmse, svd_U, svd_sigma, svd_V_transposed.T

############################################################################
