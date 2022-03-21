# pyRecommender - A Recommender System Python Library

1. Install
```bash
pip install pyRecommender
```

2. Algorithms:

**Content-Based Filtering**: This approach uses a series of descriptions of an item in order to recommend additional items with similar properties . The term "content" refers to these descriptions, and in this case manipulated with TF-IDF matrices. The function returns a list with the k-top similarities (cosine similarity).

	* Xdata = Dataset Attributes. A 0-1 matrix with the content in columns.

	* k = Up to k-top similarities (cosine similarity) that are greater or equal the cut_off value. The default value is 5.

	* show_all = Boolean that indicates if the similiarities of each item will be calculated (show_all = True) or for just one item (show_all = False). The default value is True.

	* graph = Boolean that indicates if the cosine similarity will be displayed (graph = True) or not (graph = False). The default value is True.

	* target = k-top similarities of target item. Only relevant if "show_all = False". The default value is "none".	

	* cut_off = Value between -1 and 1 that filter similar item according to a certain threshold value. The default value is -0.9999.

**Collaborative Filtering - Item Based**: This approach builds a model from past behaviors, comparing items or users trough ratings, and in this case an Item Based Regression technique is used to predict the missing values. The Grey Wolf Optmizer (GWO) is used to find minimum loss value. The function returns: the prediction of the missing data and the gwo solution.

	* Xdata = Dataset Attributes. A matrix with users ratings about a set of items.

	* user_in_columns = Boolean that indicates if the user is in the column (user_in_column = True) or in the row (user_in_column = False). The default value is True.

	* pack_size = To find the weights, a metaheuristic know as Grey Wolf Optmizer (GWO) is used. The initial population (pack_size) helps to find the optimal solution. The default value is 25.

	* iterations = The total number of iterations. The defaul value is 100

**Collaborative Filtering - User Based**: This approach builds a model from past behaviors, comparing items or users trough ratings, and in this case an User Based Regression technique is used to predict the missing values. The Grey Wolf Optmizer (GWO) is used to find minimum loss value. The function returns: the prediction of the missing data and the gwo solution.

	* Xdata = Dataset Attributes. A matrix with users ratings about a set of items.

	* user_in_columns = Boolean that indicates if the user is in the column (user_in_column = True) or in the row (user_in_column = False). The default value is True.

	* pack_size = To find the weights, a metaheuristic know as Grey Wolf Optmizer (GWO) is used. The initial population (pack_size) helps to find the optimal solution. The default value is 25.

	* iterations = The total number of iterations. The defaul value is 100
	
**Collaborative Filtering - User-Item Based**: This approach builds a model from past behaviors, comparing items or users trough ratings, and in this case an User-Item Based Regression technique is used to predict the missing values. The Grey Wolf Optmizer (GWO) is used to find minimum loss value. The function returns: the prediction of the missing data and the gwo solution.

	* Xdata = Dataset Attributes. A matrix with users ratings about a set of items.

	* user_in_columns = Boolean that indicates if the user is in the column (user_in_column = True) or in the row (user_in_column = False). The default value is True.

	* pack_size = To find the weights, a metaheuristic know as Grey Wolf Optmizer (GWO) is used. The initial population (pack_size) helps to find the optimal solution. The default value is 25.

	* iterations = The total number of iterations. The defaul value is 100

**Collaborative Filtering - Latent Factors**: This approach builds a model from past behaviors, comparing items or users trough ratings, and in this case a Regression with Latent Factors technique can extract k features to predict the missing values. The Stochastic Gradient Descent is used to find minimum loss value. The function returns: the prediction of the missing data, the users features weights, the items features weights and the rmse (root mean square error).

	* Xdata = Dataset Attributes. A matrix with users ratings about a set of items.

	* mean_centering = "none", "row", "column", "global". If "none" is selected then no centering is made, if "row" is selected then a row mean centering is performed,  if "column" is selected then a column mean centering is performed and if "global" is selected then a global centering (matrix mean) is performed. The default value is "none".

	* features = Total number of features to be extracted. The default value is 2.

	* iterations = The total number of iterations. The default value is 1000.

	* alpha = The learning rate. The default value is 0.01.

**Collaborative Filtering - Nearest Neighbors**: This approach builds a model from past behaviors, comparing items or users trough ratings, and in this case the Nearest Neighbors (memory based) is used to calculate the k-top similar users/items (cosine or pearson similarity). The function returns, the k-top similar users/items, the mean and the similarity matrix.

	* Xdata = Dataset Attributes. A matrix with users ratings about a set of items.

	* k = Up to k-top similarities (cosine similarity or pearson correlation) that are greater or equal the cut_off value. The default value is 5.

	* user_in_columns = Boolean that indicates if the user is in the column (user_in_column = True) then a user-user similarity is made, if (user_in_column = False) then an item-item similarity is performed instead. The default value is True.

	* simil = "cosine", "correlation". If "cosine" is chosen then a cosine similarity is performed, and if "correlation" is chosen then a pearson correlation is performed. The default value is "correlation".

	* graph = Boolean that indicates if the similarity matrix will be displayed (graph = True) or not (graph = False). The default value is True.

	* mean_centering = "none", "row", "column", "global". If "none" is selected then no centering is made, if "row" is selected then a row mean centering is performed,  if "column" is selected then a column mean centering is performed and if "global" is selected then a global centering (matrix mean) is performed. The default value is "none".

	* cut_off = Value between -1 and 1 that filter similar item according to a certain threshold value. The default value is -0.9999.

	* Finnaly a prediction function - prediction( ) - is also included.

**Collaborative Filtering - SVD**: This approach builds a model from past behaviors, comparing items or users trough ratings, and in this case the SVD (Singular Value Decomposition) technique can extract k features that can be used to find similar users/items. The function returns: the predictions, the rmse (root mean square error), the U matrix (users relation to the features), the Sigma matrix (features matrix)and the V matrix (items relation to the features).

	* Xdata = Dataset Attributes. A matrix with users ratings about a set of items.

	* mean_centering = "none", "row", "column", "global". If "none" is selected then no centering is made, if "row" is selected then a row mean centering is performed,  if "column" is selected then a column mean centering is performed and if "global" is selected then a global centering (matrix mean) is performed. The default value is "none".

	* k = Total number of features to be extracted. The default value is 2.

	* user_in_columns = Boolean that indicates if the user is in the column (user_in_column = True) then a user-user similarity is made, if (user_in_column = False) then an item-item similarity is performed instead. The default value is True.

	* graph = Boolean that indicates if the first 2 features of the users and items will be displayed (graph = True) or not (graph = False). The default value is True.

3. Try it in **Colab**:

- Content-Based Filtering ([ Colab Demo ](https://colab.research.google.com/drive/1ZxRp88k7KcTlxqLFKsEkF0bZpUwIKzlr?usp=sharing))
- Collaborative Filtering - Item Based ([ Colab Demo ](https://colab.research.google.com/drive/1m44UNfWUJiuHVMtYKHAT9cL6cURL3doF?usp=sharing))
- Collaborative Filtering - User Based ([ Colab Demo ](https://colab.research.google.com/drive/1_UgfLagl2u_eRclo5gREjSQpvxLvZ7cG?usp=sharing))
- Collaborative Filtering - User-Item Based ([ Colab Demo ](https://colab.research.google.com/drive/1RdTdxZaCkpe9MYl9BIsOT6xJ_hw-k_yn?usp=sharing))
- Collaborative Filtering - Latent Factors ([ Colab Demo ](https://colab.research.google.com/drive/1xBFF0noZGHM0cDpeCukhlpSLrAWrntG7?usp=sharing))
- Collaborative Filtering - Nearest Neighbors ([ Colab Demo ](https://colab.research.google.com/drive/1uNl34kRrj4ktZtNSLgZApSR_f0MVuoHf?usp=sharing))
- Collaborative Filtering - SVD ([ Colab Demo ](https://colab.research.google.com/drive/1zUoL82j58Wl1tv2ycfGCMLm-vmPH4o2o?usp=sharing))

