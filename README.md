# Data Science Portfolio - Xuyang Ji
This Portfolio is a compilation of all the Data Science and Data Analysis projects I have done for academic, self-learning and hobby purposes. This portfolio also contains my Achievements, skills, and certificates. It is updated on the regular basis.

- **Email**: [xj2272@columbia.edu](xj2272@columbia.edu)
- **LinkedIn**: [https://www.linkedin.com/in/xuyang-j-a34a1a93/](https://www.linkedin.com/in/xuyang-j-a34a1a93/)

## Projects

<img align="left" width="250" height="150" src="https://github.com/Celinejxy/Porforlio_data_science/blob/main/Images/toxic_comm.jpeg"> **[Toxic Comment Classification using Wikipedia's Talk Page Edits Data](https://github.com/jjbocek/ToxicApp.git)**

In this project, K-Means clustering analysis and multiple predictive models were created and executed to identify various forms of toxicity, such as threats, obscenity, insults, and identity-based hate. These models included logistic regression, Naïve Bayes, K-Means Clustering, KNN, SVD, and regularization for data analysis. The models were evaluated using TFIDF and Doc2Vec with cross-validation using a pipeline. To enhance the performance, an ensemble model was developed by combining the top three models via a "hard" vote.

#

<img align="left" width="250" height="150" src="https://github.com/archd3sai/Portfolio/blob/master/Images/instacart.jpeg"> **[Predicting Spending Based on Consumer Analysis](https://github.com/Celinejxy/Predicting-Consumer-Spending.git)**

The data source used is obtained from Kaggle, comprised of 2,240 observed consumer characteristics across 29 attributes. The objective of the study is to divide the target customers on the basis of their significant features which could help the company maintain stronger bond with those high-spending customers in less marketing expenses.The data was analyzed by utilizing KMeans clustering along with PCA to identify clusters with similar consumer behaviors. Subsequently, kNN and Decision Tree classifiers were utilized for prediction, with hyperparameter tuning and cross-validation. Finally, the models were validated through advanced techniques such as ROC, precision, and recall.


#

<img align="left" width="250" height="250" src="https://github.com/Celinejxy/Porforlio_data_science/blob/main/Images/dog-puns-collie-you-later.jpg"> **[Collaborative-filtering Jokes Recommendation Engine](https://github.com/Celinejxy/JESTER-DS)**

Alternating least squares and item-based collaborative-filtering algorithms were built for personalized jokes recommendation system. In addition, Singular Value Decomposition was used with IBCF to improve the computation cost of recommendation. Both models are tested with cross-validated and compared using RMSE. 

#

<img align="left" width="250" height="150" src="https://github.com/Celinejxy/Porforlio_data_science/blob/main/Images/wine.jpeg"> **[A Comparision of ML algorithm Using Wine Quality Dataset ](wine_quality)**

Using k-Nearest Neighbors, Support Vector Machines, Decision Trees, and clustering algorithms on Wine_quality dataset with PCA for dimension reduction. By tuning parameters for each classifer, model qualities are comparied and studied using precision and recall metrics. The results show that the KNN algorithm has the best accuracy in predicting with a precision value of 98%. In comparison, the SVM will occasionally misclassify an item in the minority class. Meantime, decision trees are able to generate understandable rules without requiring much computation and prior knowledge. Additionally, using HAC and K-Means Clustering with various distance and linkahe functions to examine data structure. 

#

<img align="left" width="250" height="150" src="https://github.com/Celinejxy/Porforlio_data_science/blob/main/Images/storm.jpeg"> **[Mining Geophysical Parameters Through Decision-tree Analysis](storm_ds)**

Propose a simple deterministic classification model based on the co-occurrence of environmental parameters to predict the severity of certain storms. More specifically, decision tree models with parallel hyperparameter tuning were used to gain intuition about the tradeoffs of model quality and complexity for the imbalanced dataset.  The algorithm allocate a grid of 3 hyperparameters – minbucket (5,20,35), minsplit (20,60,100), and maxdepth(3,5,7,10), for 10-fold stratified cross- validated decision tree models, which results in a total of 36 configurations. 


<br />

## Micro Projects
- ### Statistics and Machine Learning
    - [Classifying/Predicting Modelling with scikit_Learn ](Classification) : In this folder, I have done exploratory analysis and implemented kNN, decision tree, Naive Bayes, and linear discriminant analysis with cross-validations on the adult census dataset. 

    - [Bayesian Statistics]: In this file, I explored how bayesian statistics works and how prior assumption reflects posterior probabilities using Gun control example. 

    - [Classifying Modelling and Relevance Feedback Algorithm](https://github.com/Celinejxy/kNN_Rocchio_NewsGroups.git): I implemented KNN and Rocchio methods with TFIDF for text categorization. The output has shown that the most optimal kNN classifer outperforms the Rocchio classifer by approximately 1.5%, while with constantly better accuracy rate regardless of the number of K. While the dataset is large and the classes may not be linearly separable, kNN can handle complex classes better than Rocchio, which has a high bias and low variance.

    - [Linear Regression](linear_regression): Performed standard multiple linear regression, lasso regression, rigid regression, and stochastic gradient descent for regression algorithms with feature selection methods. Finally, perform model selection to find the best "l1_ratio" parameter using SGDRegressor with  the "elasticnet" penalty parameter. 

    - [Clustering on Newsgroup Subsets](KMeans_Newsgroup_subset): In this file, I implemented KMeans Clustering based on TFIDF_transformed text dataset. For analyzing results, I have wwritten a function to display the top N terms in each cluster sorted by the cluster DF values for each term, the centroid weights for each term in the top N terms in the cluster (mean TFxIDF weight of the term), and the size of the cluster.
 
## Core Competencies

- **Methodologies**: Machine Learning, Deep Learning, Time Series Analysis, Natural Language Processing, Statistics, Explainable AI, A/B Testing and Experimentation Design, Big Data Analytics
- **Languages**: Python (Pandas, Numpy, Scikit-Learn, Scipy, Keras, Matplotlib), R (Dplyr, Tidyr, Caret, Ggplot2), SQL

