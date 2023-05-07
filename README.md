# Data Science Portfolio - Xuyang Ji
This Portfolio is a compilation of all the Data Science and Data Analysis projects I have done for academic, self-learning and hobby purposes. This portfolio also contains my Achievements, skills, and certificates. It is updated on the regular basis.

- **Email**: [xj2272@columbia.edu](xj2272@columbia.edu)
- **LinkedIn**: [https://www.linkedin.com/in/xuyang-j-a34a1a93/](https://www.linkedin.com/in/xuyang-j-a34a1a93/)

## Projects

<img align="left" width="250" height="180" src="https://github.com/Celinejxy/Porforlio_data_science/blob/main/Images/toxic_comm.jpeg"> **[Toxic Comment Classification using Wikipedia's Talk Page Data](https://github.com/jjbocek/ToxicApp.git)**

In this project, we aim to identify various forms of toxicity.  Multiple Supervised and Unsupervised methods were developed, including Logistic Regression, Naïve Bayes, K-Means Clustering and KNN; and TFIDF, Doc2Vec and SVD for Feature Engineering. A Pipeline was defined for full cycle implementation, allowing Cross Validation Optimization and model evaluation. Final model was implemented as the top three ensemble model via a "hard" vote.


#

<img align="left" width="250" height="180" src="https://github.com/archd3sai/Portfolio/blob/master/Images/instacart.jpeg"> **[Predicting Spending Based on Consumer Analysis](https://github.com/Celinejxy/Predicting-Consumer-Spending.git)**

The objective of the study to help the company maintain stronger bonds with those high-spending customers in less marketing expenses. Here, I apply Segmentation on customers population and analyse their significant features. By applying KMeans Clustering along with PCA, I detect clusters with similar consumer behaviours. Clustering results were followed by predictive ML models, including training K-NN and Decision Tree classifiers, with Hyperparameter tuning and Cross-Validation. Finally, the models were validated through techniques such as ROC, precision, and recall.



#

<img align="left" width="250" height="200" src="https://github.com/Celinejxy/Porforlio_data_science/blob/main/Images/dog-puns-collie-you-later.jpg"> **[Collaborative-filtering Jokes Recommendation Engine](https://github.com/Celinejxy/JESTER-DS)**

Alternating least squares(ALS) and item-based collaborative-filtering(IBCF) algorithms were built for personalized jokes recommendation system. In addition, self-implemented Singular Value Decomposition was used with IBCF to improve the computation cost of recommendation. All three models are tested with cross-validated and compared using RMSE. In development process, I take into account that the IBCF is a memory-based method that computes similarity scores between items based on their co-occurrence in user interactions, while ALS is a matrix factorization algorithm that allows running in an efficient parallel fashion.

#

<img align="left" width="250" height="180" src="Images/wine.jpg"> **[A Comparision of ML algorithm Using Wine Quality Dataset ](wine_quality)**

In this process, wine quality is modeled based on physicochemical tests by using various ML models and approaches as well as PCA for Dimension Reduction. 
Models include K-NN, SVM, Decision Trees and Clustering, where each model’s performance was analyzed using precision and recall.Results shows that the KNN algorithm has the best accuracy with a precision value of 98%. In comparison, the SVM will occasionally misclassify an item in the minority class, whereas the Decision trees are able to generate understandable rules without requiring much computation and prior knowledge

#

<img align="left" width="250" height="160" src="https://github.com/Celinejxy/Porforlio_data_science/blob/main/Images/storm.jpeg"> **[Mining Geophysical Parameters Through Decision-tree Analysis](storm_ds)**

Developed a simple deterministic classification model to predict the severity of certain storms. The model is based on the co-occurrence of environmental parameters. Main effort was made on optimizing a Decision Tree model with parallel hyperparameter tuning. Process allowed us to gain intuition about the tradeoffs of model quality and complexity for the imbalanced dataset. Grid Search was applied on minbucket, minsplit and maxdepth hyperparameters, using Stratified Cross-Validated.

<br />

## Micro Projects
- ### Statistics and Machine Learning
    - [Classifying/Predicting Modelling with scikit_Learn ](Classification) : In this folder, I have done exploratory analysis and implemented kNN, decision tree, Naive Bayes, and linear discriminant analysis with cross-validations on the adult census dataset. 

    - [Bayesian Statistics]: In this file, I explored how bayesian statistics works and how prior assumption reflects posterior probabilities using Gun control example. 

    - [Classifying Modelling and Relevance Feedback Algorithm](https://github.com/Celinejxy/kNN_Rocchio_NewsGroups.git): I implemented KNN and Rocchio methods with TFIDF for text categorization. The output has shown that the most optimal kNN classifer outperforms the Rocchio classifer by approximately 1.5%, while with constantly better accuracy rate regardless of the number of K. While the dataset is large and the classes may not be linearly separable, kNN can handle complex classes better than Rocchio, which has a high bias and low variance.

    - [Linear Regression](linear_regression): Performed standard multiple linear regression, lasso regression, rigid regression, and stochastic gradient descent for regression algorithms with feature selection methods. Finally, perform model selection to find the best "l1_ratio" parameter using SGDRegressor with  the "elasticnet" penalty parameter. 

    - [Clustering on Newsgroup Subsets](KMeans_Newsgroup_subset): In this file, I implemented KMeans Clustering based on TFIDF_transformed text dataset. For analyzing results, I have wwritten a function to display the top N terms in each cluster sorted by the cluster DF values for each term, the centroid weights for each term in the top N terms in the cluster (mean TFxIDF weight of the term), and the size of the cluster.
 
## Core Competencies

- **Programming**: Python (Pandas, Numpy, Scikit-Learn, Scipy, Keras, Matplotlib), R (Dplyr, Tidyr, Caret, Ggplot2), SAS;
- **Analysis**: A/B Testing and Experimentation Design, Statistical Inference, Big Data Analytics, Tableau, SQL, Power BI;
- **Machine Learning**: Supervised/Unsupervised, Regression, Optimization, Pipeline, NLP, Feature Engineering;
- **Models**: Linear/Non-Linear, Ensemble methods, Decision Trees, Social Network Analysis, Time Series Analysis;


