library(tidyverse)
library(dplyr) #For data manipulation
library(ggplot2) #For plotting 
library(caret) #For correlation and dummy variable 
library(readr)
library(ggpubr)
library(psych)
library(readr)
library(rpart)
library(tibble)
library(doSNOW)
library(foreach)
library(car)
library(corrplot)
library(magrittr) #For pipes
library(stats)
source("~/Documents/DSC424/Module 3&4/PCA_Plot.R")

#################PS1##########################
#Load Dataset
wineWhite <- read_delim("Desktop/DSC441 DM/Assignment/Assignment4/winequality-white.csv", 
                                delim = ";", escape_double = FALSE, trim_ws = TRUE)
wineRed <- read_delim("Desktop/DSC441 DM/Assignment/Assignment4/winequality-red.csv", 
                        delim = ";", escape_double = FALSE, trim_ws = TRUE)

#(1)Convert to fac if column has less than 10 distinct values 
##Quality converted with 6 levels 
conv <- sapply(wineWhite, function(x) is.numeric(x) && length(unique(x)) < 10)
wineWhite[conv] <- lapply(wineWhite[conv], as.factor)
conv <- sapply(wineRed, function(x) is.numeric(x) && length(unique(x)) < 10)
wineRed[conv] <- lapply(wineRed[conv], as.factor)

#(2) Adding a type column and merge datasets
wineWhite$type<- "white"
wineRed$type<- "red"
wine<- wineWhite %>% 
  full_join(wineRed) %>% 
  arrange(type)
wine$type<- as.factor(wine$type)

#check missing value 
any(is.na.data.frame(wine))

#Use PCA to create a projection of the data to 2D and 
#show a scatterplot with color showing the wine type. 

pairs.panels(wine, gap=0, bg=c('purple','yellow', 'blue')[wine$type],pch = 21)

#Data preprocessing before PCA by converting quality to numeric 
corrT <-corr.test(wine[,1:12],adjust="none")
signTest <- ifelse(corrT$p <0.05, T, F)
colSums(signTest)-1

#separate x and y for pca
wineX <- wine %>%
  select(!c(type,quality)) %>%
  drop_na() 

winePCA<- prcomp(wineX,scale. = T)
screeplot(x=winePCA, type="line",ylim=c(0,4), 
          main="Screeplot:PCA on Scaled Data", 
          cex.lab=0.8, cex.main=0.8,cex.axis=0.8,cex.sub=0.8)
abline(1,0,col="blue",lty=5)
legend("topright", legend = c("Eigenvalue=1"), col = c("blue"),
       lty = 5,cex = 0.6)
summary(winePCA)
round(winePCA$rotation,2)

rawLoadings <- winePCA$rotation[,1:4] %*% 
  diag(winePCA$sdev,4,4)
rotatedLoadings <- varimax(rawLoadings)$loadings
print(rotatedLoadings, cutoff = .5,sort = T)

library(devtools)
require(ggbiplot)
ggbiplot(winePCA)
biplot = ggbiplot(pcobj = winePCA,obs.scale = 1, var.scale = 1,  # Scaling of axis
                  groups = wine$type,    # Add labels as rownames
                  labels.size = 2, varname.size = 3,      # Remove variable vectors (TRUE)
                  ellipse = TRUE, circle = TRUE) +
  scale_color_manual(name="Wine Type", values = c("darkblue","pink")) +  
  scale_shape_manual(name="Wine Type",values=c(17:18)) +
  geom_point(aes(colour=wine$type, shape=wine$type), size = 1,alpha=0.8) +
  theme(legend.direction ="horizontal", legend.position = "top")+
  labs(title = "PCA of yield contributing parameters")
print(biplot)

biplot2 = biplot 

#components
preProc <- preProcess(wineX, method = "pca", pcaComp = 2)
wineP <- predict(preProc, wineX)
wineP$type <- wine$type
head(wineP)
str(wineP)

##train-test split- using wineTrain for model building 
set.seed(961)
indexWine<- createDataPartition(y=wineP$type, p=0.8, list = FALSE)
wineTrain <- wineP[indexWine,]
wineTest <- wineP[-indexWine,]

########Decision Tree
colnames(wineTrain) <- make.names(colnames(wineTrain))
hypers<- rpart.control(minsplit = 10, maxdepth = 5, minbucket = 3)
##Fit the model 
stormDT<- train(type ~., data=wineTrain, method="rpart1SE", 
                control=hypers, trControl=train_control_strate)
stormDT

## evaluate the fit with whole set
pred_storm1 <- predict(stormDT, wineTest)
confusionMatrix(wineTest$type, pred_storm1)
fancyRpartPlot(stormDT$finalModel,main="Decision Tree of Wine",caption = "")


########SVM
library(e1071)
idType <- createFolds(wineTrain$type,k=10, returnTrain = TRUE)
train_control_strate <- trainControl(index=idType, method = "cv", number = 10)
grid <- expand.grid(C=10^seq(-5,1,.5))
svm_grid <- train(type~., data = wineTrain, method="svmLinear",
                  trControl=train_control_strate, preProcess=c("center","scale"),
                  tuneGrid=grid)
svm_grid

##confusion matrix on test set 
predWine_SVM <- predict(svm_grid, wineTest)

# then use that to get a matrix
confusionMatrix(reference= wineTest$type, data=predWine_SVM)


install.packages('kknn')
library(kknn)
#KNN
tuneGrid <- expand.grid(kmax = 3:7,                        # test a range of k values 3 to 7
                        kernel = c("rectangular", "cos"),  # regular and cosine-based distance functions
                        distance = 1:3)                    # powers of Minkowski 1 to 3

# tune and fit the model with 10-fold cross validation,
# standardization, and our specialized tune grid
kknn_fit <- train(type ~ ., 
                  data = wineTrain,
                  method = 'kknn',
                  trControl = train_control_strate,
                  preProcess = c('center', 'scale'),
                  tuneGrid = tuneGrid)

# Printing trained model provides report
kknn_fit

# Predict
pred_knn <- predict(kknn_fit, wineTest)

# Generate confusion matrix
confusionMatrix(wineTest$type, pred_knn)




###################PS2
library(caret)
data("Sacramento")
str(Sacramento)
scm_subset <- Sacramento %>%
  select(!c(city)) 
dummy <- dummyVars(type ~ ., data = scm_subset)
scmDummy <- as.data.frame(predict(dummy, newdata = scm_subset))
scmDummy <- cbind(type=as.factor(scm_subset$type),scmDummy)
str(scmDummy)

remove_cols <- nearZeroVar(scmDummy, names=TRUE)
scmDummy <-scmDummy[,setdiff(names(scmDummy),remove_cols)]
str(scmDummy)

##train-test split- using wineTrain for model building 
set.seed(961)
index<- createDataPartition(y=scmDummy$type, p=0.8, list = FALSE)
scmTrain <- scmDummy[index,]
scmTest <- scmDummy[-index,]

id <- createFolds(scmTrain$type,k=10, returnTrain = TRUE)
train_control_strate <- trainControl(index=id, method = "cv", number = 10)

# setup a tuneGrid with the tuning parameters
tuneGridscm <- expand.grid(kmax = 3:7, # test a range of k values 3 to 7
                        kernel = c( "cos","rectangular"), # regular and cosinebased distance functions
                        distance = 1:3) # powers of Minkowski 1 to 3
# tune and fit the model with 10-fold cross validation,
# standardization, and our specialized tune grid
kknn_fit <- train(type ~ .,
                  data = scmTrain,
                  method = 'kknn',
                  trControl = train_control_strate,
                  preProcess = c('center', 'scale'),
                  tuneGrid = tuneGridscm)

kknn_fit

##########PS3
install.packages('cluster')
install.packages('factoextra')
library(factoextra)
library(cluster)
#wineX,wine$type
#separate x and y for pca
wine <- wine %>%
  select(!c(quality)) %>%
  drop_na() 
##train-test split- using wineTrain for model building 
set.seed(961)
indexWine<- createDataPartition(y=wine$type, p=0.8, list = FALSE)
wineTrain <- wine[indexWine,]
wineTest <- wine[-indexWine,]

#separate x and y for kmean
wineX <- wineTrain %>%
  select(!c(type)) %>%
  drop_na() 
preproc <- preProcess(wineX, method=c("center", "scale"))
predictors <- predict(preproc, wineX)

#KMEans 
# Find the knee
fviz_nbclust(predictors, kmeans, method = "wss")+ 
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10),
        title = element_text(size = 10))  + 
  geom_line(aes(group = 1), color = "pink", linetype = "dashed",size = 1) + 
  geom_point(group = 1, size = 1, color = "darkblue")
fviz_nbclust(predictors, kmeans, method = "silhouette")+ 
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10),
        title = element_text(size = 10))  + 
  geom_line(aes(group = 1), color = "pink", linetype = "dashed",size = 1) + 
  geom_point(group = 1, size = 1, color = "darkblue")

#fit the data
fit <- kmeans(predictors, centers = 3, nstart = 15)
#display the k means object information
fit

# Display the cluster plot
fviz_cluster(fit, data = predictors)

# Calculate PCA
pca = prcomp(predictors)
rotated_data = as.data.frame(pca$x)
# Add original labels as a reference
rotated_data$type <- wineTrain$type
rotated_data$Clusters<- as.factor(fit$cluster)
# Plot and color by labels
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col =type)) + geom_point(alpha = 0.3)

# Plot and color by labels

ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = Clusters)) + geom_point()


#####part b

##create hyperparameter matrix
distance<- c('euclidean','manhattan')
aggMethod <- c('complete','average','centroid')
hyperMethod= expand.grid(distance=distance, aggMethod=aggMethod)
hyperMethod

#loop through parms values
library(doParallel)
library(gridExtra)
library(cowplot)


foreach(i=1:nrow(hyperMethod))%do%{
  d=hyperMethod[i,]$distance
  m=hyperMethod[i,]$aggMethod
  hfit<- hclust(dist(predictors, method = d),method=m)
  #perform the clustering and generate the silhouette plot. 
  nClust<- fviz_nbclust(predictors,FUN = hcut , method = "silhouette")
  numClust<-nClust$data
  maxCluster<-as.numeric(numClust$clusters[which.max(numClust$y)])
  h<- cutree(hfit,k= maxCluster)
  # Assign clusters as a new column
  fitKM<- kmeans(predictors,centers = maxCluster,nstart = 15)
  df1<- data.frame(Type = wineTrain$type, HAC = h, Kmeans = fitKM$cluster)
  #crosstab for km
  CT_km<- df1 %>% group_by(Kmeans) %>% select(Kmeans, Type) %>% table()
  CT_km
  
  }


foreach(i=1:nrow(hyperMethod))%do%{
  d=hyperMethod[i,]$distance
  m=hyperMethod[i,]$aggMethod
  hfit<- hclust(dist(predictors, method = d),method=m)
  #perform the clustering and generate the silhouette plot. 
  nClust<- fviz_nbclust(predictors,FUN = hcut , method = "silhouette")
  numClust<-nClust$data
  maxCluster<-as.numeric(numClust$clusters[which.max(numClust$y)])
  h<- cutree(hfit,k= maxCluster)
  # Assign clusters as a new column
  fitKM<- kmeans(predictors,centers = maxCluster,nstart = 15)
  df1<- data.frame(Type = wineTrain$type, HAC = h, Kmeans = fitKM$cluster)
  #crosstab for HAC
  CT_HAC<- df1 %>% group_by(HAC) %>% select(HAC, Type) %>% table()
  CT_HAC
}


# Calculate PCA
pca = prcomp(predictors)
rotated_data = as.data.frame(pca$x)
# Add original labels as a reference
rotated_data$type <- wineTrain$type
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col =type)) + 
  geom_point(alpha = 0.3)+
  labs(title = "PCA of yield contributing parameters")+
  theme(axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 8),
        title = element_text(size = 8))
#HAC
hfit<- hclust(dist(predictors, method = 'euclidean'),method='average')
fviz_nbclust(predictors,FUN = hcut , method = "silhouette")
hfitTree <- cutree(hfit,k=2) 
fitKM<- kmeans(predictors,centers = 2,nstart = 15)

rotated_data$HClusters = as.factor(hfitTree)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col =HClusters)) + 
  geom_point(alpha = 0.3)+
  labs(title = "PCA of HAC Cluster Label")+
  theme(axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 8),
        title = element_text(size = 8))
##Kmean
rotated_data$KMClusters = as.factor(fitKM$cluster)
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col=KMClusters)) + 
  geom_point(alpha = 0.3)+
  labs(title = "PCA of KMean Cluster Label")+
  theme(axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 8),
        title = element_text(size = 8))



ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = Clusters)) + geom_point()
k-means:
  rotated_data$Clusters = as.factor(fit$cluster)
# Plot and color by labels
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = Clusters)) + geom_point()


















  