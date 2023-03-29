library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(readr)
library(ggpubr)
library(psych)
library(readr)
library(rpart)
library(tibble)
library(doSNOW)
library(foreach)
bc <- read_csv("Downloads/breast_cancer_updated.csv")
View(bc)
str(bc)
summary(bc)

#Preprocessing 
#remove the IDNumber column and exclude rows with NA from the dataset.
bc <- bc %>%
  select(-1) %>%
  na.omit()
sum(is.na(bc))

#convert class label to factor 
unique(bc$Class)
bc[sapply(bc,is.character)]<-lapply(
  bc[sapply(bc,is.character)],as.factor)

# a. Apply decision tree & report the accuracy using 10-fold cross validation.
## splitting into train&test set 
set.seed(961)
index= createDataPartition(y=bc$Class, p=0.7, list = FALSE)
train_bc <- bc[index,]
test_bc <- bc[-index,]

summary(train_bc)
str(train_bc)

##build model
idClass <- createFolds(train_bc$Class, k=10, returnTrain = TRUE)
train_control<- trainControl(index=idClass, method = "cv", number = 10)
##Fit the model 
tree1<- train(Class ~., data=train_bc, method="rpart", trControl=train_control)
tree1
## evaluate the fit with test set
pred_tree1 <- predict(tree1, test_bc)
confusionMatrix(test_bc$Class, pred_tree1)

##evaluate the fit with train set 
pred_tree1_train <- predict(tree1, train_bc)
confusionMatrix(train_bc$Class, pred_tree1_train)

#b. Generate a visualization of the decision tree.
install.packages("rattle")
library(rattle)
fancyRpartPlot(tree1$finalModel,main="DT of Breast Cancer Data",caption = "")


#PS2
#You will be using the storms data, a subset of the NOAA Atlantic hurricane 
#database , which includes the positions and attributes of 198 tropical storms
#(potential hurricanes), measured every six hours during the lifetime of a storm. 
data("storms", package = "dplyr")

#check for missing values & convert chr into categorical 
storms[sapply(storms,is.character)]<-lapply(
  storms[sapply(storms,is.character)],as.factor)
str(stormsDS)

stormsDS <- storms%>%
  as.data.frame()%>%
  na.omit()
(colMeans(is.na(storms)))*100

##Imbalanced class
ggplot(stormsDS, aes(category)) + 
  geom_bar(fill="lightblue")+
  labs(x="Category", title ="Counts by Hurrican Categories")+
  theme(plot.title=element_text(size = 9))+
  theme(axis.title.x=element_text(size = 8))+
  theme(axis.title.y=element_text(size = 8))

prop.table(table(stormsDS$category))
  
#a.	Build a decision tree using the following hyperparameters, maxdepth=2, 
#minsplit=5 and minbucket=3. Be careful to use the right method of training 
#so that you are not automatically tuning the cp parameter, but you are 
#controlling the aforementioned parameters specifically. 

set.seed(961)
#set parameter
idCategory <- createFolds(stormsDS$category, k=10, returnTrain = TRUE)
train_control<- trainControl(index=idCategory, method = "cv", number = 10)
hypers<- rpart.control(minsplit = 5, maxdepth = 2, minbucket = 3)
##Fit the model 
stormDT<- train(category ~., data=stormsDS, method="rpart1SE", 
                control=hypers, trControl=train_control)
stormDT

## evaluate the fit with whole set
pred_storm1 <- predict(stormDT, stormsDS)
confusionMatrix(stormsDS$category, pred_storm1)
fancyRpartPlot(stormDT$finalModel,main="Initial DT of Storms",caption = "")

##train-test split
index<- createDataPartition(y=stormsDS$category, p=0.7, list = FALSE)
train_storm <- stormsDS[index,]
test_storm <- stormsDS[-index,]
str(train_storm)
ggplot(train_storm, aes(category)) + 
  geom_bar(fill="lightblue")+
  labs(x="Category", title ="Counts by Hurrican Categories")+
  theme(plot.title=element_text(size = 9))+
  theme(axis.title.x=element_text(size = 8))+
  theme(axis.title.y=element_text(size = 8))

#set parameter
idCategory <- createFolds(train_storm$category, k=10, returnTrain = TRUE)
train_control<- trainControl(index=idCategory, method = "cv", number = 10)
hypers<- rpart.control(minsplit = 5, maxdepth = 2, minbucket = 3)
##Fit the model 
trainDT<- train(category ~., data=train_storm, method="rpart1SE", 
                control=hypers, trControl=train_control)
trainDT

## evaluate the fit with test set
pred_treeStorm <- predict(trainDT, test_storm)
confusionMatrix(test_storm$category, pred_treeStorm)

##evaluate the fit with train set 
pred_treeStorm_train <- predict(trainDT, train_storm)
confusionMatrix(train_storm$category, pred_treeStorm_train)

###############################
#PS3
##train-test split
set.seed(961)
indexStorm<- createDataPartition(y=stormsDS$category, p=0.8, list = FALSE)
train_storm <- stormsDS[indexStorm,]
test_storm <- stormsDS[-indexStorm,]
str(train_storm)
prop.table(table(train_storm$category))
ggplot(train_storm, aes(category)) + 
  geom_bar(fill="lightblue")+
  labs(x="Category", title ="Counts by Hurrican Categories in train set(80%)")+
  theme(plot.title=element_text(size = 9))+
  theme(axis.title.x=element_text(size = 8))+
  theme(axis.title.y=element_text(size = 8))

ggplot(test_storm, aes(category)) + 
  geom_bar(fill="lightblue")+
  labs(x="Category", title ="Counts by Hurrican Categories in test set(20%)")+
  theme(plot.title=element_text(size = 9))+
  theme(axis.title.x=element_text(size = 8))+
  theme(axis.title.y=element_text(size = 8))

##initialize stratified cross validation 
idCategory <- createFolds(train_storm$category, k=10, returnTrain = TRUE)
train_control<- trainControl(index=idCategory, method = "cv", number = 10)

##create hyperparameter matrix
maxdepth<- c(3,5,7,10)
minsplit<- c(20,60,100)
minbucket<- c(5,20,35)
hyperparms= expand.grid(maxdepth=maxdepth, minsplit=minsplit,minbucket=minbucket)
#loop through parms values
library(doParallel)
results=foreach(i=1:nrow(hyperparms),.combine=rbind)%dopar%{
  d=hyperparms[i,]$maxdepth
  s=hyperparms[i,]$minsplit
  b=hyperparms[i,]$minbucket
  fit= train(category ~., data=train_storm, method="rpart1SE", 
             control=rpart.control(minsplit = s, maxdepth = d, minbucket = b),
             trControl=train_control)
  pred_train<- predict(fit, train_storm)
  accuracy_train<- (confusionMatrix(train_storm$category, pred_train))$overall[1]
  pred=predict(fit, test_storm)
  accuracy_test<- (confusionMatrix(test_storm$category, pred))$overall[1]   
  node<- nrow(fit$finalModel$frame)
  data.frame(Node=node, AccuracyTrain=accuracy_train, 
             AccuracyTestn=accuracy_test)
}

hyperparms[which.min(results$AccuracyTrain),]
comp<-cbind(hyperparms,results)
View(comp)

hyperparms[which.max(results$AccuracyTestn),]
hyperparms[which.max(results$AccuracyTrain),]

fit= train(category ~., data=train_storm, method="rpart1SE", 
           control=rpart.control(minsplit = 20, maxdepth = 7, minbucket = 5),
           trControl=train_control,metric="Accuracy")
predFinal=predict(fit, test_storm)
accuracy_test<- confusionMatrix(test_storm$category, predFinal)  
accuracy_test
fancyRpartPlot(fit$finalModel,main="DT of Storms Data",caption = "")


##plot
comp_reshape <- data.frame(y=comp$AccuracyTestn,
                           x=c(comp$maxdepth,comp$minbucket,comp$minsplit),
                           group=c(rep("maxdepth",nrow(comp)),
                                   rep("minbucket",nrow(comp)),
                                   rep("minsplit",nrow(comp))))
ggplot(comp_reshape,aes(x,y,col=group))+
  geom_line()
pairs.panels(comp, gap=0, bg=c('red','yellow', 'blue'),pch = 21)



#################################
##PS4
#As a preprocessing step, remove the ID column and make sure to convert the 
#target variable, approval, from a string to a factor. 
Bank_Modified <- read_csv("Desktop/Bank_Modified.csv")
Bank_Modified <- Bank_Modified %>%
  select(-1) %>%
  na.omit()
Bank_Modified[sapply(Bank_Modified,is.character)]<-lapply(
  Bank_Modified[sapply(Bank_Modified,is.character)],as.factor)

Bank_Modified[sapply(Bank_Modified,is.logical)]<-lapply(
  Bank_Modified[sapply(Bank_Modified,is.logical)],as.factor)
str(Bank_Modified)

Bank_Modified %>%
  ggplot(aes(x="", fill=approval))+
  geom_bar(position = "fill",width = 1)+
  coord_polar(theta = "y")+
  labs(x="",y="") +
  theme(text = element_text(size = 9))

#a. Build your initial decision tree model with minsplit=10 and maxdepth=20 
#set parameter
##train-test split
set.seed(961)
index<- createDataPartition(y=Bank_Modified$approval, p=0.8, list = FALSE)
trainBank <- Bank_Modified[index,]
testBank<- Bank_Modified[-index,]

idApproval <- createFolds(trainBank$approval, k=10, returnTrain = TRUE)
train_control<- trainControl(index=idApproval, method = "cv", number = 10)
hypers<- rpart.control(minsplit = 10, maxdepth = 20)

##Fit the model 
initFit<- train(approval ~., data=trainBank, method="rpart1SE", 
                control=hypers, trControl=train_control,preProcess=c("center","scale"))
initFit
### evaluate the fit with test set
predinit <- predict(initFit, testBank)
confusionMatrix(testBank$approval, predinit)
fancyRpartPlot(initFit$finalModel,main="DT of Initial Bank Data",caption = "")

#relevance analysis 
var_imp <- varImp(initFit,scale = FALSE)
var_imp
plot(var_imp)

str(Bank_Modified)

#b. model tuning
tunedBank <- trainBank[,c("bool1","cont4", "bool2","cont3","ages","cont2","approval")]

tunedTest <- testBank[,c("bool1","cont4", "bool2","cont3","ages","cont2","approval")]

idTuned <- createFolds(tunedBank$approval, k=10, returnTrain = TRUE)
train_controlTuned<- trainControl(index=idTuned, method = "cv", number = 10)
tunedFit<- train(approval ~., data=tunedBank, method="rpart1SE", 
                control=hypers, trControl=train_controlTuned,preProcess=c("center","scale"))
tunedFit

### evaluate the fit with test set
predTrain <- predict(tunedFit, tunedBank)
confusionMatrix(tunedBank$approval, predTrain)

### evaluate the fit with test set
predT <- predict(tunedFit, tunedTest)
confusionMatrix(tunedTest$approval, predT)
fancyRpartPlot(tunedFit$finalModel,main="DT of Tuned Bank Data",caption = "")














