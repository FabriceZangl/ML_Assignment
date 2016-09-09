# Predicting the right class of exercise
Fabrice Zangl  
September 9, 2016  
## Introduction 
The objective of this project is to predict the class of a physical exercise. 
The physical exercise is to lift a weight. The class of the phyiscal exercise is the way in which the exercise is executed, which can be correct or with some form of error. The actual class of the physical exercise should be predicted by the measurements of the accelerometers. The accelerometers were located on 4 spots ( belt, forearm, arm, and dumbell) of 6 participants. 
This could for instance be used to support an automatic training coach. Such a digital coach would make suggestions on how to correct the movements to correctly execute the weight lifting exercise, for instance to avoid injuries. Of course, we would want the digital coach to correctly identify the wrong move, so that it wouldn't give wrong advice, possibly increasing the risk of injury.

## Data Gathering and Partitioning
Further details and description about the data can be found on the website describing the research experiment: <http://groupware.les.inf.puc-rio.br/har>. We will now first gather the data from the links provided in the assignment:

```r
url_training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url_training, "assignment_training.csv")
src_training <- read.csv("assignment_training.csv")
url_validation <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url_validation, "assignment_validation.csv")
validation <- read.csv("assignment_validation.csv")
```

The 20 observations in the "pml-testing.csv" file were stored in the data frame named 'validation'. For the training and testing of the different model options, we therefore need to further partition the original training data into a training and a testing set. With the testing set, we will be able to evaluate the out-of-sample error of the different models we want to evaluate.

```r
set.seed(1977)
if(!require(caret)) {install.packages("caret"); require(caret)}
inTrain <- createDataPartition(y = src_training$classe, p = 0.75, list = F)
training <- src_training[inTrain,]
testing <- src_training[-inTrain,]
```

## Feature selection & Pre-Processing

Before training the model, we will review the data. Using the below 4 functions in conjunction with the research description on the above website, we can conclude that the original training set is composed of 19,622 observations with 160 variables, including the variable "classe", which is the dependant variable we will want to predict in a classification exercise. 


We will therefore need to exclude the "classe" variable from the training set, as well as the record ID and the name of the subject. The model could include the possibility to recognize a pattern in which a certain subject is more likely to make certain type of errors. However, given the nature of the experiment described, we will assume that this is not the case.
We can notice that many variables have many 'N/A' values. Actually, out of these observations, only 406 have a value for each variable. All these variables reflect summarizing values (maximum, minimum, average, amplitude, variance, skewness, kurtosis, standard deviation) of time windows. 
We will exclude these, as we won't have these features available for any given observations, notably those of the validation set.
We will also exclude any time related feature, as the duration of the exercise, the time window they belong to or the time it took place is irrelevant to the error in executing the exercise.
Lastly, we will need to apply the feature selection to all 3 datasets. For now, we will leave 'classe' as part of the datasets and remove it when needed. The other variables are simply removed from the datasets as follows:

Note that the validation dataset also as 53 variables. Although it doesn't have a classe defined, the last column includes the problem ID.

## Model selection & training, including pre-rocessing
We're doing a classification exercise, and will therefore try 3 different methods better suited for classification, Decision Tree (rpart), Generalized Boosted Regression and Random Forest. We will consider an accuracy of .97 sufficient.
We will also reduce the cross-validation to 4, in order to reduce our out of sample error rate.


```r
set.seed(1977)
trControl <- trainControl(method = "cv", number = 4)
m1 <- train(classe ~ ., method = "rpart", trControl = trControl, data = training)
```

```
## Loading required package: rpart
```

```r
m2 <- train(classe ~ ., method = "rf", trControl = trControl, data = training)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
m3 <- train(classe ~ ., method = "gbm", trControl = trControl, data = training)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```r
m1$modelType; m2$modelType; m3$modelType
summary(m1); m2$modelType; m3$modelType
```
We will select the model to use on the validation, based on the model that has the lowest out-of-sample error. 

```r
p1 <- predict(m1, testing); p2 <- predict(m2, testing); p3 <- predict(m3, testing)

a1 <- confusionMatrix(testing$classe,p1)
a2 <- confusionMatrix(testing$classe,p2)
a3 <- confusionMatrix(testing$classe,p3)

round(a1$overall[[1]],3); round(a2$overall[[1]],3);round(a3$overall[[1]],3)
```

```
## [1] 0.502
```

```
## [1] 0.995
```

```
## [1] 0.967
```
**The outcome confirms that all 3 models are classification models. Out of the 3 models, that Random Forest is the best performing model, with an out of sample rate of 0.997, which is more than enough in terms of accuracy. Hence, we won't need to stack our models to increase performance further and we can choose the Random Forest model 'm2' as the model we use to predict the classes for the validation set to use in the final quiz.**

## Prediction & conclusion

Based on the model assessment conducted above, we will use m2 to predict the classes for the validation set we use in the quiz 4. Below are the results of the prediction, with which the quiz could be completed with 100%.


```r
set.seed(1977)
p <- predict(m2, newdata = validation)
p
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
