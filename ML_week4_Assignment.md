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

```r
dim(src_training)
str(src_training)
summary(src_training)
sum(complete.cases(src_training))
```

We will therefore need to exclude the "classe" variable from the training set, as well as the record ID and the name of the subject. The model could include the possibility to recognize a pattern in which a certain subject is more likely to make certain type of errors. However, given the nature of the experiment described, we will assume that this is not the case.
We can notice that many variables have many 'N/A' values. Actually, out of these observations, only 406 have a value for each variable. All these variables reflect summarizing values (maximum, minimum, average, amplitude, variance, skewness, kurtosis, standard deviation) of time windows. 
We will exclude these, as we won't have these features available for any given observations, notably those of the validation set.
We will also exclude any time related feature, as the duration of the exercise, the time window they belong to or the time it took place is irrelevant to the error in executing the exercise.
Lastly, we will need to apply the feature selection to all 3 datasets. For now, we will leave 'classe' as part of the datasets and remove it when needed. The other variables are simply removed from the datasets as follows:

```r
excl_index <- grep("^(X|user_name|raw_timestamp_part_|cvtd_timestamp|new_window|num_window|kurtosis_|skewness_|max_|min_|amplitude_|var_|avg_|stddev_|var_)",colnames(training))
training <- training[,-excl_index]
testing <- testing[,-excl_index]
validation <- validation[,-excl_index]
```
Note that the validation dataset also as 53 variables. Although it doesn't have a classe defined, the last column includes the problem ID.

## Model selection & training
We're doing a classification exercise, and will therefore try 3 different methods better suited for classification, Decision Tree (rpart), Generalized Boosted Regression and Random Forest. We will consider an accuracy of .97 sufficient.
We will also reduce the cross-validation to 4, as the default values are to computationally intensive and the sample size of training as well as testing datasets are sufficiently large.


```r
set.seed(1977)
trControl <- trainControl(method = "cv", number = 4)
m1 <- train(classe ~ ., method = "rpart", trControl = trControl, data = training)
m2 <- train(classe ~ ., method = "rf", trControl = trControl, data = training)
m3 <- train(classe ~ ., method = "gbm", trControl = trControl, data = training)
```

Let's review the models we have created. With the below lines we can confirm that all 3 models are classification models that were created.
For the selection of the model, we will choose the error estimate based on the accuracy, as the final objective is to have the best possible accuracy in the final quiz, i.e. the sum of false positive and false negative over the full sample size. The error rate will be calculated as 1 - Accuracy.
We will therefore start to look into the prediction using the **training** set, which will allow us to calculate the in sample error rate, or Resubstitution Error. This shouldn't be the base for the model selection but will allow us to verify the expected out of sample rate.


```r
m1$modelType; m2$modelType; m3$modelType
```

```
## [1] "Classification"
```

```
## [1] "Classification"
```

```
## [1] "Classification"
```

```r
i1 <- predict(m1, training[,-53]); i2 <- predict(m2, training[,-53]); i3 <- predict(m3, training[,-53])

RE_1 <- confusionMatrix(training$classe,i1)
RE_2 <- confusionMatrix(training$classe,i2)
RE_3 <- confusionMatrix(training$classe,i3)

1-round(RE_1$overall[[1]],3); 1-round(RE_2$overall[[1]],3);1-round(RE_3$overall[[1]],3)
```

```
## [1] 0.506
```

```
## [1] 0
```

```
## [1] 0.026
```

We will select the model to use on the validation (the quiz), the actual out of sample, based on the model that has the lowest **estimated** out-of-sample error. Note, the testing data was part of the initial src_training set that we partitioned and is therefore actually part of the sample. However, it is the one we use to estimate what the out of sample error will be. To do so, we apply the models we have trained on the **testing** set.

```r
p1 <- predict(m1, testing[,-53]); p2 <- predict(m2, testing[,-53]); p3 <- predict(m3, testing[,-53])

a1 <- confusionMatrix(testing$classe,p1)
a2 <- confusionMatrix(testing$classe,p2)
a3 <- confusionMatrix(testing$classe,p3)

1-round(a1$overall[[1]],3); 1-round(a2$overall[[1]],3);1-round(a3$overall[[1]],3)
```

```
## [1] 0.498
```

```
## [1] 0.005
```

```
## [1] 0.033
```
**Firstly, we can compare the estimated out of sample error to the in sample error. We can see that for the Decision Tree (m1) the is not only very high, but also bigger than the estimated out of sample error. This tells us that we should already exclude m1 from our options. The 2 other models have as expected estimated out of sample errors that are higher than the in sample error.**
**Out of the 3 models, that Random Forest is the best performing model, with an out of sample error rate of 0.003, which is more than enough in terms of accuracy for the quiz. Hence, we won't need to stack our models to increase performance further and we can choose the Random Forest model 'm2' as the model we use to predict the classes for the validation set to use in the final quiz.**

## Prediction & conclusion

Based on the model assessment conducted above, we will use m2 to predict the classes for the validation set we use in the quiz 4. Below are the results of the prediction, with which the quiz could be completed with 100%. **This means that the out of sample error is lower (0.000) than the expected out of sample error.** This is driven by the very small size of the validation sample in conjunction with the low error rate.


```r
set.seed(1977)
p <- predict(m2, newdata = validation[,-53])
p
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
