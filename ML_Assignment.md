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
#setwd("C:/Users/zangl.f/Box Sync/private/Coursera/Data Science/Data Scientist Specialization/08 Practical Machine Learning")
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
corM <- cor(training[,-53])
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

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1248
##      2        1.5246             nan     0.1000    0.0852
##      3        1.4675             nan     0.1000    0.0670
##      4        1.4234             nan     0.1000    0.0545
##      5        1.3883             nan     0.1000    0.0502
##      6        1.3551             nan     0.1000    0.0431
##      7        1.3269             nan     0.1000    0.0358
##      8        1.3036             nan     0.1000    0.0312
##      9        1.2830             nan     0.1000    0.0329
##     10        1.2610             nan     0.1000    0.0322
##     20        1.1054             nan     0.1000    0.0178
##     40        0.9334             nan     0.1000    0.0103
##     60        0.8255             nan     0.1000    0.0081
##     80        0.7421             nan     0.1000    0.0056
##    100        0.6786             nan     0.1000    0.0045
##    120        0.6269             nan     0.1000    0.0033
##    140        0.5834             nan     0.1000    0.0019
##    150        0.5632             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1799
##      2        1.4904             nan     0.1000    0.1273
##      3        1.4063             nan     0.1000    0.1067
##      4        1.3389             nan     0.1000    0.0877
##      5        1.2830             nan     0.1000    0.0722
##      6        1.2365             nan     0.1000    0.0648
##      7        1.1945             nan     0.1000    0.0528
##      8        1.1595             nan     0.1000    0.0531
##      9        1.1270             nan     0.1000    0.0504
##     10        1.0963             nan     0.1000    0.0452
##     20        0.8860             nan     0.1000    0.0195
##     40        0.6758             nan     0.1000    0.0102
##     60        0.5540             nan     0.1000    0.0078
##     80        0.4649             nan     0.1000    0.0055
##    100        0.3982             nan     0.1000    0.0032
##    120        0.3446             nan     0.1000    0.0025
##    140        0.3029             nan     0.1000    0.0017
##    150        0.2864             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2390
##      2        1.4592             nan     0.1000    0.1558
##      3        1.3589             nan     0.1000    0.1306
##      4        1.2769             nan     0.1000    0.1012
##      5        1.2128             nan     0.1000    0.0888
##      6        1.1551             nan     0.1000    0.0729
##      7        1.1088             nan     0.1000    0.0816
##      8        1.0591             nan     0.1000    0.0559
##      9        1.0215             nan     0.1000    0.0583
##     10        0.9857             nan     0.1000    0.0543
##     20        0.7519             nan     0.1000    0.0215
##     40        0.5276             nan     0.1000    0.0109
##     60        0.4008             nan     0.1000    0.0071
##     80        0.3160             nan     0.1000    0.0040
##    100        0.2605             nan     0.1000    0.0028
##    120        0.2218             nan     0.1000    0.0025
##    140        0.1894             nan     0.1000    0.0019
##    150        0.1752             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1283
##      2        1.5247             nan     0.1000    0.0880
##      3        1.4677             nan     0.1000    0.0647
##      4        1.4249             nan     0.1000    0.0542
##      5        1.3892             nan     0.1000    0.0517
##      6        1.3570             nan     0.1000    0.0386
##      7        1.3306             nan     0.1000    0.0407
##      8        1.3052             nan     0.1000    0.0333
##      9        1.2834             nan     0.1000    0.0329
##     10        1.2611             nan     0.1000    0.0309
##     20        1.1065             nan     0.1000    0.0151
##     40        0.9350             nan     0.1000    0.0091
##     60        0.8283             nan     0.1000    0.0069
##     80        0.7484             nan     0.1000    0.0055
##    100        0.6860             nan     0.1000    0.0032
##    120        0.6352             nan     0.1000    0.0024
##    140        0.5909             nan     0.1000    0.0015
##    150        0.5711             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1814
##      2        1.4913             nan     0.1000    0.1228
##      3        1.4102             nan     0.1000    0.1029
##      4        1.3433             nan     0.1000    0.0845
##      5        1.2883             nan     0.1000    0.0717
##      6        1.2421             nan     0.1000    0.0616
##      7        1.2023             nan     0.1000    0.0596
##      8        1.1631             nan     0.1000    0.0531
##      9        1.1294             nan     0.1000    0.0537
##     10        1.0961             nan     0.1000    0.0375
##     20        0.8985             nan     0.1000    0.0180
##     40        0.6851             nan     0.1000    0.0095
##     60        0.5569             nan     0.1000    0.0050
##     80        0.4685             nan     0.1000    0.0068
##    100        0.4035             nan     0.1000    0.0023
##    120        0.3488             nan     0.1000    0.0034
##    140        0.3059             nan     0.1000    0.0019
##    150        0.2889             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2340
##      2        1.4612             nan     0.1000    0.1598
##      3        1.3613             nan     0.1000    0.1191
##      4        1.2846             nan     0.1000    0.1096
##      5        1.2153             nan     0.1000    0.0824
##      6        1.1631             nan     0.1000    0.0726
##      7        1.1169             nan     0.1000    0.0706
##      8        1.0715             nan     0.1000    0.0679
##      9        1.0301             nan     0.1000    0.0627
##     10        0.9908             nan     0.1000    0.0563
##     20        0.7589             nan     0.1000    0.0252
##     40        0.5332             nan     0.1000    0.0111
##     60        0.4100             nan     0.1000    0.0051
##     80        0.3245             nan     0.1000    0.0032
##    100        0.2670             nan     0.1000    0.0028
##    120        0.2236             nan     0.1000    0.0029
##    140        0.1877             nan     0.1000    0.0010
##    150        0.1739             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1276
##      2        1.5260             nan     0.1000    0.0867
##      3        1.4675             nan     0.1000    0.0668
##      4        1.4233             nan     0.1000    0.0496
##      5        1.3897             nan     0.1000    0.0495
##      6        1.3572             nan     0.1000    0.0456
##      7        1.3282             nan     0.1000    0.0378
##      8        1.3043             nan     0.1000    0.0341
##      9        1.2821             nan     0.1000    0.0320
##     10        1.2598             nan     0.1000    0.0285
##     20        1.1079             nan     0.1000    0.0179
##     40        0.9354             nan     0.1000    0.0093
##     60        0.8286             nan     0.1000    0.0066
##     80        0.7445             nan     0.1000    0.0044
##    100        0.6814             nan     0.1000    0.0042
##    120        0.6289             nan     0.1000    0.0036
##    140        0.5858             nan     0.1000    0.0021
##    150        0.5661             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1817
##      2        1.4889             nan     0.1000    0.1271
##      3        1.4053             nan     0.1000    0.1028
##      4        1.3391             nan     0.1000    0.0908
##      5        1.2824             nan     0.1000    0.0755
##      6        1.2351             nan     0.1000    0.0592
##      7        1.1970             nan     0.1000    0.0569
##      8        1.1608             nan     0.1000    0.0534
##      9        1.1266             nan     0.1000    0.0423
##     10        1.0987             nan     0.1000    0.0417
##     20        0.8960             nan     0.1000    0.0192
##     40        0.6819             nan     0.1000    0.0105
##     60        0.5531             nan     0.1000    0.0064
##     80        0.4639             nan     0.1000    0.0034
##    100        0.4011             nan     0.1000    0.0049
##    120        0.3459             nan     0.1000    0.0025
##    140        0.3051             nan     0.1000    0.0030
##    150        0.2859             nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2345
##      2        1.4596             nan     0.1000    0.1541
##      3        1.3596             nan     0.1000    0.1319
##      4        1.2766             nan     0.1000    0.1107
##      5        1.2080             nan     0.1000    0.0891
##      6        1.1503             nan     0.1000    0.0772
##      7        1.1018             nan     0.1000    0.0691
##      8        1.0570             nan     0.1000    0.0586
##      9        1.0194             nan     0.1000    0.0495
##     10        0.9878             nan     0.1000    0.0529
##     20        0.7561             nan     0.1000    0.0209
##     40        0.5270             nan     0.1000    0.0101
##     60        0.4075             nan     0.1000    0.0055
##     80        0.3227             nan     0.1000    0.0046
##    100        0.2636             nan     0.1000    0.0033
##    120        0.2209             nan     0.1000    0.0021
##    140        0.1890             nan     0.1000    0.0018
##    150        0.1751             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1292
##      2        1.5247             nan     0.1000    0.0839
##      3        1.4677             nan     0.1000    0.0647
##      4        1.4245             nan     0.1000    0.0525
##      5        1.3904             nan     0.1000    0.0493
##      6        1.3579             nan     0.1000    0.0407
##      7        1.3315             nan     0.1000    0.0417
##      8        1.3058             nan     0.1000    0.0299
##      9        1.2860             nan     0.1000    0.0331
##     10        1.2643             nan     0.1000    0.0307
##     20        1.1085             nan     0.1000    0.0163
##     40        0.9382             nan     0.1000    0.0094
##     60        0.8299             nan     0.1000    0.0064
##     80        0.7489             nan     0.1000    0.0053
##    100        0.6865             nan     0.1000    0.0039
##    120        0.6364             nan     0.1000    0.0032
##    140        0.5911             nan     0.1000    0.0030
##    150        0.5704             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1883
##      2        1.4887             nan     0.1000    0.1293
##      3        1.4060             nan     0.1000    0.1017
##      4        1.3401             nan     0.1000    0.0824
##      5        1.2858             nan     0.1000    0.0711
##      6        1.2386             nan     0.1000    0.0650
##      7        1.1972             nan     0.1000    0.0639
##      8        1.1565             nan     0.1000    0.0497
##      9        1.1250             nan     0.1000    0.0419
##     10        1.0978             nan     0.1000    0.0449
##     20        0.9018             nan     0.1000    0.0219
##     40        0.6928             nan     0.1000    0.0119
##     60        0.5653             nan     0.1000    0.0061
##     80        0.4721             nan     0.1000    0.0071
##    100        0.4014             nan     0.1000    0.0047
##    120        0.3496             nan     0.1000    0.0032
##    140        0.3103             nan     0.1000    0.0019
##    150        0.2931             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2367
##      2        1.4602             nan     0.1000    0.1566
##      3        1.3620             nan     0.1000    0.1220
##      4        1.2841             nan     0.1000    0.1047
##      5        1.2181             nan     0.1000    0.0903
##      6        1.1595             nan     0.1000    0.0681
##      7        1.1152             nan     0.1000    0.0669
##      8        1.0733             nan     0.1000    0.0721
##      9        1.0293             nan     0.1000    0.0600
##     10        0.9902             nan     0.1000    0.0460
##     20        0.7589             nan     0.1000    0.0280
##     40        0.5369             nan     0.1000    0.0116
##     60        0.4081             nan     0.1000    0.0057
##     80        0.3272             nan     0.1000    0.0068
##    100        0.2719             nan     0.1000    0.0027
##    120        0.2257             nan     0.1000    0.0023
##    140        0.1907             nan     0.1000    0.0015
##    150        0.1767             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2288
##      2        1.4601             nan     0.1000    0.1601
##      3        1.3577             nan     0.1000    0.1313
##      4        1.2768             nan     0.1000    0.1046
##      5        1.2112             nan     0.1000    0.0891
##      6        1.1554             nan     0.1000    0.0704
##      7        1.1104             nan     0.1000    0.0592
##      8        1.0722             nan     0.1000    0.0669
##      9        1.0308             nan     0.1000    0.0537
##     10        0.9973             nan     0.1000    0.0543
##     20        0.7596             nan     0.1000    0.0234
##     40        0.5341             nan     0.1000    0.0103
##     60        0.4067             nan     0.1000    0.0073
##     80        0.3250             nan     0.1000    0.0046
##    100        0.2707             nan     0.1000    0.0036
##    120        0.2277             nan     0.1000    0.0023
##    140        0.1945             nan     0.1000    0.0020
##    150        0.1799             nan     0.1000    0.0026
```

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
We will select the model to use on the validation, based on the model that has the lowest out-of-sample error. 

```r
p1 <- predict(m1, testing); p2 <- predict(m2, testing); 
p3 <- predict(m3, testing)

a1 <- confusionMatrix(testing$classe,p1)
a2 <- confusionMatrix(testing$classe,p2)
a3 <- confusionMatrix(testing$classe,p3)

round(a1$overall[[1]],3); round(a2$overall[[1]],3);
```

```
## [1] 0.502
```

```
## [1] 0.995
```

```r
round(a3$overall[[1]],3)
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
