Simple GLM-Random Forest Models
================
Connor Duplessis
1/16/2018

``` r
library(tidyverse)
library(caret)
library(ranger)
```

Read in Datasets
----------------

``` r
train <- read_csv("train.csv", col_types = cols(Embarked = col_factor(levels = c("S","C",
"Q")), Pclass = col_factor(levels = c("1", "2", "3")), Sex = col_factor(levels = c("male", 
"female")), Survived = col_factor(levels = c("0", "1"))))

test <- read_csv("test.csv", col_types = cols(Embarked = col_factor(levels = c("Q","S", "C")), 
Pclass = col_factor(levels = c("1","2", "3")), Sex = col_factor(levels = c("male","female"))))
```

Data Cleaning
-------------

``` r
#Find NA's
head(which(is.na(train), arr.ind = TRUE))
```

    ##      row col
    ## [1,]   6   6
    ## [2,]  18   6
    ## [3,]  20   6
    ## [4,]  27   6
    ## [5,]  29   6
    ## [6,]  30   6

``` r
#Remove Cabin Column
train_clean <- train[,-11]

#Find median age and add in
median(train_clean$Age, na.rm = TRUE)
```

    ## [1] 28

``` r
train_clean[,6][is.na(train_clean[,6])] <- 28

#Find NA's again
head(which(is.na(train_clean), arr.ind = TRUE))
```

    ##      row col
    ## [1,]  62  11
    ## [2,] 830  11

``` r
#table of different embarked locations
table(train_clean$Embarked)
```

    ## 
    ##   S   C   Q 
    ## 644 168  77

``` r
#Majority left from "S" so assign the NA's to there
train_clean[,11][is.na(train_clean[,11])] <- "S"

#Look for NA's once more
which(is.na(train_clean), arr.ind = TRUE)
```

    ##      row col

``` r
#Drop ticket and name column
train_clean <- train_clean[,-c(4,9)]

#Clean test set in same fashion
head(which(is.na(test), arr.ind = TRUE))
```

    ##      row col
    ## [1,]  11   5
    ## [2,]  23   5
    ## [3,]  30   5
    ## [4,]  34   5
    ## [5,]  37   5
    ## [6,]  40   5

``` r
#Remove Cabin column
test_clean <- test[,-10]
#Replace median age
test_clean[,5][is.na(test_clean[,5])] <- 28
#Look for NA's
head(which(is.na(test_clean), arr.ind = TRUE))
```

    ##      row col
    ## [1,] 153   9

``` r
#Find median Fare, add to missing value
median(test_clean$Fare, na.rm = TRUE)
```

    ## [1] 14.4542

``` r
test_clean[,9][is.na(test_clean[,9])] <- 14.45
#Look for NA's
which(is.na(test_clean), arr.ind = TRUE)
```

    ##      row col

``` r
#Add Survived Column
test_clean$Survived <- 0
#Drop name and ticket column
test_clean <- test_clean[,-c(3,8)]
```

GLM Models
----------

``` r
set.seed(42)
GLM_Model <- train(Survived ~ Sex + Age, train_clean, method = "glm", trControl = trainControl(method = "cv", number = 5))
GLM_Model2 <- train(Survived ~ Sex + Age + Pclass, train_clean, method = "glm", trControl = trainControl(method = "cv", number = 5))
GLM_Model3 <- train(Survived ~ ., train_clean, method = "glm", trControl = trainControl(method = "cv", number = 5))
#GLM_Model2 came in with the highest accuracy

GLM_Model2
```

    ## Generalized Linear Model 
    ## 
    ## 891 samples
    ##   3 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 713, 712, 713, 712, 714 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.7845686  0.5425605

``` r
GLM_Prediction <- predict(GLM_Model2, test_clean, type = "raw")
GLM_Submission <- data.frame(PassengerID = test_clean$PassengerId, Survived = GLM_Prediction)
write.csv(GLM_Submission, file = "Simple_GLM_Submission.csv", row.names = FALSE)
```

Random Forest Models
--------------------

``` r
set.seed(42)
RF_Model <- train(Survived ~ ., data = train_clean, method = "ranger", trControl = trainControl(method = "cv", number = 5))
RF_Model2 <- train(Survived ~ ., data = train_clean, method = "ranger", tuneLength = 10, trControl = trainControl(method = "cv", number = 5))
```

    ## note: only 9 unique complexity parameters in default grid. Truncating the grid to 9 .

``` r
#Model2 came with higher accuracy
RF_Model2
```

    ## Random Forest 
    ## 
    ## 891 samples
    ##   8 predictor
    ##   2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 713, 712, 713, 712, 714 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  splitrule   Accuracy   Kappa    
    ##    2    gini        0.8215722  0.6091622
    ##    2    extratrees  0.8170650  0.5935582
    ##    3    gini        0.8260225  0.6228751
    ##    3    extratrees  0.8193186  0.6017798
    ##    4    gini        0.8282949  0.6282284
    ##    4    extratrees  0.8170398  0.6014273
    ##    5    gini        0.8237815  0.6194212
    ##    5    extratrees  0.8293806  0.6312146
    ##    6    gini        0.8159036  0.6023245
    ##    6    extratrees  0.8271523  0.6264208
    ##    7    gini        0.8226642  0.6185743
    ##    7    extratrees  0.8248924  0.6228924
    ##    8    gini        0.8215468  0.6164228
    ##    8    extratrees  0.8237941  0.6211436
    ##    9    gini        0.8204232  0.6133643
    ##    9    extratrees  0.8238005  0.6211489
    ##   10    gini        0.8226769  0.6189516
    ##   10    extratrees  0.8260476  0.6264212
    ## 
    ## Tuning parameter 'min.node.size' was held constant at a value of 1
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were mtry = 5, splitrule =
    ##  extratrees and min.node.size = 1.

``` r
RF_Prediction <- predict(RF_Model2, test_clean, type = "raw")
RF_Submission <- data.frame(PassengerID = test_clean$PassengerId, Survived = RF_Prediction)
write.csv(RF_Submission, file = "First_RF_Submission.csv", row.names = FALSE)
```

Conclusion
----------

GLM\_Model2 and RF\_Model2 were submitted to Kaggle and both had an accuracy of 75.12%. As of January 16, 2017 that is only good enough for the 82nd percentile. In the future, I will be implementing features into the models such as custom grid tuning, feature engineering, and different model types to hopefully improve these scores.
