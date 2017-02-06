# Example of a Random Forest Machine Learning Model in R
DL  
February 5, 2017  

# Introduction


This report presents an example of using the Random Forest machine learning model in R to predict how well people are doing a physical exercise. This report was written as coursework for part of the Data Science specialization by Johns Hopkins University on Coursera.

It will cover:

* How the data was prepared 
* The machine model chosen and how it was tested
* Model performance

# Data Preparation / Feature Selection

The steps for data preparation are as follows:

* Include any required libraries
* Download and read in the data
* Restrict the training to only what's present in the final prediction data set
* Remove other columns not needed for the model
* Split the data into training and test
* Create the required features


### Include required libraries

Before doing anything else, include the required libraries.


```r
library (caret)
```


### Download and read in the data

First download and import:

* the training data 
* the final prediction set (20 cases) -- the model will make predictions for these cases which are then submitted to Coursera as part of the marking scheme.


```r
trainingURL <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
predictionURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainingFileName <- "training.csv"
finalPredictionFileName <- "finalTest.csv"

download.file(trainingURL, destfile = trainingFileName, mode = "wb")
download.file(predictionURL, destfile = finalPredictionFileName, mode = "wb")

activityData <- read.csv (trainingFileName)
finalPredictionData <- read.csv (finalPredictionFileName)
```


### Restrict the training only to what's present in the final prediction data set


We should not be doing any data exploration on the final prediction set for example, to build features. However, pragmatically we need to make sure that the final model does not use any data columns that are not present in the prediction set. Otherwise, the mining model would not work. We need to remove any columns that are either:

* all blank
* all NA

Either 

* one could manually inspect the final prediction set and then remove those columns from the train (very unreliable)
* or write some code to identify those columns.

In this example, a short cut-can be taken but it's not guaranteed to work for all scenarios. In this case, the *nearZeroVar* function can be used. I manually checked the columns identified.


```r
nsv <- nearZeroVar(finalPredictionData)
```

Next let's check that the identified columns are the same in both the prediction and the training set (just to be cautious!)


```r
ifelse (
    sum(names(finalPredictionData)[nsv] == names(activityData)[nsv]) / length (nsv) == 1, 
    "Names match", 
    "Names did not match")
```

```
## [1] "Names match"
```

Those columns should now be removed from the training set as below.


```r
activityData <- activityData[,-nsv]
```

### Remove other columns not needed for the model

Next there are some columns in the activity data that we are not going to use. For example, the first column is a row identifier and should not be used.


```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"
```

These should now be removed.

```r
activityData <- activityData [,-c (1,2,3,4,5,6)]
```

### Split the data into training and test

The following code now splits the data into 70% training and 30% test.


```r
set.seed (8000)

inTrain <- createDataPartition(activityData$classe, p = 0.7, list= FALSE)

trainingDataSet <- activityData [inTrain,]
testingDataSet <- activityData [-inTrain,]
```

### Create features
The training set has a total of 52 possible predictors and 1 outcome. There were no further columns with zero or near zero variance.

No additional features were investigated as the first non-prototype model gave the required accuracy.

# Model

## Testing Strategy

* Split the training data into a 70-30 split
* Build the model over 70% of the data; calculate in-sample error with k-fold validation
* Use the test data (30%) to calculate out of sample error (compare predictions against actual outcomes)
* Create predictions for the final 20 cases; upload to Coursera website; obtain results through the website


## Model created
I first built a decision tree to prototype the end-to-end process. This gave very poor results (<50% out of sample error).

I then used a random forest over all attributes which gave satisfactory results but did require performance tuning.

The random forest algorithm samples a number of attributes each times it splits the tree. By default this is the sqrt (number of features). In this scenario (> 50 features), that would mean at least 7 features. Reducing this to 3 reduced producing time to less than 25% with no change in accuracy. The parameter is *mtry* and can be found on page 18 of the documentation. 

[https://cran.r-project.org/web/packages/randomForest/randomForest.pdf](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf)

For k-fold validation, rather than writing code to sample the data and iterate through the different folds, I used the training control parameter (trControl) to confure configure cross validation.


```r
config <- data.frame (mtry=3)
tc <- trainControl (method="cv", number = 10)
model <- train (classe ~ ., method = "rf", prox = TRUE, 
                data = trainingDataSet,
                trControl = tc,
                tuneGrid = config)
```

## Model performance





### Summary of Performance Metrics

Metric                                                              |Value                        |
--------------------------------------------------------------------|-----------------------------|
In Sample Error                                                     |0.993%       |
Out of Sample Error                                                 |0.007%    |
Final error rate (by submitting predictions for the 20 blind cases) |0%          |


### Method to create the predictions


```r
testPredictions <- predict (model, testingDataSet)
finalPredictions <- predict (model, finalPredictionData)
```

### Method used to derive the metrics


```r
inSampleErrorRate <- model$results[2]

outOfSampleErrorRate <- 1 - 
  confusionMatrix(testPredictions,
                  testingDataSet$classe)$overall["Accuracy"]

finalErrorRate <- 1 - (20/20)
```

### Method used to create the final predictions
The code supplied by the project was used


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
```

The following code then writes them out.


```r
if (dir.exists ("submission") == FALSE)
  dir.create ("submission")

setwd ("submission")

pml_write_files (finalPredictions)

setwd ("..")
```
