---
title: "Practical Machine Learning - Coursera Project"
author: "Yosuke Ishizaka"

---
##Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website
here: http://groupware.les.inf.puc?rio.br/har (http://groupware.les.inf.puc?rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

##Summary
The goal of this project is to predict the manner in which they did the exercise.  The `classe` which we are trying to predict are classified in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Downloading training and testing data.

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
# Read CSV files into data frames. We'll use trainData to build a model to predict 20 different test cases in testData.
trainData <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
testData <- read.csv("pml-testing.csv")
```



Explore our trainData dataset which we'll use to build a model.

```r
dim(trainData)
```

```
## [1] 19622   160
```

```r
summary(trainData)
```
Basic data transformation. Goal is to remove all NA values.
Count NA's.  Count includes values that were blank or division by zero.

```r
NAcount = vector(mode="integer", length=0)
for(i in 1:length(trainData)) {
        NAcount[i] = sum(is.na(trainData[, i]))
}
table(NAcount)
```

```
## NAcount
##     0 19216 19217 19218 19220 19221 19225 19226 19227 19248 19293 19294 
##    60    67     1     1     1     4     1     4     2     2     1     1 
## 19296 19299 19300 19301 19622 
##     2     1     4     2     6
```
There are 67 columns with 19216 NA values. In total there are 100 columns with at least 19216 NA's. At least 97% of the observation in these parameters are NA's.

```r
removeNA <- which(NAcount/nrow(trainData) > .97)
```
Remove NA columns in trainData dataset.

```r
trainData <- trainData[, -removeNA]
summary(trainData)
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :3892   Min.   :1.322e+09    Min.   :   294      
##  1st Qu.: 4906   carlitos:3112   1st Qu.:1.323e+09    1st Qu.:252912      
##  Median : 9812   charles :3536   Median :1.323e+09    Median :496380      
##  Mean   : 9812   eurico  :3070   Mean   :1.323e+09    Mean   :500656      
##  3rd Qu.:14717   jeremy  :3402   3rd Qu.:1.323e+09    3rd Qu.:751891      
##  Max.   :19622   pedro   :2610   Max.   :1.323e+09    Max.   :998801      
##                                                                           
##           cvtd_timestamp  new_window    num_window      roll_belt     
##  28/11/2011 14:14: 1498   no :19216   Min.   :  1.0   Min.   :-28.90  
##  05/12/2011 11:24: 1497   yes:  406   1st Qu.:222.0   1st Qu.:  1.10  
##  30/11/2011 17:11: 1440               Median :424.0   Median :113.00  
##  05/12/2011 11:25: 1425               Mean   :430.6   Mean   : 64.41  
##  02/12/2011 14:57: 1380               3rd Qu.:644.0   3rd Qu.:123.00  
##  02/12/2011 13:34: 1375               Max.   :864.0   Max.   :162.00  
##  (Other)         :11007                                               
##    pitch_belt          yaw_belt       total_accel_belt  gyros_belt_x      
##  Min.   :-55.8000   Min.   :-180.00   Min.   : 0.00    Min.   :-1.040000  
##  1st Qu.:  1.7600   1st Qu.: -88.30   1st Qu.: 3.00    1st Qu.:-0.030000  
##  Median :  5.2800   Median : -13.00   Median :17.00    Median : 0.030000  
##  Mean   :  0.3053   Mean   : -11.21   Mean   :11.31    Mean   :-0.005592  
##  3rd Qu.: 14.9000   3rd Qu.:  12.90   3rd Qu.:18.00    3rd Qu.: 0.110000  
##  Max.   : 60.3000   Max.   : 179.00   Max.   :29.00    Max.   : 2.220000  
##                                                                           
##   gyros_belt_y       gyros_belt_z      accel_belt_x       accel_belt_y   
##  Min.   :-0.64000   Min.   :-1.4600   Min.   :-120.000   Min.   :-69.00  
##  1st Qu.: 0.00000   1st Qu.:-0.2000   1st Qu.: -21.000   1st Qu.:  3.00  
##  Median : 0.02000   Median :-0.1000   Median : -15.000   Median : 35.00  
##  Mean   : 0.03959   Mean   :-0.1305   Mean   :  -5.595   Mean   : 30.15  
##  3rd Qu.: 0.11000   3rd Qu.:-0.0200   3rd Qu.:  -5.000   3rd Qu.: 61.00  
##  Max.   : 0.64000   Max.   : 1.6200   Max.   :  85.000   Max.   :164.00  
##                                                                          
##   accel_belt_z     magnet_belt_x   magnet_belt_y   magnet_belt_z   
##  Min.   :-275.00   Min.   :-52.0   Min.   :354.0   Min.   :-623.0  
##  1st Qu.:-162.00   1st Qu.:  9.0   1st Qu.:581.0   1st Qu.:-375.0  
##  Median :-152.00   Median : 35.0   Median :601.0   Median :-320.0  
##  Mean   : -72.59   Mean   : 55.6   Mean   :593.7   Mean   :-345.5  
##  3rd Qu.:  27.00   3rd Qu.: 59.0   3rd Qu.:610.0   3rd Qu.:-306.0  
##  Max.   : 105.00   Max.   :485.0   Max.   :673.0   Max.   : 293.0  
##                                                                    
##     roll_arm         pitch_arm          yaw_arm          total_accel_arm
##  Min.   :-180.00   Min.   :-88.800   Min.   :-180.0000   Min.   : 1.00  
##  1st Qu.: -31.77   1st Qu.:-25.900   1st Qu.: -43.1000   1st Qu.:17.00  
##  Median :   0.00   Median :  0.000   Median :   0.0000   Median :27.00  
##  Mean   :  17.83   Mean   : -4.612   Mean   :  -0.6188   Mean   :25.51  
##  3rd Qu.:  77.30   3rd Qu.: 11.200   3rd Qu.:  45.8750   3rd Qu.:33.00  
##  Max.   : 180.00   Max.   : 88.500   Max.   : 180.0000   Max.   :66.00  
##                                                                         
##   gyros_arm_x        gyros_arm_y       gyros_arm_z       accel_arm_x     
##  Min.   :-6.37000   Min.   :-3.4400   Min.   :-2.3300   Min.   :-404.00  
##  1st Qu.:-1.33000   1st Qu.:-0.8000   1st Qu.:-0.0700   1st Qu.:-242.00  
##  Median : 0.08000   Median :-0.2400   Median : 0.2300   Median : -44.00  
##  Mean   : 0.04277   Mean   :-0.2571   Mean   : 0.2695   Mean   : -60.24  
##  3rd Qu.: 1.57000   3rd Qu.: 0.1400   3rd Qu.: 0.7200   3rd Qu.:  84.00  
##  Max.   : 4.87000   Max.   : 2.8400   Max.   : 3.0200   Max.   : 437.00  
##                                                                          
##   accel_arm_y      accel_arm_z       magnet_arm_x     magnet_arm_y   
##  Min.   :-318.0   Min.   :-636.00   Min.   :-584.0   Min.   :-392.0  
##  1st Qu.: -54.0   1st Qu.:-143.00   1st Qu.:-300.0   1st Qu.:  -9.0  
##  Median :  14.0   Median : -47.00   Median : 289.0   Median : 202.0  
##  Mean   :  32.6   Mean   : -71.25   Mean   : 191.7   Mean   : 156.6  
##  3rd Qu.: 139.0   3rd Qu.:  23.00   3rd Qu.: 637.0   3rd Qu.: 323.0  
##  Max.   : 308.0   Max.   : 292.00   Max.   : 782.0   Max.   : 583.0  
##                                                                      
##   magnet_arm_z    roll_dumbbell     pitch_dumbbell     yaw_dumbbell     
##  Min.   :-597.0   Min.   :-153.71   Min.   :-149.59   Min.   :-150.871  
##  1st Qu.: 131.2   1st Qu.: -18.49   1st Qu.: -40.89   1st Qu.: -77.644  
##  Median : 444.0   Median :  48.17   Median : -20.96   Median :  -3.324  
##  Mean   : 306.5   Mean   :  23.84   Mean   : -10.78   Mean   :   1.674  
##  3rd Qu.: 545.0   3rd Qu.:  67.61   3rd Qu.:  17.50   3rd Qu.:  79.643  
##  Max.   : 694.0   Max.   : 153.55   Max.   : 149.40   Max.   : 154.952  
##                                                                         
##  total_accel_dumbbell gyros_dumbbell_x    gyros_dumbbell_y  
##  Min.   : 0.00        Min.   :-204.0000   Min.   :-2.10000  
##  1st Qu.: 4.00        1st Qu.:  -0.0300   1st Qu.:-0.14000  
##  Median :10.00        Median :   0.1300   Median : 0.03000  
##  Mean   :13.72        Mean   :   0.1611   Mean   : 0.04606  
##  3rd Qu.:19.00        3rd Qu.:   0.3500   3rd Qu.: 0.21000  
##  Max.   :58.00        Max.   :   2.2200   Max.   :52.00000  
##                                                             
##  gyros_dumbbell_z  accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z 
##  Min.   : -2.380   Min.   :-419.00   Min.   :-189.00   Min.   :-334.00  
##  1st Qu.: -0.310   1st Qu.: -50.00   1st Qu.:  -8.00   1st Qu.:-142.00  
##  Median : -0.130   Median :  -8.00   Median :  41.50   Median :  -1.00  
##  Mean   : -0.129   Mean   : -28.62   Mean   :  52.63   Mean   : -38.32  
##  3rd Qu.:  0.030   3rd Qu.:  11.00   3rd Qu.: 111.00   3rd Qu.:  38.00  
##  Max.   :317.000   Max.   : 235.00   Max.   : 315.00   Max.   : 318.00  
##                                                                         
##  magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z  roll_forearm      
##  Min.   :-643.0    Min.   :-3600     Min.   :-262.00   Min.   :-180.0000  
##  1st Qu.:-535.0    1st Qu.:  231     1st Qu.: -45.00   1st Qu.:  -0.7375  
##  Median :-479.0    Median :  311     Median :  13.00   Median :  21.7000  
##  Mean   :-328.5    Mean   :  221     Mean   :  46.05   Mean   :  33.8265  
##  3rd Qu.:-304.0    3rd Qu.:  390     3rd Qu.:  95.00   3rd Qu.: 140.0000  
##  Max.   : 592.0    Max.   :  633     Max.   : 452.00   Max.   : 180.0000  
##                                                                           
##  pitch_forearm     yaw_forearm      total_accel_forearm gyros_forearm_x  
##  Min.   :-72.50   Min.   :-180.00   Min.   :  0.00      Min.   :-22.000  
##  1st Qu.:  0.00   1st Qu.: -68.60   1st Qu.: 29.00      1st Qu.: -0.220  
##  Median :  9.24   Median :   0.00   Median : 36.00      Median :  0.050  
##  Mean   : 10.71   Mean   :  19.21   Mean   : 34.72      Mean   :  0.158  
##  3rd Qu.: 28.40   3rd Qu.: 110.00   3rd Qu.: 41.00      3rd Qu.:  0.560  
##  Max.   : 89.80   Max.   : 180.00   Max.   :108.00      Max.   :  3.970  
##                                                                          
##  gyros_forearm_y     gyros_forearm_z    accel_forearm_x   accel_forearm_y 
##  Min.   : -7.02000   Min.   : -8.0900   Min.   :-498.00   Min.   :-632.0  
##  1st Qu.: -1.46000   1st Qu.: -0.1800   1st Qu.:-178.00   1st Qu.:  57.0  
##  Median :  0.03000   Median :  0.0800   Median : -57.00   Median : 201.0  
##  Mean   :  0.07517   Mean   :  0.1512   Mean   : -61.65   Mean   : 163.7  
##  3rd Qu.:  1.62000   3rd Qu.:  0.4900   3rd Qu.:  76.00   3rd Qu.: 312.0  
##  Max.   :311.00000   Max.   :231.0000   Max.   : 477.00   Max.   : 923.0  
##                                                                           
##  accel_forearm_z   magnet_forearm_x  magnet_forearm_y magnet_forearm_z
##  Min.   :-446.00   Min.   :-1280.0   Min.   :-896.0   Min.   :-973.0  
##  1st Qu.:-182.00   1st Qu.: -616.0   1st Qu.:   2.0   1st Qu.: 191.0  
##  Median : -39.00   Median : -378.0   Median : 591.0   Median : 511.0  
##  Mean   : -55.29   Mean   : -312.6   Mean   : 380.1   Mean   : 393.6  
##  3rd Qu.:  26.00   3rd Qu.:  -73.0   3rd Qu.: 737.0   3rd Qu.: 653.0  
##  Max.   : 291.00   Max.   :  672.0   Max.   :1480.0   Max.   :1090.0  
##                                                                       
##  classe  
##  A:5580  
##  B:3797  
##  C:3422  
##  D:3216  
##  E:3607  
##          
## 
```
Remove bookkeeping variables from our training and testing dataset.

```r
trainData <- trainData[, -c(1:7)]
trainData2 <- trainData[, -53]
testData_id <- testData[, 160]
testData <- testData[, -160]
testData <- testData[colnames(trainData2)]
testData <- cbind(testData, testData_id)
rm(trainData2)
rm(testData_id)
```
Check for near zero variables. We found none, so we can keep 53 variables.

```r
nzv <- nearZeroVar(trainData, saveMetrics=TRUE)
nzv
```

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt             1.101904     6.7781062   FALSE FALSE
## pitch_belt            1.036082     9.3772296   FALSE FALSE
## yaw_belt              1.058480     9.9734991   FALSE FALSE
## total_accel_belt      1.063160     0.1477933   FALSE FALSE
## gyros_belt_x          1.058651     0.7134849   FALSE FALSE
## gyros_belt_y          1.144000     0.3516461   FALSE FALSE
## gyros_belt_z          1.066214     0.8612782   FALSE FALSE
## accel_belt_x          1.055412     0.8357966   FALSE FALSE
## accel_belt_y          1.113725     0.7287738   FALSE FALSE
## accel_belt_z          1.078767     1.5237998   FALSE FALSE
## magnet_belt_x         1.090141     1.6664968   FALSE FALSE
## magnet_belt_y         1.099688     1.5187035   FALSE FALSE
## magnet_belt_z         1.006369     2.3290184   FALSE FALSE
## roll_arm             52.338462    13.5256345   FALSE FALSE
## pitch_arm            87.256410    15.7323412   FALSE FALSE
## yaw_arm              33.029126    14.6570176   FALSE FALSE
## total_accel_arm       1.024526     0.3363572   FALSE FALSE
## gyros_arm_x           1.015504     3.2769341   FALSE FALSE
## gyros_arm_y           1.454369     1.9162165   FALSE FALSE
## gyros_arm_z           1.110687     1.2638875   FALSE FALSE
## accel_arm_x           1.017341     3.9598410   FALSE FALSE
## accel_arm_y           1.140187     2.7367241   FALSE FALSE
## accel_arm_z           1.128000     4.0362858   FALSE FALSE
## magnet_arm_x          1.000000     6.8239731   FALSE FALSE
## magnet_arm_y          1.056818     4.4439914   FALSE FALSE
## magnet_arm_z          1.036364     6.4468454   FALSE FALSE
## roll_dumbbell         1.022388    84.2065029   FALSE FALSE
## pitch_dumbbell        2.277372    81.7449801   FALSE FALSE
## yaw_dumbbell          1.132231    83.4828254   FALSE FALSE
## total_accel_dumbbell  1.072634     0.2191418   FALSE FALSE
## gyros_dumbbell_x      1.003268     1.2282132   FALSE FALSE
## gyros_dumbbell_y      1.264957     1.4167771   FALSE FALSE
## gyros_dumbbell_z      1.060100     1.0498420   FALSE FALSE
## accel_dumbbell_x      1.018018     2.1659362   FALSE FALSE
## accel_dumbbell_y      1.053061     2.3748853   FALSE FALSE
## accel_dumbbell_z      1.133333     2.0894914   FALSE FALSE
## magnet_dumbbell_x     1.098266     5.7486495   FALSE FALSE
## magnet_dumbbell_y     1.197740     4.3012945   FALSE FALSE
## magnet_dumbbell_z     1.020833     3.4451126   FALSE FALSE
## roll_forearm         11.589286    11.0895933   FALSE FALSE
## pitch_forearm        65.983051    14.8557741   FALSE FALSE
## yaw_forearm          15.322835    10.1467740   FALSE FALSE
## total_accel_forearm   1.128928     0.3567424   FALSE FALSE
## gyros_forearm_x       1.059273     1.5187035   FALSE FALSE
## gyros_forearm_y       1.036554     3.7763735   FALSE FALSE
## gyros_forearm_z       1.122917     1.5645704   FALSE FALSE
## accel_forearm_x       1.126437     4.0464784   FALSE FALSE
## accel_forearm_y       1.059406     5.1116094   FALSE FALSE
## accel_forearm_z       1.006250     2.9558659   FALSE FALSE
## magnet_forearm_x      1.012346     7.7667924   FALSE FALSE
## magnet_forearm_y      1.246914     9.5403119   FALSE FALSE
## magnet_forearm_z      1.000000     8.5771073   FALSE FALSE
## classe                1.469581     0.0254816   FALSE FALSE
```
## Model Building
Splitting data using partition of 70% training and 30% testing.

```r
set.seed(1122)
inTrain <- createDataPartition(trainData$classe, p=.7, list=FALSE)
training <- trainData[inTrain, ]
testing <- trainData[-inTrain, ]
```
Choosing random forest for its high accuracy of prediction. After testing different ntree values, 40 trees was the reasonable number which resulted in high accuracy.

```r
if(!file.exists("modelFit.rds")) {
        modelFit <- train(classe~., data = training, method = "rf", ntree=40)
        saveRDS(modelFit, "modelFit.rds")
}
modelFit <- readRDS("modelFit.rds")        
predict <- predict(modelFit, testing[, -53])
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
confusionMatrix(predict, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673   11    0    0    0
##          B    0 1125    5    1    1
##          C    1    3 1016    3    2
##          D    0    0    5  957    2
##          E    0    0    0    3 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9937          
##                  95% CI : (0.9913, 0.9956)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.992           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9877   0.9903   0.9927   0.9954
## Specificity            0.9974   0.9985   0.9981   0.9986   0.9994
## Pos Pred Value         0.9935   0.9938   0.9912   0.9927   0.9972
## Neg Pred Value         0.9998   0.9971   0.9979   0.9986   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1912   0.1726   0.1626   0.1830
## Detection Prevalence   0.2862   0.1924   0.1742   0.1638   0.1835
## Balanced Accuracy      0.9984   0.9931   0.9942   0.9957   0.9974
```
Model using Bootstrap with 10 resamples

```r
if(!file.exists("modelFitBS.rds")) {
        tcBS <- trainControl(method="boot", number=10)
        modelFitBS <- train(classe~., data = training, trControl = tcBS, method = "rf", ntree=40)
        saveRDS(modelFitBS, "modelFitBS.rds")
}
modelFitBS <- readRDS("modelFitBS.rds")
predictBS <- predict(modelFitBS, testing[, -53])
confusionMatrix(predictBS, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1667   17    0    0    0
##          B    5 1118    5    0    1
##          C    1    4 1012    5    2
##          D    0    0    9  955    1
##          E    1    0    0    4 1078
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9907         
##                  95% CI : (0.9879, 0.993)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9882         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9816   0.9864   0.9907   0.9963
## Specificity            0.9960   0.9977   0.9975   0.9980   0.9990
## Pos Pred Value         0.9899   0.9903   0.9883   0.9896   0.9954
## Neg Pred Value         0.9983   0.9956   0.9971   0.9982   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2833   0.1900   0.1720   0.1623   0.1832
## Detection Prevalence   0.2862   0.1918   0.1740   0.1640   0.1840
## Balanced Accuracy      0.9959   0.9896   0.9919   0.9943   0.9976
```
Model using k-fold cross validation

```r
if(!file.exists("modelFitKF.rds")) {
        tcKF <- trainControl(method = "cv", number=10)
        modelFitKF <- train(classe~., data = training, trControl = tcKF, method = "rf", ntree=40)
        saveRDS(modelFitKF, "modelFitKF.rds")
}
modelFitKF <- readRDS("modelFitKF.rds")        
predictKF <- predict(modelFitKF, testing[, -53])
confusionMatrix(predictKF, testing$classe)    
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1668   12    0    0    0
##          B    4 1125    4    0    1
##          C    1    2 1017    6    4
##          D    0    0    5  954    2
##          E    1    0    0    4 1075
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9922          
##                  95% CI : (0.9896, 0.9943)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9901          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9877   0.9912   0.9896   0.9935
## Specificity            0.9972   0.9981   0.9973   0.9986   0.9990
## Pos Pred Value         0.9929   0.9921   0.9874   0.9927   0.9954
## Neg Pred Value         0.9986   0.9971   0.9981   0.9980   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2834   0.1912   0.1728   0.1621   0.1827
## Detection Prevalence   0.2855   0.1927   0.1750   0.1633   0.1835
## Balanced Accuracy      0.9968   0.9929   0.9943   0.9941   0.9962
```
Model using repeated k-fold cross validation

```r
if(!file.exists("modelFitRKF.rds")) {
        tcRKF <- trainControl(method = "repeatedcv", number=10, repeats=3)
        modelFitRKF <- train(classe~., data = training, trControl = tcRKF, method = "rf", ntree=40)
        saveRDS(modelFitRKF, "modelFitRKF.rds")
}
modelFitRKF <- readRDS("modelFitRKF.rds")
predictRKF <- predict(modelFitRKF, testing[, -53])
confusionMatrix(predictRKF, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671   13    0    0    0
##          B    2 1124    5    0    1
##          C    0    2 1012    5    2
##          D    0    0    9  954    1
##          E    1    0    0    5 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9922          
##                  95% CI : (0.9896, 0.9943)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9901          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9868   0.9864   0.9896   0.9963
## Specificity            0.9969   0.9983   0.9981   0.9980   0.9988
## Pos Pred Value         0.9923   0.9929   0.9912   0.9896   0.9945
## Neg Pred Value         0.9993   0.9968   0.9971   0.9980   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1910   0.1720   0.1621   0.1832
## Detection Prevalence   0.2862   0.1924   0.1735   0.1638   0.1842
## Balanced Accuracy      0.9976   0.9926   0.9923   0.9938   0.9975
```
Using random forest algorithm with k-fold cross validation and repeated k-fold cross validation yielded the best accuracy.  Accuracy of my model is 99.22% with out of sample error 0.78%

```r
acc_model <- confusionMatrix(predictRKF, testing$classe)$overall[1]
acc_model
```

```
##  Accuracy 
## 0.9921835
```

```r
error_oos <- 1 - as.numeric(confusionMatrix(predictRKF, testing$classe)$overall[1])
error_oos
```

```
## [1] 0.007816483
```
## Prediction
Using my prediction model to predict 20 test cases from testData.

```r
predictTest <- predict(modelFitRKF, testData[, -53])
predictTest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
