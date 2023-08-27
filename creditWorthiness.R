#import libraries
library(ggplot2)
library(lattice)
library(caret)
library(tree)
library(party)
library(rpart)
library(pROC)
library(tidyverse)
library(randomForest)
library(e1071)
library(ROCR)


cw.all  <- read.csv("creditworthiness.csv")


#data preprocessing
classifiedRows = subset(cw.all, credit.rating > 0)
unClassifiedRows <- subset(cw.all,credit.rating == 0)

train <- classifiedRows[1:(nrow(classifiedRows)/2),] # first half
test <- classifiedRows[-(1:(nrow(classifiedRows)/2)),] # second half


#train decision tree
require(tree)

decisionTree= tree(as.factor(credit.rating)~., train, method = "class")
print(decisionTree)
plot(decisionTree)
text(decisionTree, 
     pretty = 0)

summary(decisionTree)
class(decisionTree)

########################. answer .##################################
# The tree has 6 terminal nodes which means that there are 6 
# classifications of the outcomes of the training set. 
# This means that the tree has identified 6 distinct categories 
# for credit.rating based on the combination of values for these 
# predictor variables in the training data.
#
# the average distance between the predicted class probabilities 
# and the actual class probabilities is 1.794 which is the 
# residual mean deviance.
#
# The rate of misclassification is 38.84% (0.3884) of the instances 
# in the training set were misclassified by the model.
####################################################################


#create a data frame for the hypothetical customer
medianCust = data.frame()
newData = c(0,1,1,0,3,0,3,3,0,3,3,3,3,
            3,3,3,3,3,3,3,3,3,3,3,3,3,
            3,3,3,3,3,3,3,3,3,3,3,3,3,
            3,3,3,3,3,3) #45 column values

medianCust = rbind(medianCust, newData)
colnames(medianCust) = names(cw.all)[-46]

cust.pred = predict(decisionTree, medianCust, type="class")
cust.pred


############. output.##########
# >[1] 2
# >Levels: 1 2 3

#credit rating of a hypothetical “median” 
#customer is most likely to be B.
##########################################

testPredict_dt =  predict(decisionTree, test, type="class")

confusion_dt = with(test, table(testPredict_dt, credit.rating))
confusion_dt

accuracy_dt = sum(diag(confusion_dt))/sum(confusion_dt)
accuracy_dt

###############. output .##############
#
#            credit.rating
# testPredict   1   2   3
#            1 162  85  37
#            2  90 361 143
#            3   5  21  77
#
# accuracy is 0.6116208.
#######################################


#compute Shannon entropy
# entropy <- function(target) 
# {
#   freq <- table(target)/length(target)
#   # vectorize
#   vec <- as.data.frame(freq)[,2]
#   #drop 0 to avoid NaN resulting from log2
#   vec<-vec[vec>0]
#   #compute entropy
#   -sum(vec * log2(vec))
# }
# 
# entropy(train$credit.rating)

#Compute the entropy before the split
# get the count of all classes in credit.rating using the table() function 
beforeCountFreq = table(train$credit.rating)
#find the probability of each class
beforeClassProb = beforeCountFreq/sum(beforeCountFreq)
#calculate entropy (before split)
beforeEntropy = -sum(beforeClassProb * log2(beforeClassProb))
beforeEntropy

# Compute the entropy for ‘functionary’ feature value 0
# functionary == 0
countFreq0 = table(train$credit.rating[train$functionary == 0]) 
classProb0 = countFreq0/sum(countFreq0)
(functionaryEnt0 = -sum(classProb0 * log2(classProb0)))

# functionary == 1
countFreq1 = table(train$credit.rating[train$functionary == 1]) 
classProb1 = countFreq1/sum(countFreq1)
(functionaryEnt1 = -sum(classProb1 * log2(classProb1)))

IG = (beforeEntropy - (functionaryEnt0 * sum(countFreq0) + 
                       functionaryEnt1 * sum(countFreq1)) /
                    sum(sum(countFreq0) + sum(countFreq1)))

IG

#############. output .###############
#
# gain in entropy is 0.0883414.
#
######################################

require(randomForest)
randomForestTest = randomForest(as.factor(credit.rating)~., data=train)
randomForestTest

testPredict_rf =  predict(randomForestTest, test, type="class")
summary(testPredict_rf)

#############. output .#################
# OOB estimate of  error rate: 42.3%
# Confusion matrix:
#      1   2  3 class.error
#   1 59 166  1   0.7389381
#   2 30 445 28   0.1153082
#   3 14 176 62   0.7539683
#
# accuracy is 57.7%
########################################



confusion_rf = with(test, table(testPredict_rf, credit.rating))
confusion_rf

accuracy_rf = sum(diag(confusion_rf))/sum(confusion_rf)
accuracy_rf


# improve the random forest model
# Fit to a model using randomForest after the tuning
tune_rf = randomForest(as.factor(credit.rating)~., data = train, mtry
                                = 12, ntree=500, stepFactor=2, improve=0.2)

tune_rf.pred = predict(tune_rf, test[,-46])

# Produce confusion matrix after the tuning
confusion_rf_tuned = with(test, table(tune_rf.pred, credit.rating))
confusion_rf_tuned

# Calculate the accuracy rate after the tuning
sum(diag(confusion_rf_tuned))/sum(confusion_rf_tuned)


#############. output .#################
# before tuning:
#              credit.rating
# testPredict_rf   1   2   3
#              1  52  40  18
#              2 203 416 190
#              3   2  11  49
# accuracy before tuning is 0.5270133
# after tuning:
#            credit.rating
# tune_rf.pred   1   2   3
#            1 105  59  24
#            2 148 391 168
#            3   4  17  65
#
#accuracy after tuning is 0.5718654
###################################################


svmfit = svm(as.factor(credit.rating) ~ ., data = train, kernel = "radial")
print(svmfit)
decision_value_svm_for_medianCust = predict(svmfit, medianCust, decision.values = TRUE)
decision_value_svm_for_medianCust

#############. output .#################
# >1 
# >2 
# >attr(,"decision.values")
# >2/1      2/3         1/3
# >1 1.021296 1.511396 -0.04938262
# >Levels: 1 2 3
#
#
# The predicted credit of the customer is 1 and 2.
###################################################

svm.pred = predict(svmfit, test[,-46])

confusionSVM = with(test, table(svm.pred, credit.rating))
confusionSVM

sum(diag(confusionSVM))/sum(confusionSVM)

#############. output .#################
# before tuning:
#         credit.rating
# svm.pred   1   2   3
#        1 109  56  22
#        2 143 393 162
#        3   5  18  73
#
# The accuracy is 0.5861366.
###################################################


#improve the SVM model via fine tuning
summary(tune.svm(as.factor(credit.rating) ~ ., data = train,
                  kernel = "radial",cost = 10^c(0:2), gamma = 10^c(-4:-1)))

# Fit a model using SVM
svmTuned = svm(as.factor(credit.rating) ~ ., data = train, kernel = "radial", cost=100,
                gamma = 0.0001)

# Predict the values on test set
svmTuned.pred = predict(svmTuned, test[,-46])

confusion_svm_tuned = with(test, table(svmTuned.pred, credit.rating))
confusion_svm_tuned

# Calculate the accuracy rate after the tuning
sum(diag(confusion_svm_tuned))/sum(confusion_svm_tuned)

#############. output .#################
# before tuning:
#         credit.rating
# svm.pred   1   2   3
#        1 109  56  22
#        2 143 393 162
#        3   5  18  73
#
# The accuracy before fine tuning is 0.5861366.
#
# after tuning:
#              credit.rating
# svmTuned.pred   1   2   3
#             1 159  87  39
#             2  93 361 146
#             3   5  19  72
#
# The accuracy after fine tuning is 0.6034659.
######################################################


nb = naiveBayes(as.factor(credit.rating)~. ,data=train) 
predict(nb, medianCust, type='class') 
predict(nb, medianCust, type='raw')


#############. output .#######################
# >[1] 2
# >Levels: 1 2 3
# The customer is likely to be in class B.              
# 
#--------------------Report----------------------
#. > predict(nb, medianCust, type='raw')
#                1         2         3
#   [1,] 0.9850729 0.01393277 0.0009942948
#
# The probability of the customer being in 
# class A = 0.9850729
# class B = 0.01393277
# class C = 0.0009942948. 
#
# The customer is more likely in credit rating A.
#######################################################


#predict the values on test set
testPredict_nb = predict(nb, test[,-46])
head(testPredict_nb, 20)

#################. output .###########################
# >[1] 1 1 3 3 1 1 1 1 1 1 1 1 1 3 1 1 1 3 1 1
# Levels: 1 2 3
#
# ####################################################


#producing confusion matrix for nb
confusion_nb = with(test, table(testPredict_nb, credit.rating))
confusion_nb

accuracy = sum(diag(confusion_nb))/sum(confusion_nb)
accuracy

#############. output .#######################
#
#              credit.rating
# testPredict_nb   1   2   3
#              1 252 439 173
#              2   0   4   6
#              3   5  24  78
#
# Accuracy for nb = 0.3404689
#
#####################################################


# below is a summary of all the 
# accuracy scores using confusion matrix for all classification models:
# 1. naive bayes before fine tuning = 0.3404689
# 2. random forest before fine tuning = 0.5270133
# 3. random forest after fine tuning = 0.5718654
# 4. support vector machines before fine tuning = 0.5861366
# 5. support vector machines after fine tuning = 0.6034659
# 6. decision trees before fine tuning = 0.6116208

# At this point the best classifier is the decision tree which is at
# 61.2% accuracy rqte. 
# Decision Trees work by recursively partitioning the data based 
# on the values of the features, so it's possible that the 
# features used for training the model are well-suited for this 
# type of algorithm. 

# i feel that naive bayes's accuracy is really bad. It is so low that I predict
# that the model is not performing well on the dataset. This could be that there
# is an insufficient number of training samples.


#logistic regression 
glm <- glm(as.factor((credit.rating==1))~., data = train, family = binomial)
glm

options(width = 130)
summary(glm)


###################. report .#######################
#
# Call:
#   glm(formula = as.factor((credit.rating == 1)) ~ ., family = binomial, 
#       data = train)
# 
# Deviance Residuals: 
#   Min        1Q    Median        3Q       Max  
# -2.00215  -0.65353  -0.42668  -0.00012   2.70789  
# 
# Coefficients:
#                                                           Estimate Std.     Error  z-value Pr(>|z|)    
#                                                              ______________________________________
# (Intercept)                                                 -17.551605 429.995589  -0.041  0.96744    
# functionary                                                   1.740533   0.183036   9.509  < 2e-16 ***
# re.balanced..paid.back..a.recently.overdrawn.current.acount   1.501222   0.550965   2.725  0.00644 ** 
# FI3O.credit.score                                            16.502759 429.993845   0.038  0.96939    
# gender                                                        0.577104   0.178807   3.228  0.00125 ** 
# X0..accounts.at.other.banks                                  -0.027413   0.063141  -0.434  0.66417    
# credit.refused.in.past.                                      -0.935877   0.341848  -2.738  0.00619 ** 
# years.employed                                                0.672572   0.269126   2.499  0.01245 *  
# savings.on.other.accounts                                    -0.548195   0.204670  -2.678  0.00740 ** 
# self.employed.                                               -0.376394   0.236506  -1.591  0.11150    
# max..account.balance.12.months.ago                           -0.004444   0.062647  -0.071  0.94345    
# min..account.balance.12.months.ago                            0.030192   0.063737   0.474  0.63572    
# avrg..account.balance.12.months.ago                           0.124651   0.065028   1.917  0.05525 .  
# max..account.balance.11.months.ago                           -0.010150   0.063924  -0.159  0.87385    
# min..account.balance.11.months.ago                           -0.110469   0.064328  -1.717  0.08593 .  
# avrg..account.balance.11.months.ago                           0.052783   0.065196   0.810  0.41816    
# max..account.balance.10.months.ago                            0.019305   0.062526   0.309  0.75750    
# min..account.balance.10.months.ago                           -0.101696   0.063199  -1.609  0.10759    
# avrg..account.balance.10.months.ago                          -0.050933   0.065720  -0.775  0.43834    
# max..account.balance.9.months.ago                             0.096730   0.062586   1.546  0.12221    
# min..account.balance.9.months.ago                            -0.038009   0.064765  -0.587  0.55728    
# avrg..account.balance.9.months.ago                           -0.032928   0.062640  -0.526  0.59912    
# max..account.balance.8.months.ago                            -0.019017   0.063459  -0.300  0.76443    
# min..account.balance.8.months.ago                            -0.041455   0.062710  -0.661  0.50858    
# avrg..account.balance.8.months.ago                           -0.106852   0.063685  -1.678  0.09338 .  
# max..account.balance.7.months.ago                            -0.018414   0.063321  -0.291  0.77120    
# min..account.balance.7.months.ago                            -0.094176   0.063702  -1.478  0.13930    
# avrg..account.balance.7.months.ago                           -0.074021   0.061950  -1.195  0.23215    
# max..account.balance.6.months.ago                             0.069171   0.064686   1.069  0.28492    
# min..account.balance.6.months.ago                            -0.033830   0.062428  -0.542  0.58788    
# avrg..account.balance.6.months.ago                           -0.025278   0.062786  -0.403  0.68724    
# max..account.balance.5.months.ago                             0.015218   0.061902   0.246  0.80581    
# min..account.balance.5.months.ago                            -0.088221   0.064391  -1.370  0.17066    
# avrg..account.balance.5.months.ago                           -0.072089   0.063401  -1.137  0.25553    
# max..account.balance.4.months.ago                             0.034718   0.062889   0.552  0.58091    
# min..account.balance.4.months.ago                            -0.036728   0.064179  -0.572  0.56714    
# avrg..account.balance.4.months.ago                            0.020068   0.063954   0.314  0.75368    
# max..account.balance.3.months.ago                            -0.144584   0.062966  -2.296  0.02166 *  
# min..account.balance.3.months.ago                             0.014149   0.064191   0.220  0.82554    
# avrg..account.balance.3.months.ago                           -0.010770   0.064635  -0.167  0.86767    
# max..account.balance.2.months.ago                             0.100711   0.063196   1.594  0.11102    
# min..account.balance.2.months.ago                            -0.065585   0.063059  -1.040  0.29832    
# avrg..account.balance.2.months.ago                           -0.038225   0.064392  -0.594  0.55276    
# max..account.balance.1.months.ago                            -0.073012   0.065482  -1.115  0.26486    
# min..account.balance.1.months.ago                            -0.000658   0.062229  -0.011  0.99156    
# avrg..account.balance.1.months.ago                           -0.068570   0.064302  -1.066  0.28626    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1058.95  on 980  degrees of freedom
# Residual deviance:  820.79  on 935  degrees of freedom
# AIC: 912.79
# 
# Number of Fisher Scoring iterations: 16

####################. report .##################
# The predictors with smaller values of standard deviance 
# residuals indicate better fit of the model to the data.

# p-value less than 0.05 indicates that the independent 
# variable is statistically significant at the 5% level, 
# and a p-value greater than 0.05 indicates that the 
# independent variable is not statistically significant.
#

#coefficients represent the change in the log odds 
#of the dependent varible for a one unit change in the 
#independent variable. A positive coefficient means 
#that the log odds of the dependent variable increase with 
#an increase in the independent variable, while a negative 
#coefficient indicates that the log odds decrease with an 
#increase in the independent variable.

# Hence the predictors that seem the most significant are;
# functionary,
# re.balanced..paid.back..a.recently.overdrawn.current.acount, 
# gender, 
# credit.refused.in.past., 
# years.employed, 
# savings.on.other.accounts, 
# max..account.balance.3.months.ago


#i feel that gender might be false.


# Fit an SVM model of your choice to the training set 
require(e1071)

summary(tune.svm(as.factor((credit.rating==1)) ~ ., data = train,
                 kernel = "radial",cost = 10^c(-2:2), gamma = 10^c(-4:1),
                 type="C"))

(svm2 = svm(I(credit.rating == 1)~ ., data = train, type = "C"))


# SVM predict
svm.fit.pred = predict(svm2, test[,-46], decision.values =TRUE)
svm.fit.pred

# GLM predict
glm.fit.pred = predict(glm, test[,-46])
glm.fit.pred



require(ROCR)
#confusion matrix SVM
confusion_svm = prediction(-attr(svm.fit.pred, 
                           "decision.values"),
                           test$credit.rating == 1)

# SVM ROCS
rocs_svm <- performance(confusion_svm, "tpr", "fpr")

#confusion matrix GLM
confusion_glm = prediction(glm.fit.pred, test$credit.rating == 1)

# GLM ROCS
rocs_glm <- performance(confusion_glm, "tpr", "fpr")

# Plot ROC chart
plot(rocs_glm, col=1)
plot(rocs_svm, col= 2 ,add=TRUE)
abline(0, 1, lty = 3)

# Add the legend to the graph
legend(0.6, 0.6, c('GLM','SVM'), 1:2)


################. report .##############

# The GLM seems to have a slighly better 
# accuracy compared to the SVM.









