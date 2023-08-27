install.packages(c("kohonen","ggplot2", "maptools", "sp", "reshape2", "rgeos"))
install.packages("wesanderson")
tinytex::install_tinytex()
install.packages("tinytex") #exporting pdf
install.packages("Hmisc")
install.packages("RSNNS")
install.packages("Rcpp")
install.packages("caret")


library(kohonen)
#library(dummies) package ‘dummies’ is not available for this version of R
library(ggplot2)
library(sp)
library(maptools)
library(reshape2)
library(rgeos)
library(wesanderson)
library(Matrix)
library(tinytex)
library(Hmisc)
#for classification
library(Rcpp)
library(caret)


# Colour palette definition
wes_palette("Moonrise3")

#read in data file
setwd("~/Documents/info411/assignments/a1/a1_support_files/")
creditWorthiness <- read.csv("creditworthiness.csv")
class(creditWorthiness)
describe(creditWorthiness)

classifiedRows = subset(creditWorthiness, creditWorthiness[,46] > 0)
unClassifiedRows = subset(creditWorthiness, creditWorthiness[,46] == 0)
#View(classifiedRows)
#View(unClassifiedRows)

corTable = cor(classifiedRows, y = classifiedRows$credit.rating)
corTable = corTable[order(corTable, decreasing = TRUE), ,drop = FALSE]
corTable
print(corTable, 6) #(,decimal places) 6d.p


#------------------------interested features to train the SOM model------------------------------------------
interestedFeatures1 <- creditWorthiness[, c(1,2,3,4,30,29,43,45,28,35,14,42,26)] #i will take (0.01 < x < 0.32) (negative corr)
#26 min..account.balance.7.months.ago                           -0.010927758
#42 avrg..account.balance.2.months.ago                          -0.012365831
#15 avrg..account.balance.11.months.ago                         -0.014216949
#35 min..account.balance.4.months.ago                           -0.014415123
#28 max..account.balance.6.months.ago                           -0.014459836
#45 avrg..account.balance.1.months.ago                          -0.014726193
#43 max..account.balance.1.months.ago                           -0.021203205
#29 min..account.balance.6.months.ago                           -0.031578162
#30 avrg..account.balance.6.months.ago                          -0.045358455
#4 gender                                                       -0.072232289
#2 re.balanced..paid.back..a.recently.overdrawn.current.acount  -0.218223135
#3 FI3O.credit.score                                            -0.279887011
#1 functionary                                                  -0.317282792
#View(interestedFeatures1)

interestedFeatures2 <- creditWorthiness[, c(6,13,37,22,21,32,11,10,12,23,5,19,41,40,38,33,24,9,17)] #i will take (0.01 < x < 0.22) (positive corr)
#6 credit.refused.in.past.                                       0.217838467
#13 max..account.balance.11.months.ago                           0.049489339
#37 max..account.balance.3.months.ago                            0.038158186
#22 max..account.balance.8.months.ago                            0.035423671
#21 avrg..account.balance.9.months.ago                           0.029746003
#32 min..account.balance.5.months.ago                            0.029634472
#11 min..account.balance.12.months.ago                           0.028736469
#10 max..account.balance.12.months.ago                           0.027334233
#12 avrg..account.balance.12.months.ago                          0.026958574
#23 min..account.balance.8.months.ago                            0.025463879
#5 X0..accounts.at.other.banks                                   0.023139554
#19 max..account.balance.9.months.ago                            0.022032572
#41 min..account.balance.2.months.ago                            0.021604156
#40 max..account.balance.2.months.ago                            0.018010779
#38 min..account.balance.3.months.ago                            0.016808148
#33 avrg..account.balance.5.months.ago                           0.015258396
#24 avrg..account.balance.8.months.ago                           0.013587599
#9 self.employed.                                                0.012380228
#17 min..account.balance.10.months.ago                           0.010292940

#View(interestedFeatures2)
#which( colnames(creditWorthiness)=="self.employed." ) //Get the column number in R given the column name

# --------------------------- SOM TRAINING (negative corr)----------------------------------
require(kohonen)
data_train <- interestedFeatures1 #13 features
# Change the data frame with training data to a matrix
# Also center and scale all variables to give them equal importance during
# the SOM training process. 
data_train_matrix <- as.matrix(scale(data_train)) 

# Create the SOM Grid - 
#Hexagonal and Circular topologies are possible
som_grid <- somgrid(xdim = 20, ydim=20, topo="hexagonal") #400 nodes 

# train the SOM
if (packageVersion("kohonen") < 3){
  system.time(som_model <- som(data_train_matrix, 
                               grid=som_grid, 
                               rlen=1000,  #no of iterations
                               alpha=c(0.9,0.01),#ANN learning rate
                               n.hood = "circular",#neighbourhood shape
                               keep.data = TRUE )) #keep data in model
}else{
  system.time(som_model <- som(data_train_matrix, 
                               grid=som_grid, 
                               rlen=1000, 
                               alpha=c(0.9,0.01),
                               mode="online",
                               normalizeDataLayers=false,
                               keep.data = TRUE ))
}

summary(som_model)


# ------------------------ SOM VISUALISATION -----------------------
#negative corr
source('./coolBlueHotRed.R')
plot(som_model, type="changes")
plot(som_model, type="count",main="Node Counts")
plot(som_model, type="dist.neighbours")
plot(som_model, type="codes")
source('./plotHeatMap.R')

for (var in 1:13) 
{
  var_unscaled <- aggregate(as.numeric(data_train[,var]), by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,2] 
  plot(som_model, type = "property", property=var_unscaled, main=names(data_train)[var], palette.name=coolBlueHotRed)
  rm(var_unscaled)
}

genderT = with(classifiedRows, table(credit.rating, gender))
barplot(genderT, beside = TRUE,
        legend = c("Credit Rating A", "Credit Rating B", "Credit Rating C"),
        col = c("darkgreen","yellow", "red"),
        main = "Gender vs Credit Rating",
        sub="0 = Male, 1 = Female")

functional = with(classifiedRows, table(credit.rating, functionary))
barplot(functional, beside = TRUE,
        legend = c("Credit Rating A", "Credit Rating B", "Credit Rating C"),
        col = c("darkgreen","yellow", "red"),
        main = "Functionary vs Credit Rating",
        sub="0 = No, 1 = Yes")

paidBack = with(classifiedRows, table(credit.rating, re.balanced..paid.back..a.recently.overdrawn.current.acount))
barplot(paidBack, beside = TRUE,
        legend = c("Credit Rating A", "Credit Rating B", "Credit Rating C"),
        col = c("darkgreen","yellow", "red"),
        main = "paid back vs Credit Rating",
        sub="0 = No, 1 = Yes")

fico = with(classifiedRows, table(credit.rating, FI3O.credit.score))
barplot(fico, beside = TRUE,
        legend = c("Credit Rating A", "Credit Rating B", "Credit Rating C"),
        col = c("darkgreen","yellow", "red"),
        main = "fico vs Credit Rating",
        sub="0 = No, 1 = Yes")

avr6months = with(classifiedRows, table(credit.rating, avrg..account.balance.6.months.ago))
barplot(avr6months, beside = TRUE,
        legend = c("Credit Rating A", "Credit Rating B", "Credit Rating C"),
        col = c("darkgreen","yellow", "red"),
        main = "avr6monthsago vs Credit Rating",
        sub="1 = <100, 2 =<1000, 3= <7500 4=<10000, 5=<20000")

min6months = with(classifiedRows, table(credit.rating, min..account.balance.6.months.ago))
barplot(min6months, beside = TRUE,
        legend = c("Credit Rating A", "Credit Rating B", "Credit Rating C"),
        col = c("darkgreen","yellow", "red"),
        main = "min6monthsago vs Credit Rating",
        sub="1 = <100, 2 =<1000, 3= <7500 4=<10000, 5=<20000")



# ------------------ Clustering SOM results -------------------
mydata <- matrix(unlist(som_model$codes), ncol = length(data_train), byrow = FALSE)

wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(mydata,centers=i)$withinss)

par(mar=c(5.1,4.1,4.1,2.1))
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares", main="Within cluster sum of
squares (WCSS)")

# Form clusters on grid
## use hierarchical clustering to cluster the codebook vectors
som_cluster <- cutree(hclust(dist(mydata)), 3)
# Show the map with different colours for every cluster

plot(som_model, type="mapping", bgcol = wes_palette("Moonrise3")[som_cluster], main = "Clusters")
add.cluster.boundaries(som_model, som_cluster)

#show the same plot with the codes instead of just colours
plot(som_model, type="codes", bgcol = wes_palette("Moonrise3")[som_cluster], main = "Clusters")
add.cluster.boundaries(som_model, som_cluster)


#disadvatages of som i noticed
#1.Lack of parallelisation capabilities for very large data sets.
#2.Difficult to represent very many variables in two dimensional plane.
#3.Requires clean, numeric data.


# ------------------------ Classification -------------------------------
# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(y = classifiedRows$credit.rating, p = 0.8, list = FALSE)

trainData <- classifiedRows[trainIndex, ]
testData <- classifiedRows[-trainIndex, ]

trainData <- classifiedRows[trainIndex, c("functionary", 
                                          "FI3O.credit.score",
                                          "re.balanced..paid.back..a.recently.overdrawn.current.acount",
                                          "gender",
                                          "avrg..account.balance.6.months.ago",
                                          "min..account.balance.6.months.ago",
                                          "max..account.balance.1.months.ago",
                                          "avrg..account.balance.1.months.ago",
                                          "max..account.balance.6.months.ago",
                                          "min..account.balance.4.months.ago",
                                          "avrg..account.balance.11.months.ago",
                                          "avrg..account.balance.2.months.ago",
                                          "min..account.balance.7.months.ago",
                                          "credit.rating")]
head(testData)
head(trainData)
testData <- classifiedRows[-trainIndex,  c("functionary", 
                                           "FI3O.credit.score",
                                           "re.balanced..paid.back..a.recently.overdrawn.current.acount",
                                           "gender",
                                           "avrg..account.balance.6.months.ago",
                                           "min..account.balance.6.months.ago",
                                           "max..account.balance.1.months.ago",
                                           "avrg..account.balance.1.months.ago",
                                           "max..account.balance.6.months.ago",
                                           "min..account.balance.4.months.ago",
                                           "avrg..account.balance.11.months.ago",
                                           "avrg..account.balance.2.months.ago",
                                           "min..account.balance.7.months.ago",
                                           "credit.rating")]


train_labels <- trainData[, 14]
train_labels <- as.factor(train_labels)
levels(train_labels) <- c('1','2','3')
trainData <- trainData[, c(1:13)]

test_labels <- testData[,14]
test_labels <- as.factor(test_labels)
levels(test_labels) <- c('1','2','3')
testData <- testData[, c(1:13)]

model_mlp <- train(x = trainData,
                   y = train_labels, 
                   method = "mlp")

predictions <- predict(model_mlp, newdata = testData)
confusionMatrix(predictions, test_labels)


#fine tuning
mlp_grid = expand.grid(layer1 = 10,
                       layer2 = 10,
                       layer3 = 10)

mlp_fit = caret::train(x = trainData, 
                       y = train_labels, 
                       method = "mlpML", 
                       preProc =  c('center', 'scale', 'knnImpute', 'pca'),
                       trControl = trainControl(method = "cv", verboseIter = TRUE, returnData = FALSE),
                       tuneGrid = mlp_grid)


mlp_fit








