library(tidyverse)
library (rpart)
library(rpart.plot)
library(e1071)
setwd("/Users/elansangan/Documents/2Rdir/at2_36106/data/")
creditdata <- read_csv("AT2_credit_train_fixed.csv")
creditdata <- read_csv("credit_train_with_counts2.csv")

#cleaning
creditdata <- creditdata %>% mutate(PAY_AMT6 = ifelse(PAY_AMT6 > 0, "Y", "N"))
creditdata <- creditdata %>% mutate(PAY_AMT5 = ifelse(PAY_AMT5 > 0, "Y", "N"))
creditdata <- creditdata %>% mutate(PAY_AMT4 = ifelse(PAY_AMT4 > 0, "Y", "N"))
creditdata <- creditdata %>% mutate(PAY_AMT3 = ifelse(PAY_AMT3 > 0, "Y", "N"))
creditdata <- creditdata %>% mutate(PAY_AMT2 = ifelse(PAY_AMT2 > 0, "Y", "N"))
creditdata <- creditdata %>% mutate(PAY_AMT1 = ifelse(PAY_AMT1 > 0, "Y", "N"))
#if graduate school or uni, 1, else 0
creditdata <- creditdata %>% mutate(EDUCATION = ifelse(EDUCATION > 2, "Low", "High"))
#if limit is -99, set to 50000 (mode)
creditdata <- creditdata %>% mutate(LIMIT_BAL = ifelse(LIMIT_BAL < 0, 50000, LIMIT_BAL))
#fix SEX
creditdata <- creditdata %>% mutate(SEX = ifelse(SEX == 2113, 3, SEX))
creditdata <- creditdata %>% mutate(SEX = ifelse(is.na(SEX), 3, SEX))
#creditdata <- creditdata %>% mutate(SEX = ifelse(SEX == 3, 2, SEX))

## 80% of the sample size, use floor to round down to nearest integer
trainset_size <- floor(0.80 * nrow(creditdata))
trainset_indices <- sample(seq_len(nrow(creditdata)), size = trainset_size)
#assign observations to training and testing sets
trainset <- creditdata[trainset_indices, ]
testset <- creditdata[-trainset_indices, ]

###LOG MODEL
#all
logmodel1 <- glm(formula = default ~ COUNTS + LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data = trainset, family = "binomial")
#less LIMIT and SEX
logmodel1 <- glm(formula = default ~ EDUCATION + MARRIAGE + AGE + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data = trainset, family = "binomial")
# PAY_AMT removed + COUNTS
logmodel1 <- glm(formula = default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + COUNTS + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6, data = trainset, family = "binomial")
# add the probabilities to the testing data
testset$probability = predict(logmodel1, newdata = testset, type = "response")

# assume that the optimum probability threshold is 0.5
# Create the class prediction
testset$prediction = 0
testset[testset$probability >= 0.5, "prediction"] = 1

# Create a confusion matrix
cfm <- table(predicted=testset$prediction,true=testset$default)
cfm
### END

###TREE MODEL
# Get index of predicted variable
class_col_num <- grep("default",names(trainset))

#rpart
#all (BEST A28)
rpart_model <-  rpart(default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data = trainset, method = "class")
#less limit and sex
rpart_model <-  rpart(default ~ EDUCATION + MARRIAGE + AGE + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data = trainset, method = "class")
#less pays with COUNTS
rpart_model <-  rpart(default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + COUNTS + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data = trainset, method = "class")
#

# Predict on test data
rpart_predict <- predict(rpart_model,testset[,-class_col_num],type="class")
# Create a confusion matrix
cfm <- table(pred=rpart_predict,true=testset$default)
cfm

plotcp(rpart_model)

##GRAPHS
#simple
plot(rpart_model)
prp(rpart_model)

# Try out an even prettier graph. 
# Customise as per https://www.rdocumentation.org/packages/rpart.plot/versions/3.0.6/topics/rpart.plot
rpart.plot(rpart_model, # middle graph
           type=2,
           extra=101, 
           box.palette="GnBu",
           shadow.col="gray"
)

#pruned model
#Initialising variables for determining at what value to prune
optmin <- which.min(rpart_model$cptable[,"xerror"])
cpminerr <- rpart_model$cptable[optmin, "xerror"]
cpminse <- rpart_model$cptable[optmin, "xstd"]
opt1se <- which.min(abs(rpart_model$cptable[,"xerror"]-cpminerr-cpminse))
# Value of the complexity parameter (alpha) for that gives a a tree of optmin or opt1se
cpmin <- rpart_model$cptable[optmin, "CP"]
cp1se <- rpart_model$cptable[opt1se, "CP"]
# "prune" the tree using that value of the complexity parameter (try both cpmin and cp1se)
pruned_model <- prune(rpart_model,0.005) 
rpart.plot(pruned_model, # middle graph
           type=2,
           extra=101, 
           box.palette="GnBu",
           shadow.col="gray"
)
# Predict on test data
rpart_predict <- predict(pruned_model,testset[,-class_col_num],type="class")
# Create a confusion matrix
cfm <- table(pred=rpart_predict,true=testset$default)
cfm


accuracies <- rep(NA,100)
for (i in 1:100){
  creditdata[,"train"] <- ifelse(runif(nrow(creditdata))<0.8,1,0)
  trainset <- creditdata[creditdata$train==1,]
  testset <- creditdata[creditdata$train==0,]
  trainColNum <- grep("train",names(trainset))
  trainset <- trainset[,-trainColNum]
  testset <- testset[,-trainColNum]
  rpart_model <-  rpart(default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data = trainset, method = "class")
  
  rpart_predict <- predict(rpart_model,testset[,-class_col_num],type="class")
#  pred_test <- predict(svm_model,testset)
  accuracies[i] <- mean(rpart_predict==testset$default)
  
#  cfm <- table(pred=rpart_predict,true=testset$default)
#  accuracies[i] <- (cfm[1,1]+cfm[2,2])/sum(cfm)
}



#SVM model
#linear
svm_model<- svm(default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data=trainset, type="C-classification",  kernel="linear", cost=100, scale=FALSE)
#polynomial deg=2
svm_model<- svm(default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data=trainset, type="C-classification",  kernel="polynomial", degree=3, cost=20)
svm_model<- svm(default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + COUNTS + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data=trainset, type="C-classification",  kernel="polynomial", degree=2)
#radial
svm_model<- svm(default ~ LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE + PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 + BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6, data=trainset, type="C-classification",  kernel="radial", scale=FALSE)
#radial reduced
svm_model <- svm(default ~ LIMIT_BAL + AGE + PAY_0 + PAY_2 + EDUCATION, data=trainset, type="C-classification",  kernel="radial", cost=10, scale=FALSE)
summary(svm_model)

plot(svm_model, trainset)

pred_train <- predict(svm_model,trainset)
mean(pred_train==trainset$default)
pred_test <- predict(svm_model,testset)
mean(pred_test==testset$default)

cfm <- table(pred=pred_test,true=testset$default)
cfm

#Accuracy = fraction of correct predictions
#i.e. -sum of diagonal terms in confusion matrix by the total number of instances
accuracy <- (cfm[1,1]+cfm[2,2])/sum(cfm)
accuracy

#Precision = TP/(TP+FP)
#precision <- cfm[1,1]/(cfm[1,1]+cfm[1,2])
precision <- cfm[2,2]/(cfm[2,2]+cfm[2,1])
precision

#Recall = TP/(TP+FN)
#recall <- cfm[1,1]/(cfm[1,1]+cfm[2,1])
recall <- cfm[2,2]/(cfm[2,2]+cfm[1,2])
recall

#F1
f1 <- 2*(precision*recall/(precision+recall))
f1

#EVAL
testdata <- read_csv("AT2_credit_test.csv")
val_rpart_predict <- predict(rpart_model,testdata,type="class")
val_rpart_predict_prob <- predict(rpart_model,testdata,type="prob")
#write.csv(val_rpart_predict,file="validation_predictions.csv",row.names = FALSE)
write.csv(val_rpart_predict_prob,file="validation_predictions.csv",row.names = FALSE)
