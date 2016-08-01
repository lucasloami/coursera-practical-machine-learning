library(caret)
library(ggplot2)
library(klaR)
library(randomForest)

set.seed(12345)


#load data into memory
training_data <- read.csv("data/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))

testing_data <- read.csv("data/pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))

# summary(training_data)
# summary(testing_data)
# names(training_data)
# names(testing_data)
# dim(training_data)
# dim(testing_data)

# Remove variables that are irrelevant for this project
training_data <- training_data[,-c(1:7)]
testing_data <- testing_data[,-c(1:7)]

# Remove variables that contains NA
# we need to do this, otherwise caret will show errors 
# training_data <- training_data[, colSums(is.na(training_data)) != nrow(training_data)]

NAs <- apply(training_data, 2, function(x) {
    sum(is.na(x))
})
training_data <- training_data[, which(NAs == 0)]

# Remove near zero variance variables
nzv_training_cols <- nearZeroVar(training_data)

if(length(nzv_training_cols) > 0) training_data <- training_data[, -nzv_training_cols]


# Model Selection
tc <- trainControl(method = "cv", number = 5, allowParallel=TRUE, verboseIter=FALSE,  preProcOptions="pca")

# Naive Bayes
nb <- train(classe ~ ., data = training_data, method = "nb", trControl= tc)

# Random Forest
rf <- train(classe ~ ., data = training_data, method = "rf", trControl= tc)

# Classification Trees
ct <- train(classe ~ ., data = training_data, method = "rpart", trControl= tc)

print(c("Naive Bayes", max(nb$results$Accuracy)))
print(c("Random Forest", max(rf$results$Accuracy)))
print(c("Classification Tree", max(ct$results$Accuracy)))

predictions <- predict(rf, testing_data)

# Distribution of 'classe' variable in the training set
p <- ggplot(data = training_data, aes(x=classe)) 
p <- p + geom_bar()
p

