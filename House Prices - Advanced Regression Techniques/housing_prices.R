# Load necessary libraries
library(caret)
library(xgboost)
library(ranger)
library(gbm)
library(nnet)

# Load the dataset
train_data <- read.csv("train.csv")
test_data <- read.csv("test.csv")

# Data preprocessing
# Handling missing values by filling with the mean
train_data[is.na(train_data)] <- mean(train_data, na.rm=TRUE)
test_data[is.na(test_data)] <- mean(test_data, na.rm=TRUE)

# Feature selection (you can modify this based on your analysis)
features <- c("OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt")
target <- "SalePrice"

# Split the data into predictors and target variable
X <- train_data[features]
y <- train_data[[target]]

# Train GBM model
gbm_model <- train(X, y, method = "gbm", trControl = trainControl(method = "cv", number = 5), verbose = FALSE)

# Train XGBoost model
xgb_model <- xgboost(data = as.matrix(X), label = y, nrounds = 100, verbose = 0)

# Train Ranger model
ranger_model <- ranger(y ~ ., data = X, num.trees = 100, write.forest = TRUE)

# Train Neural Network model
nn_model <- nnet(y ~ OverallQual + GrLivArea + GarageCars + TotalBsmtSF + FullBath + YearBuilt, 
                      data = train_data, size = 5, linear.output = TRUE)

# Make predictions using the models
gbm_preds <- predict(gbm_model, newdata = test_data)
xgb_preds <- predict(xgb_model, as.matrix(test_data[features]))
ranger_preds <- predict(ranger_model, data = test_data[features])
nn_preds <- predict(nn_model, newdata = test_data)

# Ensemble predictions (average of predictions)
ensemble_preds <- (gbm_preds + xgb_preds + ranger_preds + nn_preds) / 4

# Prepare the submission file
submission <- data.frame(Id = test_data$Id, SalePrice = ensemble_preds)

# Save the submission file
write.csv(submission, file = "submission.csv", row.names = FALSE)
