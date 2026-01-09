########## Dataset Information
# Dataset Link: https://drive.google.com/uc?export=download&id=1hvufXTktx1bwuseAe_oVjB38z-Zz9zy9
# Source: Kaggle 
# Description: Bank Marketing dataset used to predict a continuous value using customer and campaign features


# Install and Load Necessary Packages

install.packages(c("tidyverse", "dplyr", "ggplot2", "caret", "corrplot", "reshape2", "e1071"))
library(tidyverse)
library(dplyr)
library(ggplot2)
library(reshape2)
library(caret)
library(corrplot)
library(e1071)

# Set a random seed for reproducibility
set.seed(123)

########### A. DATA UNDERSTANDING ####################
# 1. Load dataset
# Import the dataset directly from a URL
url <- "https://drive.google.com/uc?export=download&id=1hvufXTktx1bwuseAe_oVjB38z-Zz9zy9"
data <- read.csv(url)

# 2. Display first few rows
head(data)

# 3. Show dataset shape (rows & columns)
cat("Rows: ", nrow(data), "\n")
cat("Columns: ", ncol(data), "\n")

# 4. Display data types
str(data)

# 5. Basic descriptive statistics
print(summary(data))

# 6. Identify categorical vs numerical features
cat_cols <- names(data)[sapply(data, function(x) is.character(x) || is.factor(x))]
num_cols <- names(data)[sapply(data, is.numeric)]
cat("Categorical columns:\n"); print(cat_cols)
cat("Numeric columns:\n"); print(num_cols)

# Target column (continuous)
target_col <- "balance"
if (!(target_col %in% names(data))) {
  stop(paste("Target column not found:", target_col, "\nUpdate 'target_col' to match your dataset."))
}

########### B. DATA EXPLORATION & VISUALIZATION ###########
# Missing values
cat("Missing values per column:\n")
print(colSums(is.na(data)))

# Target distribution plot
# Visualize the distribution of the target variable (balance)
print(
  ggplot(data, aes_string(x = target_col)) +
    geom_histogram(bins = 40, fill = "skyblue", color = "black") +
    ggtitle(paste("Target Distribution:", target_col)) +
    theme_minimal()
)

# Histograms for numeric variables
num_show <- head(num_cols, 6)
for (c in num_show) {
  print(
    ggplot(data, aes_string(x = c)) +
      geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
      ggtitle(paste("Histogram:", c)) +
      theme_minimal()
  )
}

########### C. DATA PREPROCESSING ####################
# Handle missing values
getmode <- function(v) {
  v <- v[!is.na(v)]
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]  # Return the mode (most frequent value)
}
for (c in num_cols) {
  data[[c]][is.na(data[[c]])] <- mean(data[[c]], na.rm = TRUE)
}
for (c in cat_cols) {
  data[[c]][is.na(data[[c]])] <- getmode(data[[c]])
  data[[c]] <- as.factor(data[[c]])  # Convert to factor after imputation
}

# Handle outliers using IQR
for (c in num_cols) {
  Q1 <- quantile(data[[c]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[c]], 0.75, na.rm = TRUE)
  IQRv <- Q3 - Q1
  lower <- Q1 - 1.5 * IQRv
  upper <- Q3 + 1.5 * IQRv
  data[[c]] <- ifelse(data[[c]] < lower, lower, ifelse(data[[c]] > upper, upper, data[[c]]))
}

# Feature engineering (if applicable)
if (all(c("duration", "campaign") %in% names(data))) {
  data$dur_per_campaign <- data$duration / pmax(data$campaign, 1)
  num_cols <- unique(c(num_cols, "dur_per_campaign"))
}

# Standardize features
pp <- preProcess(data[, num_cols], method = c("center", "scale"))
data <- predict(pp, data)

########### D. MODELING (Linear Regression) #############
# Train/Test split
# Split data into training and testing sets (80% training, 20% testing)
train_index <- createDataPartition(data[[target_col]], p = 0.80, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Train Linear Regression model
# Fit the linear regression model using the training data
lm_model <- lm(balance ~ ., data = train_data)
cat("\nLinear Regression Model Summary:\n")
print(summary(lm_model))

########### E. EVALUATION & INTERPRETATION ################
# Make predictions
# Predict the target variable (balance) on the test set
pred <- predict(lm_model, newdata = test_data)

# Calculate RMSE, MAE, and R²
rmse <- sqrt(mean((pred - test_data[[target_col]])^2))  # Root Mean Squared Error
mae  <- mean(abs(pred - test_data[[target_col]]))  # Mean Absolute Error
r2   <- cor(pred, test_data[[target_col]])^2  # R-squared

# Print metrics
cat("\nRegression Metrics:\n")
cat("RMSE:", round(rmse, 4), "\n")
cat("MAE :", round(mae, 4), "\n")
cat("R^2 :", round(r2, 4), "\n")

# Model Interpretation
cat("\nModel Interpretation:\n")
cat(" - Lower RMSE and MAE indicate better prediction accuracy\n")
cat(" - Higher R² indicates better model fit and explanatory power\n")
