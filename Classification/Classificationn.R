# Dataset Information
# Dataset Link: https://drive.google.com/uc?export=download&id=171fUhcfZBNhq74xIehaqfK3KfMvwt-IZ
# Source: Kaggle / UCI / Data.gov (real-world)
# Description: Heart Disease dataset used to predict the presence of heart disease (target variable: HeartDisease)
# Ensure the code is dataset-independent and can run without modification on any machine.

# Install and Load Necessary Packages
# These are the packages required for data manipulation, visualization, and modeling
install.packages(c("tidyverse", "dplyr", "ggplot2", "caret", "corrplot", "reshape2", "e1071", "rpart", "rpart.plot"))
library(tidyverse)
library(dplyr)
library(ggplot2)
library(reshape2)
library(caret)
library(corrplot)
library(e1071)
library(rpart)
library(rpart.plot)

# Set a random seed for reproducibility
set.seed(123)

########### A. DATA UNDERSTANDING ####################
# 1. Load dataset
# Importing data directly from the URL, ensuring the code is dataset-independent
url <- "https://drive.google.com/uc?export=download&id=171fUhcfZBNhq74xIehaqfK3KfMvwt-IZ"
data <- read.csv(url)

# 2. Display first few rows
# View the first few rows of the dataset to understand its structure
head(data)

# 3. Show dataset shape (rows & columns)
# Print the number of rows and columns in the dataset
cat("Rows: ", nrow(data), "\n")
cat("Columns: ", ncol(data), "\n")

# 4. Display data types
# Check the structure of the dataset, showing data types of each column
str(data)

# 5. Basic descriptive statistics
# Display basic summary statistics (mean, median, min, max, etc.)
print(summary(data))

# 6. Identify categorical vs numerical features
# Identify which columns are categorical and which are numerical
cat_cols <- names(data)[sapply(data, function(x) is.character(x) || is.factor(x))]
num_cols <- names(data)[sapply(data, is.numeric)]
cat("Categorical columns:\n"); print(cat_cols)
cat("Numeric columns:\n"); print(num_cols)

# Target column
# Define the target variable we want to predict (HeartDisease)
target_col <- "HeartDisease"
# Check if the target column exists in the dataset
if (!(target_col %in% names(data))) {
  stop(paste("Target column not found:", target_col, "\nUpdate 'target_col' to match your dataset."))
}

########### B. DATA EXPLORATION & VISUALIZATION ###########
# Missing values
# Count the number of missing values in each column
cat("Missing values per column:\n")
print(colSums(is.na(data)))

# Basic target distribution
# Check the distribution of the target variable (HeartDisease) to understand class imbalance
print(table(data[[target_col]]))

# Histograms for numeric variables
# Create histograms for the first 6 numeric columns to explore their distributions
num_show <- head(num_cols, 6)
for (c in num_show) {
  print(
    ggplot(data, aes_string(x = c)) +
      geom_histogram(bins = 30, fill = "skyblue", color = "black") +
      ggtitle(paste("Histogram:", c)) +
      theme_minimal()
  )
}

# Boxplots for numeric variables
# Create boxplots for the first 6 numeric columns to check for outliers
for (c in num_show) {
  print(
    ggplot(data, aes_string(y = c)) +
      geom_boxplot(fill = "orange", color = "black") +
      ggtitle(paste("Boxplot:", c)) +
      theme_minimal()
  )
}

# Correlation heatmap for numeric variables
# Create a correlation heatmap to visualize the relationships between numeric variables
if (length(num_cols) >= 2) {
  num_data <- data[, num_cols, drop = FALSE]
  cor_matrix <- cor(num_data, use = "pairwise.complete.obs")
  corrplot(cor_matrix, method = "color", addCoef.col = "black", number.cex = 0.7)
}

########### C. DATA PREPROCESSING ####################
# Handle missing values
# Define a function to calculate the mode for categorical variables
getmode <- function(v) {
  v <- v[!is.na(v)]  # Remove NA values
  uniqv <- unique(v)  # Get unique values
  uniqv[which.max(tabulate(match(v, uniqv)))]  # Return the most frequent value
}
# Replace missing numeric values with the column mean
for (c in num_cols) {
  data[[c]][is.na(data[[c]])] <- mean(data[[c]], na.rm = TRUE)
}
# Replace missing categorical values with the mode
for (c in cat_cols) {
  data[[c]][is.na(data[[c]])] <- getmode(data[[c]])
  data[[c]] <- as.factor(data[[c]])  # Convert to factor after imputation
}

# Handle outliers using IQR
# Use the Interquartile Range (IQR) method to cap outliers to a valid range
for (c in num_cols) {
  Q1 <- quantile(data[[c]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[c]], 0.75, na.rm = TRUE)
  IQRv <- Q3 - Q1
  lower <- Q1 - 1.5 * IQRv  # Lower bound for outliers
  upper <- Q3 + 1.5 * IQRv  # Upper bound for outliers
  data[[c]] <- ifelse(data[[c]] < lower, lower, ifelse(data[[c]] > upper, upper, data[[c]]))
}

# Feature engineering: Add a new feature CholPerAge
# This is an example feature engineering step where we create a new feature CholPerAge
if (all(c("Cholesterol", "Age") %in% names(data))) {
  data$CholPerAge <- data$Cholesterol / pmax(data$Age, 1)  # Prevent division by zero
  num_cols <- unique(c(num_cols, "CholPerAge"))
}

# Standardize features
# Standardize the numeric features to ensure they have a mean of 0 and a standard deviation of 1
pp <- preProcess(data[, num_cols], method = c("center", "scale"))
data <- predict(pp, data)

########### D. MODELING (Decision Tree) #################
# Train/Test split
# Split the dataset into training and testing sets (80% for training, 20% for testing)
train_index <- createDataPartition(data[[target_col]], p = 0.80, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Train Decision Tree model
# Use the rpart function to train a decision tree model on the training data
tree_model <- rpart(HeartDisease ~ ., data = train_data, method = "class")
cat("\nDecision Tree Model Summary:\n")
print(summary(tree_model))

# Plot the decision tree
# Visualize the trained decision tree
rpart.plot(tree_model, main = "Decision Tree")

########### E. EVALUATION & INTERPRETATION ################
# Make predictions
pred <- predict(tree_model, newdata = test_data, type = "class")

# Ensure actual labels and predictions are factors with the same levels
actual <- as.factor(test_data[[target_col]])
all_levels <- union(levels(pred), levels(actual))
pred <- factor(pred, levels = all_levels)
actual <- factor(actual, levels = all_levels)

# Calculate confusion matrix
cm <- confusionMatrix(pred, actual)
print(cm)

# Model Interpretation
cat("\nModel Interpretation:\n")
cat(" - High Precision: Fewer false positives\n")
cat(" - High Recall: Fewer false negatives\n")
cat(" - F1-score: Balances Precision and Recall\n")
