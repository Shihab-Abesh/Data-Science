################ Dataset Information
# Dataset Link: https://drive.google.com/uc?export=download&id=160hZgo5_LXDOQykEhqzLmLkwJO2TSVkR
# Source: Keggle
# Description: contains information about students' performance, including attributes such as gender, race/ethnicity, parental education level, lunch type, test preparation course, and scores for math, reading, and writing.

# Install and Load Necessary Packages
install.packages(c("tidyverse", "dplyr", "ggplot2", "caret", "corrplot", "reshape2", "e1071", "cluster"))
library(tidyverse)
library(dplyr)
library(ggplot2)
library(reshape2)
library(caret)
library(corrplot)
library(e1071)
library(cluster)

# Set a random seed for reproducibility
set.seed(123)

########### A. DATA UNDERSTANDING ####################
# 1. Load dataset
# Import the dataset directly from a URL to avoid hard-coded paths
url <- "https://drive.google.com/uc?export=download&id=160hZgo5_LXDOQykEhqzLmLkwJO2TSVkR"
data <- read.csv(url)

# 2. Display first few rows
head(data)

# 3. Show dataset shape (rows & columns)
cat("Rows: ", nrow(data), "\n")
cat("Columns: ", ncol(data), "\n")

# 4. Display data types
# Check the structure of the dataset, showing data types of each column
str(data)

# 5. Basic descriptive statistics
# Display basic summary statistics (mean, median, min, max, etc.)
print(summary(data))

# 6. Identify categorical vs numerical features
# Identify which columns are categorical (factor/character) and which are numerical
cat_cols <- names(data)[sapply(data, function(x) is.character(x) || is.factor(x))]
num_cols <- names(data)[sapply(data, is.numeric)]
cat("Categorical columns:\n"); print(cat_cols)
cat("Numeric columns:\n"); print(num_cols)

########### B. DATA EXPLORATION & VISUALIZATION ###########
# Missing values
# Count the number of missing values in each column
cat("Missing values per column:\n")
print(colSums(is.na(data)))

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

# Bar plots for categorical variables
# Create bar plots for the first 6 categorical columns to understand their distributions
cat_show <- head(cat_cols, 7)
for (c in cat_show) {
  print(
    ggplot(data, aes_string(x = c)) +
      geom_bar(fill = "steelblue") +
      ggtitle(paste("Bar plot:", c)) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )
}

# Correlation heatmap for numeric variables
# A correlation heatmap helps visualize the relationships between numeric variables
if (length(num_cols) >= 2) {
  num_data <- data[, num_cols, drop = FALSE]
  cor_matrix <- cor(num_data, use = "pairwise.complete.obs")
  corrplot(cor_matrix, method = "color", addCoef.col = "black", number.cex = 0.6)
}

########### C. DATA PREPROCESSING ####################
# Handle missing values
# Function to calculate the mode for categorical columns (most frequent value)
getmode <- function(v) {
  v <- v[!is.na(v)]  # Remove NA values
  uniqv <- unique(v)  # Get unique values
  uniqv[which.max(tabulate(match(v, uniqv)))]  # Return the most frequent value (mode)
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

# Handle outliers using IQR (Interquartile Range) method
# We calculate the lower and upper bounds for each column using the IQR and cap the outliers at these bounds
for (c in num_cols) {
  Q1 <- quantile(data[[c]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[c]], 0.75, na.rm = TRUE)
  IQRv <- Q3 - Q1
  lower <- Q1 - 1.5 * IQRv  # Lower bound for outliers
  upper <- Q3 + 1.5 * IQRv  # Upper bound for outliers
  data[[c]] <- ifelse(data[[c]] < lower, lower, ifelse(data[[c]] > upper, upper, data[[c]]))
}

# Standardize features
# Standardizing the data to ensure each variable has mean = 0 and standard deviation = 1
# Step 1: Identify zero variance columns
# These columns have the same value for all rows, so they won't contribute to the model
zero_var_cols <- sapply(data, function(x) length(unique(x)) == 1)

# Print the names of zero variance columns
cat("Zero variance columns:", names(data)[zero_var_cols], "\n")

# Step 2: Remove zero variance columns
# Keep only the columns that have more than one unique value (non-zero variance)
data_clean <- data[, !zero_var_cols]

# Step 3: Recompute num_cols based on the cleaned data (after removing zero variance columns)
# This ensures num_cols refers to the numeric columns left in data_clean
num_cols <- names(data_clean)[sapply(data_clean, is.numeric)]

# Step 4: Standardize the features (only the non-zero variance columns)
# Standardizing the data to ensure each variable has mean = 0 and standard deviation = 1
pp <- preProcess(data_clean[, num_cols], method = c("center", "scale"))
data_clean <- predict(pp, data_clean)

# Step 5: Check the cleaned and standardized data
head(data_clean)

########### D. CLUSTERING (K-Means) ####################

# 1. Identify numeric columns only
# Select only the numeric columns from the dataset
numeric_data <- data[, sapply(data, is.numeric)]

# 2. Remove rows with missing values (required for K-Means)
# Remove rows with missing values in numeric columns
numeric_data <- na.omit(numeric_data)

# 3. Scale the numeric data
# Standardizing the data to ensure each variable has mean = 0 and standard deviation = 1
data_scaled <- scale(numeric_data)

# Step 4: Check if there are any NA/NaN/Inf values in the scaled data
cat("Checking for NA, NaN, or Inf values in the scaled data...\n")
cat("Rows with NA/NaN/Inf values after scaling:\n")

# Check for NA, NaN, or Inf values in the scaled data
na_nan_inf_scaled <- apply(data_scaled, 1, function(x) any(is.na(x) | is.nan(x) | is.infinite(x)))

# If any problematic rows exist, print the number of such rows
cat("Rows with NA/NaN/Inf values after scaling:", sum(na_nan_inf_scaled), "\n")

# If there are problematic rows, remove them
if (sum(na_nan_inf_scaled) > 0) {
  data_scaled <- data_scaled[!na_nan_inf_scaled, ]
  cat("Problematic rows with NA/NaN/Inf have been removed.\n")
}

# Elbow Method: Determine the optimal number of clusters (k)
wss <- numeric(10)
for (k in 2:11) {
  km <- kmeans(data_scaled, centers = k, nstart = 10)
  wss[k - 1] <- km$tot.withinss  # Store the WSS for each k
}

# Create a data frame to plot the elbow curve
elbow_df <- data.frame(k = 2:11, WSS = wss)

# Plot the elbow method to visualize the optimal k
print(
  ggplot(elbow_df, aes(x = k, y = WSS)) +
    geom_line() +
    geom_point() +
    ggtitle("Elbow Method (WSS vs k)") +
    theme_minimal()
)

# Based on the elbow plot, choose k = 3 (or any other optimal value from the plot)
k <- 3
kmeans_model <- kmeans(data_scaled, centers = k, nstart = 25)

# Step 5: Silhouette Score: Measure the quality of the clusters
sil <- silhouette(kmeans_model$cluster, dist(data_scaled))  # Calculate silhouette scores
sil_score <- mean(sil[, 3])  # Calculate the average silhouette score
cat("Silhouette Score:", round(sil_score, 4), "\n")

########### E. CLUSTER VISUALIZATION (PCA) ################
# Principal Component Analysis (PCA) for visualization
pca <- prcomp(data_scaled, center = TRUE, scale. = TRUE)  # Apply PCA to scaled data
pca_df <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2], Cluster = as.factor(kmeans_model$cluster))

# Plot the clusters in 2D using PCA components
print(
  ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(alpha = 0.7) +
    ggtitle("Cluster Visualization using PCA (PC1 vs PC2)") +
    theme_minimal()
)

# Print the results of the K-Means clustering
cat("K-Means clustering completed. Chosen k =", k, "\n")
cat("Cluster sizes:\n")
print(kmeans_model$size)
