############################################################
#   INSTALL AND LOAD NECESSARY PACKAGES
############################################################

# Installation of packages 
install.packages(c(
  "tidyverse",      # Data manipulation + visualization
  "dplyr",          # Data wrangling
  "ggplot2",        # Visualization
  "reshape2",       # For correlation heatmap
  "caret",          # Preprocessing
  "corrplot"        # Correlation plot
))

# Loading packages
library(tidyverse)
library(dplyr)
library(ggplot2)
library(reshape2)
library(caret)
library(corrplot)


############################################################
###########   A. DATA UNDERSTANDING     ####################
############################################################

# 1. Load dataset
url <- "https://drive.google.com/uc?export=download&id=160hZgo5_LXDOQykEhqzLmLkwJO2TSVkR"
data <- read.csv(url)

# 2. Display first few rows
head(data)

# 3. Show dataset shape (rows & columns)
cat("Rows: ", nrow(data), "\n")
cat("Columns: ", ncol(data), "\n")

# 4. Display data types
str(data)

# 5. Basic descriptive statistics
summary(data)

# Additional statistics (mean, median, mode, std, etc.)
numeric_cols <- sapply(data, is.numeric)
num_data <- data[, numeric_cols]

apply(num_data, 2, mean, na.rm = TRUE)     # Mean
apply(num_data, 2, median, na.rm = TRUE)   # Median
apply(num_data, 2, sd, na.rm = TRUE)       # Standard deviation
apply(num_data, 2, min, na.rm = TRUE)      # Min
apply(num_data, 2, max, na.rm = TRUE)      # Max

# Function to compute mode
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

apply(num_data, 2, getmode)                # Mode

# 6. Identify categorical vs numerical features
cat_cols <- names(data)[sapply(data, is.character)]
num_cols <- names(data)[sapply(data, is.numeric)]

cat_cols
num_cols


############################################################
#######   B. DATA EXPLORATION & VISUALIZATION    ###########
############################################################

### 1. UNIVARIATE ANALYSIS

# Identify numeric and categorical features
numeric_features <- names(data)[sapply(data, is.numeric)]
categorical_features <- names(data)[sapply(data, is.character)]

options(repr.plot.width = 5, repr.plot.height = 6)


# 1. Histogram for math.score (capped at 100)
ggplot(data %>% filter(math.score <= 100), aes(x = math.score)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  scale_x_continuous(limits = c(0, 100)) +      # X-axis fixed: 0-100
  ggtitle("Histogram of Math Score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 20))

### Histogram for reading.score
ggplot(data %>% filter(reading.score <= 100), aes(x = reading.score)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  scale_x_continuous(limits = c(0, 100)) +      # X-axis fixed: 0-100
  ggtitle("Histogram of reading score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 20))

###Histogram for writing.score
ggplot(data %>% filter(writing.score <= 100), aes(x = writing.score)) +
  geom_histogram(bins = 30, fill = "green", color = "black") +
  scale_x_continuous(limits = c(0, 100)) +      # X-axis fixed: 0-100
  ggtitle("Histogram of writing score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 20))


# 2. Boxplots (Numeric)
# Boxplot for math.score (capped at 0-100)
ggplot(data %>% filter(math.score <= 100), aes(y = math.score)) +
  geom_boxplot(fill = "red", color = "black") +
  scale_y_continuous(limits = c(0, 100)) +      # Y-axis fixed: 0-100
  ggtitle("Boxplot of Math Score (0-100)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16))

# Boxplot for reading.score (capped at 0-100)
ggplot(data %>% filter(reading.score <= 100), aes(y = reading.score)) +
  geom_boxplot(fill = "gray", color = "black") +
  scale_y_continuous(limits = c(0, 100)) +      # Y-axis fixed: 0-100
  ggtitle("Boxplot of Reading Score (0-100)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16))

# Boxplot for writing.score (capped at 0-100)
ggplot(data %>% filter(writing.score <= 100), aes(y = writing.score)) +
  geom_boxplot(fill = "yellow", color = "black") +
  scale_y_continuous(limits = c(0, 100)) +      # Y-axis fixed: 0-100
  ggtitle("Boxplot of Reading Score (0-100)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16))


# 3. Barplots (Categorical)

# Frequency distribution of math.score (0-100)
ggplot(data %>% filter(math.score <= 100), aes(x = math.score)) +
  geom_histogram(binwidth = 5, fill = "brown", color = "black") +
  scale_x_continuous(limits = c(0, 100)) +
  ggtitle("Frequency Distribution of Math Score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16))

# Frequency distribution of reading.score (0-100)
ggplot(data %>% filter(reading.score <= 100), aes(x = reading.score)) +
  geom_histogram(binwidth = 5, fill = "lightyellow", color = "black") +
  scale_x_continuous(limits = c(0, 100)) +
  ggtitle("Frequency Distribution of reading Score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16))

# Frequency distribution of writing.score (0-100)
ggplot(data %>% filter(writing.score <= 100), aes(x = writing.score)) +
  geom_histogram(binwidth = 5, fill = "lightgreen", color = "black") +
  scale_x_continuous(limits = c(0, 100)) +
  ggtitle("Frequency Distribution of writing Score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16))


### 2. BIVARIATE ANALYSIS

# Correlation matrix + heatmap
num_data <- data %>% select_if(is.numeric)
cor_matrix <- cor(num_data, use = "complete.obs")
corrplot(cor_matrix, method = "color", addCoef.col = "red", number.cex = 1.2)

# Scatter plots between numeric pairs
for (i in 1:(length(num_cols)-1)) { for (j in (i+1):length(num_cols))
  { print( ggplot(data, aes_string(num_cols[i], num_cols[j])) + geom_point() + ggtitle(paste("Scatterplot:", num_cols[i], "vs", num_cols[j])) ) } }



# Boxplots between categorical & numeric variables
for (cat in cat_cols) {
  for (num in num_cols) {
    print(
      ggplot(data, aes_string(cat, num)) +
        geom_boxplot() +
        ggtitle(paste("Boxplot of", num, "by", cat))
    )
  }
}


### 3. Detect patterns, skewness, outliers 

# Load necessary library
if(!require(e1071)) install.packages("e1071")
library(e1071)

# Select numeric columns
num_cols <- names(data)[sapply(data, is.numeric)]

# Loop through each numeric column
for(col in num_cols){
  x <- data[[col]]  # extract the column
  
  # Basic statistics
  min_val <- min(x, na.rm = TRUE)
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  median_val <- median(x, na.rm = TRUE)
  mean_val <- mean(x, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  max_val <- max(x, na.rm = TRUE)
  
  # Skewness
  skew <- skewness(x, na.rm = TRUE)
  
  # Detect outliers (IQR method)
  IQR_val <- q3 - q1
  lower <- q1 - 1.5 * IQR_val
  upper <- q3 + 1.5 * IQR_val
  outliers <- x[x < lower | x > upper]
  
  # Print results
  cat("Column:", col, "\n")
  cat("Min:", min_val, " Q1:", q1, " Median:", median_val, " Mean:", mean_val, " Q3:", q3, " Max:", max_val, "\n")
  cat("Skewness:", round(skew, 2), "\n")
  if(length(outliers) == 0){
    cat("Outliers: None\n")
  } else {
    cat("Outliers:", paste(round(outliers, 2), collapse = ", "), "\n")
  }
  cat("################################################################\n")
  cat("################################################################\n")
  cat("################################################################\n")
}






############################################################
############   C. DATA PREPROCESSING    ####################
############################################################

### 1. HANDLING MISSING VALUES

# Detect missing values
colSums(is.na(data))

# Replace numeric missing values with mean
data[num_cols] <- lapply(data[num_cols], function(x){
  ifelse(is.na(x), mean(x, na.rm = TRUE), x)
})

# Replace categorical missing values with mode
data[cat_cols] <- lapply(data[cat_cols], function(x){
  ifelse(is.na(x), getmode(x), x)
})


### 2. HANDLING OUTLIERS (IQR METHOD)

for (col in num_cols) {
  Q1 <- quantile(data[[col]], 0.25)
  Q3 <- quantile(data[[col]], 0.75)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  # Capping outliers
  data[[col]] <- ifelse(data[[col]] < lower, lower,
                        ifelse(data[[col]] > upper, upper, data[[col]]))
}


### 3. DATA CONVERSION (ENCODING)

# One-hot encoding for categorical variables
data_encoded <- dummyVars(" ~ .", data = data)
data_processed <- data.frame(predict(data_encoded, newdata = data))


### 4. DATA TRANSFORMATION (Scaling & Normalization)

# Apply Z-score standardization
data_scaled <- as.data.frame(scale(data_processed))


### 5. FEATURE SELECTION


# Keep only numeric columns (after encoding)
numeric_data <- data_processed %>% select_if(is.numeric)

# Remove zero-variance columns (they produce NA correlations)
nzv <- nearZeroVar(numeric_data)
if (length(nzv) > 0) {
  numeric_data <- numeric_data[, -nzv]
}

# Scale numeric data
numeric_data_scaled <- scale(numeric_data)

# Safe correlation matrix
cor_matrix <- cor(numeric_data_scaled, use = "pairwise.complete.obs")

# Replace NA correlations
cor_matrix[is.na(cor_matrix)] <- 0

# Remove highly correlated features (> 0.80)
high_cor <- findCorrelation(cor_matrix, cutoff = 0.80)

# Final selected features
data_final <- numeric_data_scaled[, -high_cor]

# Check the final columns
names(data_final)

# Show selected features
names(data_final)
