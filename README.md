# Natural-Language-Processing


## Step 1: Reading the dataset
import pandas as pd
df = pd.read_csv("AMAZON-REVIEW-DATA-CLASSIFICATION.csv")
print('The shape of the dataset is:', df.shape)
df.head(5)


## What it does:

Imports the pandas library (used for handling tables of data).

Reads a CSV file containing Amazon reviews.

Prints the size of the dataset (rows x columns) and shows the first 5 rows.

Algorithmically:

Input: CSV file

Output: DataFrame (df) storing the data in memory

## Step 2: Exploratory Data Analysis (EDA)
df['isPositive'].value_counts()
df = df.replace({0:1, 1:0})
df.isna().sum()


## What it does:

Checks how many positive vs. negative reviews (value_counts()).

Flips 0 and 1 labels (maybe the original encoding was reversed).

Checks for missing values in all columns (isna().sum()).

## Algorithmically:

Count labels and missing values.

Transform labels if needed for consistency.

## Step 3: Text Processing (Stop words removal and stemming)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import re


## Key steps:

Download NLTK resources for text processing.

Define stop words (common words like "the", "and" which we usually remove).

Define a stemmer (reduces words to root form, e.g., "running" → "run").

## Clean the text:

Lowercase, strip whitespace, remove HTML tags.

Tokenize words.

Remove numbers, stop words, and short words.

Stem words.

Join tokens back into a string.

## Algorithmically:

for each review in dataset:
    if review is missing:
        replace with empty string
    convert text to lowercase
    remove extra spaces and HTML tags
    tokenize text into words
    for each word:
        if word is not numeric, longer than 2, and not a stop word:
            stem the word
    join words back to form cleaned review

## Step 4: Training, Validation, and Test Split
from sklearn.model_selection import train_test_split


What it does:

Splits the data into:

Training set (used to train the model)

Validation set (used to tune hyperparameters)

Test set (used to evaluate final model performance)

Processes the text fields using the process_text function from Step 3.

## Algorithmically:

Split dataset into 80% train, 20% temp
Split temp into 50% validation, 50% test
Process 'reviewText' and 'summary' using text cleaning function

## Step 5: Data processing with Pipeline and ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer


## What it does:

Numerical features (time, log_votes) → fill missing values, scale between 0-1.

Text features → convert to vectors (binary count of words using CountVectorizer).

Use ColumnTransformer to apply different preprocessing steps to different columns.

Fit the transformer on training data, transform validation and test sets.

Algorithmically:

For numerical features:
    fill missing values with mean
    scale values between 0 and 1
For 'summary' text:
    convert top 50 words to binary vector
For 'reviewText' text:
    convert top 150 words to binary vector
Combine all features into one array for the model

## Step 6: Train a classifier and evaluate
import xgboost as xgb
from sklearn.metrics import accuracy_score


## What it does:

Define an XGBoost classifier:

objective='binary:logistic' → binary classification

n_estimators=100 → 100 trees

max_depth=5 → max depth of trees

learning_rate=0.1 → step size for boosting

subsample and colsample_bytree → control randomness

Fit the model on training data.

Predict labels for validation and test sets.

Calculate accuracy.

Algorithmically:

Input: Preprocessed training features and labels
XGBoost builds trees sequentially:
    For each tree:
        Fit on errors of previous tree
        Limit depth to 5
        Randomly sample rows/columns for robustness
Output: Trained model
Use model to predict labels for validation/test
Calculate accuracy = correct predictions / total predictions

Summary of the Full Algorithm

Load and inspect data

Flip labels if needed and check for missing values

Clean and preprocess text (remove stop words, stemming, lowercase, tokenization)

Split dataset into training, validation, and test sets

Preprocess numerical and text features (scaling, vectorization)

Train XGBoost binary classifier on processed features

Predict and evaluate accuracy on validation and test sets

 This pipeline is a classic NLP classification workflow:

Text cleaning → feature extraction → train/test split → model training → evaluation.
