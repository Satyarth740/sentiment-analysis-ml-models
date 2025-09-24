
# Sentiment Analysis with Multiple ML Models

## 📌 Overview
This project performs **sentiment analysis** on a dataset of text reviews.  
The task is to classify reviews as **positive** or **negative** using multiple machine learning models and compare their performance.

## ⚙️ Approach
1. **Data Preparation**
   - Load the dataset (`Review`, `Sentiment` columns).
   - Clean and preprocess text.
   - Convert text into numerical features using **TF-IDF vectorization**.

2. **Model Training**
   - Train four different ML models:
     - Logistic Regression
     - Random Forest
     - Naive Bayes (MultinomialNB)
     - Decision Tree
   - Split the dataset into **training (80%)** and **testing (20%)**.

3. **Evaluation**
   - Evaluate each model using the following metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - Compare results to determine the best-performing model.

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.8942   | 0.8827    | 0.9111 | 0.8967   |
| Random Forest       | 0.8596   | 0.8652    | 0.8545 | 0.8598   |
| Naive Bayes         | 0.8652   | 0.8753    | 0.8541 | 0.8646   |
| Decision Tree       | 0.7249   | 0.7343    | 0.7115 | 0.7227   |



✅ In this task, **Logistic Regression** achieved the highest accuracy.


