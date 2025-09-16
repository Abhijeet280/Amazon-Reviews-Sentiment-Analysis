# 🛍️ Amazon Reviews Sentiment Analysis

This project performs **sentiment analysis** on Amazon product reviews to classify them as **Positive**, **Negative**, or **Neutral**.
It uses **Natural Language Processing (NLP)** and **Machine Learning algorithms** to process review text, handle class imbalance, and predict customer sentiment.

---

## 🚀 Project Overview

The notebook executes the following steps:

1. **Importing Libraries**

   * Pandas, NumPy for data manipulation
   * Matplotlib, WordCloud for visualization
   * NLTK for text preprocessing
   * Scikit-learn for vectorization, model building, and evaluation
   * Imbalanced-learn for handling class imbalance with **SMOTE**.

2. **Data Loading & Cleaning**

   * Dataset: `Musical_instruments_reviews.csv`
   * Filled missing values in `reviewText` column.
   * Combined `reviewText` and `summary` into a single `reviews` column for analysis.

3. **Text Preprocessing with NLTK**

   * Lowercasing text.
   * Removing punctuation, special characters, and stopwords.
   * Lemmatization using **WordNetLemmatizer**.
   * Tokenization of review text.

4. **Sentiment Labeling**

   * Based on `overall` rating:

     * **Positive** → rating > 3
     * **Negative** → rating < 3
     * **Neutral** → rating = 3

5. **Feature Engineering**

   * **CountVectorizer** and **TF-IDF Vectorizer** for text-to-numeric conversion.
   * WordCloud visualization for most common words.

6. **Handling Imbalanced Data**

   * Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes and prevent bias toward majority class.

7. **Model Building**
   Multiple machine learning models were trained and compared:

   * Logistic Regression
   * Support Vector Classifier (SVC)
   * Random Forest Classifier
   * Decision Tree Classifier
   * Bernoulli Naive Bayes
   * K-Nearest Neighbors (KNN)

8. **Hyperparameter Tuning**

   * **GridSearchCV** was applied to optimize model performance.

9. **Model Evaluation**

   * Evaluated models using:

     * **Accuracy Score**
     * **Confusion Matrix**
     * **Classification Report**
     * **Cross-validation scores**

---

## 🛠️ Technologies Used

* **Python**
* **Natural Language Toolkit (NLTK)** – Tokenization, Stopword Removal, Lemmatization
* **TextBlob** – Sentiment polarity detection
* **Scikit-learn** – Machine learning models and evaluation metrics
* **Imbalanced-learn (SMOTE)** – Handling class imbalance
* **Matplotlib & WordCloud** – Visualization
* **Pandas & NumPy** – Data manipulation
* **Jupyter Notebook**

---

## 📂 Dataset

The dataset `Musical_instruments_reviews.csv` contains Amazon product reviews with the following key features:

| Column Name    | Description                            |
| -------------- | -------------------------------------- |
| **reviewText** | Full review text provided by customers |
| **summary**    | Short summary of the review            |
| **overall**    | Numerical product rating (1 to 5)      |

---

## 📊 Expected Insights

* Identify **most frequent positive and negative keywords** using WordCloud.
* Understand **distribution of sentiments** before and after SMOTE balancing.
* Determine the **best-performing machine learning model** for sentiment prediction.
* Logistic Regression or Random Forest are expected to perform well for classification.


