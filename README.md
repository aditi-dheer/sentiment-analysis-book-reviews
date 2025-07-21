# Book Review Sentiment Analysis  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
</p>  

## Overview  
This project performs **sentiment analysis on Amazon book reviews** using a simple and effective machine learning pipeline. Reviews are vectorized using **TF-IDF** and classified with **Logistic Regression**. The model is evaluated using **ROC-AUC**, achieving a strong discriminative performance on unseen data.  

## Key Points  
- **Loads review data** from a CSV file using pandas  
- **Splits dataset** into training and test sets  
- **Transforms text** into numerical features via `TfidfVectorizer`  
- **Trains Logistic Regression** on the TF-IDF vectors  
- **Evaluates model performance** using ROC-AUC score

## Evaluation

| Metric   | Score   |
|----------|---------|
| ROC-AUC  | ~0.915  |

---

## Installation  
Clone the repository and install required dependencies:  
```bash
git clone https://github.com/aditi-dheer/sentiment-analysis-book-reviews.git
cd book-sentiment-analysis
pip install scikit-learn pandas numpy
```

## Usage

Run the sentiment analysis pipeline:

```bash
jupyter notebook book_reviews_sentiment_analysis.ipynb
```

It will output the **ROC-AUC score** on the test set.

---

## Project Structure

- **bookReviews.csv** – Dataset of book reviews  
- **book_reviews_sentiment_analysis.ipynb** – Main script for training and evaluation  
- **README.md** – Project documentation  
