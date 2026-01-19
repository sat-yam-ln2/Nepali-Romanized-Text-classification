# Romanized Nepali Text Classification

This project implements and compares various machine learning models for sentiment analysis/classification of Romanized Nepali text. The goal is to classify text into Negative, Neutral, and Positive sentiment categories.

## Dataset
The dataset consists of **1,792** Romanized Nepali sentences labeled with sentiments:
*   **-1**: Negative
*   **0**: Neutral
*   **1**: Positive

## Methodology
The project explores both classical machine learning and modern transformer-based approaches:

1.  **Classical Models**:
    *   **Preprocessing**: TF-IDF Vectorization (unigrams and bigrams, max 5000 features).
    *   **Algorithms**:
        *   Logistic Regression
        *   Support Vector Machine (SVM)
        *   Random Forest
2.  **Deep Learning**:
    *   **Model**: **mBERT** (Multilingual BERT), fine-tuned for sequence classification.


### Performance Metrics Table

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.813370 | 0.827892 | 0.813370 | **0.819233** |
| **SVM** | 0.816156 | 0.819321 | 0.816156 | 0.815104 |
| **Random Forest** | **0.827298** | 0.819059 | **0.827298** | 0.808402 |
| **mBERT** | **0.827298** | **0.832099** | **0.827298** | 0.813046 |


## Usage
To replicate the results:
1.  Ensure all dependencies are installed (`pandas`, `numpy`, `sklearn`, `transformers`, `torch`, `matplotlib`, `seaborn`, `plotly`).
2.  Run the Jupyter Notebook: `romanized_nepali_classification.ipynb`.
