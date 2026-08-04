# 🎬 IMDb Movie Review Sentiment Classification

**Binary text classification of movie reviews using classical NLP and machine learning**

**Python 3.x&nbsp;&nbsp;·&nbsp;&nbsp;scikit-learn&nbsp;&nbsp;·&nbsp;&nbsp;TF-IDF / NLP&nbsp;&nbsp;·&nbsp;&nbsp;Status: Complete**

---

## 📖 Overview

This project builds and evaluates multiple machine learning models for **binary text classification** using the IMDb Movie Reviews dataset. The goal is to classify movie reviews as either **positive** or **negative** based solely on their textual content.

It demonstrates a complete natural language processing (NLP) workflow, including:

- 🗂️ Data preparation
- ⚖️ Handling class imbalance
- 🔤 Text vectorization using TF-IDF
- 🤖 Training multiple classification models
- 📊 Model evaluation
- 🎯 Hyperparameter tuning with Grid Search

> Although this project focuses on sentiment analysis, the same workflow can be adapted for other text classification problems — spam detection, topic classification, customer feedback analysis, or support ticket categorization.

---

## 📁 Dataset

The project uses the **IMDb Movie Review Dataset**. Each record contains:

| Column      | Description                                  |
|-------------|-----------------------------------------------|
| `review`    | The text of the movie review                  |
| `sentiment` | Label — `positive` or `negative`              |

---

## 🔄 Project Workflow

### 1️⃣ Loading the Dataset

The dataset is loaded into a Pandas DataFrame:

```python
df_reviews = pd.read_csv(...)
```

This imports the raw review text and corresponding sentiment labels.

### 2️⃣ Creating an Imbalanced Dataset

To simulate a real-world scenario where one class dominates, the dataset is intentionally skewed to approximately:

- **9,000** positive reviews
- **1,000** negative reviews

The resulting imbalance is visualized using a bar chart.

### 3️⃣ Balancing the Dataset

Rather than oversampling the minority class, **Random Under Sampling** (from `imbalanced-learn`) is used:

```python
RandomUnderSampler()
```

This randomly removes samples from the majority class until both classes are equal in size, producing a balanced dataset while keeping the original review text unchanged.

### 4️⃣ Train-Test Split

The balanced dataset is split into:

- **Training set:** 67%
- **Testing set:** 33%

```python
train_test_split(...)
```

Only the training data is used to learn model parameters; the test data remains unseen until final evaluation.

### 5️⃣ TF-IDF Vectorization

Since ML models can't process raw text, reviews are converted into numerical feature vectors:

```python
TfidfVectorizer(stop_words="english")
```

**Why TF-IDF?** Term Frequency–Inverse Document Frequency assigns higher importance to words that appear frequently *within* a review but *rarely* across the entire dataset. Common English stop words are removed automatically.

> ⚠️ **Important:** The vectorizer is fitted **only on training data** (`fit_transform(train_x)`) and then applied to test data (`transform(test_x)`) to prevent information leakage.

---

## 🤖 Models Trained

Four supervised learning algorithms are trained and compared:

| # | Model | Notes |
|---|-------|-------|
| 1 | **Support Vector Machine** (`SVC(kernel="linear")`) | Finds the optimal boundary between classes; a strong baseline for sparse, high-dimensional text data |
| 2 | **Decision Tree** (`DecisionTreeClassifier()`) | Splits the feature space by informative words; easy to interpret but prone to overfitting on text features |
| 3 | **Gaussian Naive Bayes** (`GaussianNB()`) | Requires dense arrays (`toarray()`); assumes feature independence, making it computationally efficient |
| 4 | **Logistic Regression** (`LogisticRegression()`) | Estimates class probabilities; another strong, widely-used NLP baseline |

---

## 📊 Model Evaluation

Model performance is assessed using several complementary metrics:

- **Confusion Matrix** — reports True/False Positives and Negatives for a detailed view beyond raw accuracy
- **Accuracy** — via `model.score(...)`; proportion of correctly classified reviews (can be misleading on imbalanced data)
- **F1 Score** — balances precision and recall, useful when both false positives and false negatives matter
- **Classification Report** — precision, recall, F1-score, and support for each class

---

## 🎯 Hyperparameter Tuning

The SVM is optimized using `GridSearchCV` with 5-fold cross-validation:

```python
C = [1, 4, 8, 16, 32]
kernel = ["linear", "rbf"]
```

The search retrieves the **best parameter combination** and **best-performing estimator**, identifying the configuration that generalizes best on the training data.

---

## 🧰 Libraries Used

- `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`

**Key scikit-learn components:** `TfidfVectorizer`, `train_test_split`, `SVC`, `DecisionTreeClassifier`, `GaussianNB`, `LogisticRegression`, `GridSearchCV`, `confusion_matrix`, `classification_report`, `f1_score`

---

## 📂 Project Structure

```
Project
│
├── Project 4 - Text Classification.ipynb
├── IMDB Dataset.csv
└── README.md
```

---

## ✅ Strengths

- End-to-end NLP classification pipeline
- Compares multiple ML algorithms on the same dataset
- Addresses class imbalance via random under-sampling
- Uses TF-IDF, a widely adopted text representation technique
- Evaluates with multiple metrics rather than accuracy alone
- Includes hyperparameter tuning with cross-validation

---

## ⚠️ Limitations

- Class imbalance is handled through **random under-sampling**, which discards many positive reviews and may lose useful information
- Only **TF-IDF features** are used — word order, sentence structure, and context are not captured
- Minimal preprocessing beyond stop-word removal (no stemming, lemmatization, or negation handling)
- Hyperparameter tuning is applied only to the SVM; other models use default settings
- Metrics are reported on a single train-test split — results may vary across different splits
- Deep learning approaches (RNNs, LSTMs, transformer models like BERT) are outside the scope of this project

---

## 🚀 Possible Improvements

- Lemmatization or stemming
- N-gram features
- Stratified cross-validation
- Probability calibration
- Model persistence using `joblib`
- Pipeline objects for cleaner preprocessing and training
- Comparison with transformer-based language models
- Interactive web deployment using Streamlit or Gradio

---

## 🏁 Conclusion

This project demonstrates a complete supervised machine learning workflow for sentiment analysis using traditional NLP techniques. Starting from an intentionally imbalanced dataset, it applies class balancing, converts text into TF-IDF feature vectors, trains multiple classification models, evaluates them using several performance metrics, and improves the SVM through hyperparameter tuning.

It offers a practical introduction to classical text classification while highlighting the importance of thoughtful preprocessing, appropriate evaluation metrics, and systematic model selection.

---

<p align="center"><i>Built as part of a hands-on NLP & machine learning portfolio project.</i></p>
