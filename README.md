# 📰 Fake News Detection (TF-IDF + Logistic Regression)

## 📌 Overview

This project classifies news articles as **FAKE** or **REAL** using:

- Text preprocessing and cleaning  
- TF-IDF vectorization (unigrams + bigrams)  
- Logistic Regression classifier  

It also visualizes:

- Class distribution  
- Article length distribution  
- Confusion matrix  
- Most influential words for FAKE and REAL news  

Users can input news text to get real-time predictions.

---

## 🚀 Key Features

✔ Text cleaning (lowercase, remove special characters)  
✔ TF-IDF feature extraction  
✔ Logistic Regression classifier  
✔ Accuracy evaluation  
✔ Confusion matrix visualization  
✔ Top indicative words for FAKE and REAL news  
✔ Interactive news prediction  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Regex  

---

## 📂 Dataset

The script expects a CSV file:

```
15_fake_news_detection.csv
```

Required columns:

- `title` → Headline of the news  
- `text` → Full article content  
- `label` → "FAKE" or "REAL"  

The script combines `title` and `text` for model input.

---

## 🔎 Project Workflow

### 1️⃣ Text Cleaning

- Convert text to lowercase  
- Remove numbers and special characters  
- Keep only alphabetic characters  

```python
clean_text()
```

---

### 2️⃣ Data Visualization

- Bar plot of class distribution (Fake vs Real)  
- Histogram of article lengths per class  

---

### 3️⃣ Train-Test Split

- 80% training  
- 20% testing  
- Stratified to preserve class balance  

---

### 4️⃣ TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)
```

- Unigrams + Bigrams  
- Removes English stopwords  
- Limits vocabulary to 5000 features  

---

### 5️⃣ Model Training

Model used:

```
Logistic Regression
```

- Maximum iterations: 1000  
- Suitable for binary classification  

---

### 6️⃣ Evaluation

Metrics used:

- Accuracy  
- Confusion matrix (visualized as heatmap)  
- Top words indicating FAKE and REAL news  

Top 10 words per class are displayed using bar charts.

---

## 🔮 Interactive Prediction

Users can type news text to get predictions:

```
Type news text (or 'exit' to quit):
```

Returns:

- "REAL NEWS"  
- "FAKE NEWS"  

Example:

- Input: `"Government releases new stimulus package"` → `"REAL NEWS"`  
- Input: `"Celebrity endorses miracle diet pills"` → `"FAKE NEWS"`  

---

## 📦 Installation

Install required packages:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## ▶️ How to Run

```bash
python your_script_name.py
```

Ensure `15_fake_news_detection.csv` is in the same directory.

---

## 🎯 Use Cases

- Fake news detection  
- Social media content verification  
- Media analytics  
- Text classification learning project  
- NLP experimentation  

---

## 📈 What This Project Demonstrates

- Text preprocessing techniques  
- TF-IDF feature engineering  
- Logistic Regression for classification  
- Model interpretability using feature coefficients  
- Interactive prediction system  

---

## 👨‍💻 Author

Built as part of Natural Language Processing and fake news detection experimentation.

If you found this helpful, consider starring the repository ⭐
