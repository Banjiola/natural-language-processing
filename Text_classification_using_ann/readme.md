# 🧠 News Topic Classification with Deep Learning

> *"Classifying news articles into four distinct categories using deep learning on the AG News dataset."*

---

## 📌 Table of Contents
- [Overview](#overview)
- [Business Objective](#business-objective)
- [Dataset](#dataset)
- [Tools & Techniques](#tools--techniques)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results & Insights](#results--insights)
- [Dashboard / Visuals](#dashboard--visuals)
- [Conclusion](#conclusion)
- [Author](#author)

---

## 🧩 Overview

This project involves classifying English news headlines and short descriptions into one of four categories: World, Sports, Business, and Sci/Tech. Using the AG News dataset, we explored the text distribution, preprocessed the content, and trained various deep learning models including MLPs, CNNs, and RNNs with different embeddings. The goal is to build an accurate and efficient text classifier for use in news aggregation, filtering, or alerting systems.

---

## 🎯 Business Objective

> **Target user**: News aggregators, media houses, and content recommendation systems.  
> **Value**: Automatic classification of incoming articles improves user experience and enables personalized news feeds.

Example:
> "To streamline news article categorization to enhance personalized content delivery and reduce manual tagging workload."

---

## 📊 Dataset

- **Source**: [TorchText AG News Dataset](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
- **Size**: ~120,000 training samples, ~7,600 test samples
- **Classes**: 4 (World, Sports, Business, Sci/Tech)
- **Features**: Title, Description, Label (category)

---

## 🛠️ Tools & Techniques

| **Category**       | **Tools**                                     |
|--------------------|-----------------------------------------------|
| Language           | Python                                        |
| Data Processing    | `pandas`, `numpy`                             |
| NLP Preprocessing  | `re`, `nltk`, `tensorflow.keras.layers.TextVectorization` |
| Modeling           | `TensorFlow`, `Keras`                         |
| Visualization      | `matplotlib`, `seaborn`, `WordCloud`         |
| Evaluation         | Accuracy, Precision, Recall, F1, AUC          |

---

## 🔎 Exploratory Data Analysis

- Class distribution bar chart showing balanced dataset
- Word clouds per category for frequent terms
- Token count distributions and padding decisions

---

## 🏗️ Feature Engineering

- Lowercased all text
- Removed punctuation and stopwords
- Applied `TextVectorization` for converting raw text to integer sequences
- Padding to equal sequence lengths
- Defined vocabulary size = 10,000 and max sequence length = 200

---

## 🤖 Modeling

- **Models Tried**:  
  - Multilayer Perceptron (MLP)  
  - Convolutional Neural Network (CNN)  
  - Recurrent Neural Network (RNN)  
  - Bidirectional LSTM (BiLSTM)  

- **Embedding Strategies**:
  - Random initialization
  - Pretrained: GloVe, Word2Vec, FastText

- **Evaluation Metrics**:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - AUC

---

## 📈 Results & Insights

| **Model**            | **Precision** | **Recall** | **F1 Score** | **AUC**  |
|----------------------|---------------|------------|--------------|----------|
| MLP Random           | 0.91          | 0.91       | 0.91         | 0.976    |
| BiLSTM GloVe         | 0.92          | 0.92       | 0.92         | 0.9166   |
| CNN GloVe            | 0.92          | 0.92       | 0.92         | 0.9153   |
| RNN GloVe            | 0.92          | 0.91       | 0.91         | 0.9147   |
| BiLSTM Word2Vec      | 0.91          | 0.91       | 0.91         | 0.9129   |
| BiLSTM FastText      | 0.91          | 0.91       | 0.91         | 0.9120   |
| MLP GloVe            | 0.91          | 0.91       | 0.91         | 0.9101   |
| CNN FastText         | 0.91          | 0.91       | 0.91         | 0.9099   |
| CNN Word2Vec         | 0.91          | 0.91       | 0.91         | 0.9095   |
| CNN Random           | 0.91          | 0.91       | 0.91         | 0.9091   |
| BiLSTM Random        | 0.91          | 0.91       | 0.91         | 0.9089   |
| RNN FastText         | 0.91          | 0.91       | 0.91         | 0.9078   |
| RNN Word2Vec         | 0.91          | 0.91       | 0.91         | 0.9067   |
| MLP Word2Vec         | 0.91          | 0.91       | 0.91         | 0.9066   |
| RNN Random           | 0.91          | 0.91       | 0.91         | 0.9057   |
| MLP FastText         | 0.90          | 0.90       | 0.90         | 0.9021   |

- All models achieved over 90% accuracy
- GloVe-embedded BiLSTM and CNN models had the best overall F1 and precision-recall balance

---

## 📊 Dashboard / Visuals

- Confusion matrix per model (optional)
- Word clouds by class
- Model training/validation accuracy and loss plots

---

## 🧾 Conclusion

- Deep learning models effectively classify AG News articles with high accuracy.
- Pretrained embeddings offer slight boosts in recall/F1 depending on model type.
- MLPs are computationally efficient with strong AUC, while BiLSTMs and CNNs provide robust precision-recall balance.



---

## 👨‍💻 Author

**Olabanji Olaniyan**  
Data Scientist  
📫 [LinkedIn](https://www.linkedin.com/in/olabanji-olaniyan-59a6b0198/) | [Portfolio](https://banjiola.github.io/Olabanji-Olaniyan/)
