# 🩺 Breast Cancer Detection Using Machine Learning  
A machine learning project exploring clinical prediction, model evaluation, and data-driven decision making.
### *Using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) on the Breast Cancer Wisconsin dataset*

![GitHub last commit](https://img.shields.io/github/last-commit/AsianaHolloway/Breast-Cancer-Detection-ML?color=6aa6f8&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/AsianaHolloway/Breast-Cancer-Detection-ML?color=6aa6f8&style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=flat-square)
![ML Models](https://img.shields.io/badge/Models-KNN%20%7C%20SVM-blueviolet?style=flat-square)
![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange?style=flat-square)

🔗 **Interactive Google Colab Notebook:**  
https://colab.research.google.com/drive/1bRmJ6R5NdQ66lRAu6uTOjAI-IlBImHih?usp=sharing  

---

## 📘 Project Overview

This project uses the **Breast Cancer Wisconsin (Original)** dataset from the UCI Machine Learning Repository to build machine learning models that classify tumors as **benign** or **malignant**.

The work is split into two parts but implemented in a **single Colab notebook**:

🧪 **Part 1 – Exploratory Data Analysis (EDA)**  
  - Load and clean the dataset  
  - Explore feature distributions  
  - Visualize relationships between features and the target label
 
✔ Key Tasks Completed
  - Loaded the UCI dataset
  - Inspected distributions of tumor features
  - Visualized outliers and patterns
  - Generated a correlation heatmap
  - Explored differences between benign and malignant cases


## ⭐ What I Learned

Interpreting multivariate relationships

Understanding which tumor characteristics matter most

Preparing real medical data for ML modeling

🤖 **Part 2 – Machine Learning Models**  
  - Train and evaluate:
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Machine (SVM)**
  - Compute **accuracy, precision, recall, F1-score**
  - Visualize performance using **confusion matrices**
  - Compare models and discuss which one is better for breast cancer detection

Seven different models were trained, evaluated, and compared:

| Model                | Accuracy | Precision | Recall | F1-Score |
| -------------------- | -------- | --------- | ------ | -------- |
| **SVM (RBF)** ⭐      | 0.9375   | 0.944     | 0.9375 | 0.9375   |
| AdaBoost             | 0.8750   | 0.875     | 0.875  | 0.875    |
| SVM (linear)         | 0.8750   | 0.875     | 0.875  | 0.875    |
| SVM (sigmoid)        | 0.8750   | 0.809     | 0.875  | 0.874    |
| KNN                  | 0.8437   | 0.856     | 0.843  | 0.842    |
| MLP (Neural Network) | 0.8437   | 0.845     | 0.843  | 0.843    |
| GaussianNB           | 0.7188   | 0.820     | 0.718  | 0.694    |


The overall goal is to understand how ML can support **early detection** and **clinical decision support** in breast cancer.

## 🏆 Best Model: SVM with RBF Kernel
Why it performed best:

Highest accuracy (93.75%)

Excellent precision & recall balance

Strong generalization to unseen data

Captures nonlinear decision boundaries

Low misclassification of malignant cases (clinically critical)

Clinical relevance:

The RBF SVM delivered the best diagnostic performance, reducing the risk of false negatives, which is crucial in early breast cancer detection.

## 🧠 Code Walkthrough 

🔹 Data Loading & Cleaning

pd.read_csv(url, names=names)
Loads the dataset from the UCI URL and assigns human-readable column names.

df.replace('?', 99999, inplace=True)
The dataset encodes missing values as '?'. Here, they are temporarily replaced so the column can be converted to numeric. In a more advanced pipeline, we could impute or drop them.

df.drop(['id'], axis=1, inplace=True)
The ID column is just an identifier and does not contain clinical information, so it is removed before training to avoid noise.

🔹 Feature/Label Split

X = np.array(df.drop(['class'], axis=1))
Creates a NumPy array with only the predictor variables (tumor features).

y = np.array(df['class'])
Stores the target label indicating benign vs. malignant.

This separation is critical to prevent data leakage and to properly train supervised models.

🔹 Train/Test Split

test_size=0.2
Reserves 20% of the data for testing to simulate unseen patient cases.

random_state=42
Sets a seed so results are reproducible, which is important when comparing models.

🔹 KNN Intuition

KNeighborsClassifier(n_neighbors=5)
Each new sample is classified by looking at its 5 nearest neighbors in feature space and assigning the majority label.

Works well when classes are relatively well separated in the feature space.

🔹 SVM Intuition

SVC(kernel='rbf')
The SVM projects data into a higher-dimensional space using an RBF kernel and finds the best margin separating benign and malignant classes.

It is powerful for complex, nonlinear decision boundaries common in medical data.

🔹 Evaluation Metrics

accuracy_score(y_test, y_pred)
Overall fraction of correct predictions.

classification_report(...)
Provides precision, recall, F1-score, which are especially important in healthcare:

Precision: Of the tumors predicted malignant, how many truly are malignant?

Recall (Sensitivity): Of all truly malignant tumors, how many did we correctly catch?

F1-score: Balance between precision and recall.

In breast cancer detection, high recall for the malignant class is crucial since missing a malignant case (false negative) can delay diagnosis and treatment.

## 🏆 Model Selection & Conclusion

Based on the evaluation:

Both KNN and SVM achieved high performance, with accuracies around 95–96%.

KNN slightly outperformed SVM in overall accuracy in this experiment.

Both models showed strong precision and recall, but KNN had a small edge in balancing the metrics.

## 📌 Chosen Model:
For this dataset, I would slightly favor KNN because:

It achieved slightly higher accuracy on the held-out test set.

It is simpler to explain: predictions are based on “nearest neighbors” in feature space.

It is relatively easy to tune (by adjusting n_neighbors) and extend.

However, SVM remains a strong alternative and might generalize better on more complex or imbalanced datasets.

## 🎓 Skills & Knowledge Gained

Practical experience building end-to-end ML pipelines on clinical data.

Understanding of how data preprocessing impacts model performance.

Hands-on practice comparing KNN vs. SVM for a real medical classification problem.

Interpreting confusion matrices and classification reports in the context of patient safety and diagnostic accuracy.

## 💬 Contact

Asiana Holloway
📍 M.S. Health Informatics – Michigan Tech
---


