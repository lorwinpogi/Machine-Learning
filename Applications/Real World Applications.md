# Real World Applications

# Credit Card Fraud Detection using Machine Learning

## Introduction

Credit card fraud detection is a binary classification task in which machine learning is used to distinguish between legitimate and fraudulent transactions. Fraudulent transactions are rare compared to normal transactions, making this a highly imbalanced classification problem.

## Problem Definition

- **Objective**: Detect fraudulent credit card transactions
- **Classes**:
  - Class 0: Legitimate
  - Class 1: Fraudulent
- **Challenges**:
  - Class imbalance (fraud cases are very few)
  - High cost of false negatives (missing a fraud)
  - Need for real-time or near real-time prediction



## Workflow

1. Load and explore the dataset
2. Preprocess features
3. Handle class imbalance
4. Train a model
5. Evaluate performance

---

## 1: Load and Explore the Data

```python
import pandas as pd

# Load dataset
data = pd.read_csv('creditcard.csv')

# Basic shape and class distribution
print("Shape:", data.shape)
print("Class distribution:")
print(data['Class'].value_counts())

```


## 2: Preprocess the Data

```python
from sklearn.preprocessing import StandardScaler

# Normalize the 'Amount' column
data['Amount'] = StandardScaler().fit_transform(data[['Amount']])

# Drop the 'Time' column (optional)
data.drop('Time', axis=1, inplace=True)

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']
```

## 3: Handle Class Imbalance
Option A: Undersampling

```python

# Separate fraud and non-fraud
fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0].sample(n=len(fraud), random_state=42)

# Combine into a balanced dataset
balanced_data = pd.concat([fraud, non_fraud])

X_bal = balanced_data.drop('Class', axis=1)
y_bal = balanced_data['Class']

```

Option B: SMOTE Oversampling





```python

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Train-test split on original imbalanced data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

```


## 4: Train the Model
```python
from sklearn.ensemble import RandomForestClassifier

# Use undersampled or SMOTE-resampled data as input
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)
```


## 5: Evaluate the Model

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

```

### Summary
Credit card fraud detection is a rare event detection problem.

Preprocessing and class balancing (undersampling or SMOTE) are essential.

Random Forest is a strong baseline model for tabular data.

Performance should be evaluated using precision, recall, and F1-score, especially for the fraud class.



# Wealth Management in Machine Learning

## Introduction

Wealth management involves strategic financial planning and investment decision-making to help individuals and institutions grow and preserve wealth. Machine learning (ML) is increasingly used in this field to enhance portfolio management, client segmentation, risk assessment, and personalized advisory services.


## Applications of Machine Learning in Wealth Management

### 1. Portfolio Optimization
- Predict returns and volatility
- Allocate assets based on risk preferences
- Use reinforcement learning or regression models

### 2. Client Segmentation
- Cluster clients based on net worth, risk tolerance, income, and goals
- Deliver personalized products and advice

### 3. Risk Assessment
- Predict credit and investment risk
- Detect early signs of financial distress
- Use classification models

### 4. Robo-Advisory
- Provide algorithm-driven financial planning with minimal human interaction
- Tailor asset allocation and rebalancing automatically

### 5. Sentiment Analysis
- Analyze news, reports, or earnings calls to influence asset valuation
- Use NLP (Natural Language Processing)


## Sample Machine Learning Use Case: Client Risk Profiling

### Objective:
Classify clients into risk categories (e.g., low, medium, high) based on historical data.


##  1: Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
```

## 2: Prepare Sample Data

```python
# Example dataset
data = {
    'age': [35, 60, 45, 25, 50, 30, 40],
    'annual_income': [80000, 120000, 100000, 45000, 110000, 60000, 95000],
    'investment_experience': [5, 20, 10, 2, 18, 4, 8],  # in years
    'net_worth': [200000, 1500000, 750000, 50000, 1200000, 100000, 600000],
    'risk_tolerance': [1, 3, 2, 0, 3, 1, 2]  # 0: Low, 1: Medium, 2: Medium-High, 3: High
}

df = pd.DataFrame(data)

X = df.drop('risk_tolerance', axis=1)
y = df['risk_tolerance']
```


## 3: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


## 4: Train Model

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## 5: Evaluate Model
```python
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```


## 6: Predict for a New Client
```python
# Predict risk tolerance for a new client
new_client = pd.DataFrame([{
    'age': 38,
    'annual_income': 95000,
    'investment_experience': 6,
    'net_worth': 400000
}])

risk_class = model.predict(new_client)[0]
print("Predicted Risk Tolerance Class:", risk_class)
```

### Summary
Machine learning in wealth management enables:

Improved decision-making through data-driven insights

Automated client profiling and portfolio recommendations

Scalable personalization for financial advisors and firms

Models such as classification (for client profiles), regression (for return prediction), clustering (for segmentation), and NLP (for unstructured text analysis) are commonly used. These models enhance traditional wealth management processes by making them more accurate, efficient, and scalable.

# Predicting Student Behavior In Machine Learning

## Introduction

Predicting student behavior is a valuable application of machine learning (ML) in educational technology. It helps educators and institutions understand how students engage, perform, and drop out, enabling early intervention, personalized learning, and better academic outcomes.

## Common Use Cases

### 1. Dropout Prediction
- Identify students at risk of leaving a course or program early

### 2. Performance Forecasting
- Predict final grades or test outcomes based on historical data

### 3. Engagement Detection
- Analyze platform interaction (clicks, videos watched, forum participation)

### 4. Cheating or Plagiarism Detection
- Use behavioral patterns and submission similarities

### 5. Course Recommendation
- Suggest courses based on student interests and performance


## Example Use Case: Predicting Student Performance

**Goal**: Predict whether a student will pass or fail a course based on demographic and academic features.



## 1: Import Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
```



## 2: Prepare Example Dataset
```python
# Simulated dataset
data = {
    'age': [17, 18, 17, 19, 18, 20, 21],
    'study_hours': [5, 12, 3, 15, 10, 2, 6],
    'attendance_rate': [0.6, 0.9, 0.4, 1.0, 0.85, 0.3, 0.7],
    'previous_grade': [60, 85, 55, 90, 80, 50, 65],
    'passed': [0, 1, 0, 1, 1, 0, 1]  # 0: Fail, 1: Pass
}

df = pd.DataFrame(data)

X = df.drop('passed', axis=1)
y = df['passed']
```

## 3: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 4: Train Model
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## 5: Evaluate Model
```python
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## 6: Predict New Student Outcome
```python
# New student input
new_student = pd.DataFrame([{
    'age': 18,
    'study_hours': 8,
    'attendance_rate': 0.75,
    'previous_grade': 70
}])

result = model.predict(new_student)[0]
print("Prediction (1 = Pass, 0 = Fail):", result)
```


### Summary
Machine learning enables educational institutions to:

Proactively support students at risk

 Optimize learning paths

Improve course design

Personalize academic interventions

Common algorithms include decision trees, random forests, support vector machines, and neural networks. Data typically includes demographic details, attendance, activity logs, previous grades, and interaction data from learning platforms.
