# Real World Applications

## Credit Card Fraud Detection using Machine Learning

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



## Wealth Management in Machine Learning

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

## Predicting Student Behavior In Machine Learning

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


## Mitigating Bias in Machine Learning

Bias in machine learning (ML) refers to systematic errors that result in unfair outcomes, such as preferring one group over another. These biases can be introduced through data collection, model design, or deployment processes. Bias mitigation is critical in building fair, ethical, and legally compliant AI systems.

## Types of Bias

- **Historical Bias**: Embedded in the dataset due to past societal inequalities.
- **Sampling Bias**: Caused by non-representative data collection.
- **Measurement Bias**: Occurs when the features used to train the model inaccurately represent the real-world attributes.
- **Algorithmic Bias**: Arises from model assumptions and training processes.

## General Bias Mitigation Strategies

Bias mitigation can occur at three stages:

### 1. Pre-processing (Before Training)

- **Re-sampling**: Balance the dataset (e.g., oversample minority classes).
- **Reweighting**: Assign different weights to samples to counteract imbalance.
- **Data augmentation**: Generate synthetic examples for underrepresented groups.

### 2. In-processing (During Training)

- **Fairness constraints**: Modify loss functions to penalize unfairness.
- **Adversarial debiasing**: Train a model while trying to remove group information via adversaries.

### 3. Post-processing (After Training)

- **Equalized odds post-processing**: Adjust the model's predictions to equalize performance metrics across groups.
- **Thresholding**: Apply different decision thresholds per group.

## Example: Mitigating Bias with Reweighing (Pre-processing)

We will use the `Adult` dataset from the UCI repository. The target is whether an individual earns more than $50K/year. We will mitigate gender bias.

### Dependencies

```python
!pip install fairlearn scikit-learn pandas numpy
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

# Load dataset
data = fetch_openml(data_id=1590, as_frame=True)  # Adult dataset
df = data.frame

# Basic preprocessing
df = df.dropna()
X = df.drop('class', axis=1)
y = (df['class'] == '>50K').astype(int)

# Encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)
y = y.reset_index(drop=True)

# Sensitive feature: sex
A = X['sex'].reset_index(drop=True)

# Train/test split
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X_encoded, y, A, test_size=0.3, random_state=42
)

# Baseline model without mitigation
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Bias metrics (without mitigation)
mf = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=A_test)
print("Selection rates (no mitigation):")
print(mf.by_group)

print("Demographic Parity Difference (no mitigation):")
print(demographic_parity_difference(y_test, y_pred, sensitive_features=A_test))

# Fair model with mitigation (Exponentiated Gradient)
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(solver='liblinear'),
    constraints=DemographicParity(),
    eps=0.01
)

mitigator.fit(X_train, y_train, sensitive_features=A_train)
y_pred_fair = mitigator.predict(X_test)

# Bias metrics (with mitigation)
mf_fair = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred_fair, sensitive_features=A_test)
print("\nSelection rates (with mitigation):")
print(mf_fair.by_group)

print("Demographic Parity Difference (with mitigation):")
print(demographic_parity_difference(y_test, y_pred_fair, sensitive_features=A_test))

# Accuracy comparison
print("\nAccuracy (no mitigation):", accuracy_score(y_test, y_pred))
print("Accuracy (with mitigation):", accuracy_score(y_test, y_pred_fair))
```

## Key Takeaways
The ExponentiatedGradient method with a DemographicParity constraint aims to equalize the selection rate across groups (in this case, male and female).

You can compare selection rates and demographic parity differences before and after mitigation.

Fairness often introduces a trade-off with accuracy. Evaluate both metrics carefully in production systems.

## Tools for Bias Mitigation
Fairlearn – Algorithms for fairness constraints and metrics.

AIF360 – IBM's comprehensive fairness toolkit.

What-If Tool – Interactive visual debugging for fairness.

## Summary
Bias in machine learning can originate from data, algorithms, or measurement processes. It often results in unequal model performance or treatment across sensitive groups like gender or race. To address this, bias mitigation techniques can be applied at various stages of the ML pipeline:

In pre-processing, data can be reweighted or rebalanced.

During in-processing, constraints can enforce fairness objectives.

In post-processing, outputs can be adjusted to meet equity metrics.

This example demonstrates how to detect and reduce demographic disparity using the Fairlearn library. The technique balances fairness (equal selection rates between genders) against model accuracy. Monitoring and mitigating bias is essential for developing responsible, trustworthy AI systems.


## Personalizing the Customer Journey Using Machine Learning

Personalizing the customer journey means tailoring content, recommendations, and interactions to individual users based on their preferences, behavior, and history. Machine learning (ML) plays a central role in achieving this by learning patterns from customer data and predicting next-best actions, products, or messages.

## Why Personalization Matters

- Increases customer engagement and satisfaction
- Boosts conversions and revenue
- Reduces churn by enhancing relevance
- Supports scalable automation for digital marketing and UX

## Key Stages of the Customer Journey for Personalization

1. **Awareness** – Personalized ads or blog content based on browsing history
2. **Consideration** – Dynamic recommendations, product comparisons
3. **Purchase** – Customized discounts, urgency triggers
4. **Retention** – Email targeting, upsell/cross-sell offers
5. **Loyalty** – Reward systems, anniversary campaigns

## Machine Learning Techniques for Personalization

| Technique             | Use Case                                      |
|----------------------|-----------------------------------------------|
| Collaborative Filtering | Product recommendations (based on user-user or item-item similarities) |
| Content-Based Filtering | Recommend similar products based on features |
| Clustering (e.g., KMeans) | Customer segmentation |
| Classification (e.g., XGBoost) | Churn prediction, email response prediction |
| Reinforcement Learning | Adaptive offers or UX flow optimization |

## Example: Personalized Product Recommendations Using Collaborative Filtering

We will implement a basic **collaborative filtering** model using matrix factorization (SVD) to recommend products to users.

### Dependencies

```python
!pip install scikit-surprise pandas
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Sample user-product interaction data
data_dict = {
    'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U3', 'U3'],
    'product_id': ['P1', 'P2', 'P3', 'P1', 'P4', 'P2', 'P3', 'P4', 'P5'],
    'rating': [5, 3, 4, 4, 2, 2, 5, 4, 3]
}

df = pd.DataFrame(data_dict)

# Define reader for Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# Train/test split
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use Singular Value Decomposition (SVD) for collaborative filtering
model = SVD()
model.fit(trainset)

# Predict and evaluate
predictions = model.test(testset)
print("RMSE:", rmse(predictions))

# Recommend top-N items for a given user
def get_top_n(predictions, n=3):
    from collections import defaultdict
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# Predict for all user-item pairs not in the training set
trainset_iids = set(iid for (_, iid, _) in trainset.all_ratings())
all_items = set(df['product_id'].unique())

testset_full = []
for user in df['user_id'].unique():
    for item in all_items:
        if not trainset.knows_item(trainset.to_inner_iid(item)) or not trainset.knows_user(trainset.to_inner_uid(user)):
            continue
        if item not in [iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user)]]:
            testset_full.append((user, item, 0))

predictions_full = model.test(testset_full)
top_n = get_top_n(predictions_full, n=2)

print("\nTop Recommendations:")
for user_id, recs in top_n.items():
    print(f"{user_id}: {[item for item, _ in recs]}")

```

## Interpreting the Results
The model predicts how much a user would like an unseen product based on their and others' past interactions.

We return the top-N recommendations for each user not already seen in the training data.

RMSE (root mean squared error) is used to evaluate the model’s prediction accuracy.

## Extensions for Real-World Use
Integrate with real-time streaming data pipelines.

Combine collaborative and content-based approaches for hybrid recommendation systems.

Include contextual factors like time of day, device, or location.

Use customer segmentation (e.g., KMeans) to personalize email campaigns or landing pages.

Apply reinforcement learning for adaptive personalization based on rewards (clicks, purchases).

## Summary
Personalizing the customer journey with machine learning helps businesses tailor interactions to individual needs, driving better user experience and business outcomes. Techniques like collaborative filtering enable product recommendations based on user behavior, while classification and clustering assist with customer segmentation and churn prediction. In production, these models can be integrated with CRM systems, websites, or marketing platforms to provide dynamic, personalized experiences at scale.



## Inventory Management Using Machine Learning

Inventory management involves tracking stock levels, demand forecasting, reordering, and optimizing supply chain decisions. Traditional methods rely on rule-based systems or fixed models, but machine learning (ML) brings data-driven accuracy and adaptability.

## Why Use Machine Learning for Inventory Management?

- Reduces overstocking and understocking
- Improves demand forecasting accuracy
- Minimizes holding and shortage costs
- Adapts to seasonality, trends, and external factors
- Enables real-time and dynamic decision making

## Core ML Use Cases in Inventory Management

| ML Task              | Use Case                                             |
|----------------------|------------------------------------------------------|
| Supervised Regression | Forecast product demand for future dates            |
| Classification        | Predict stock-out risk or order delay               |
| Time Series Forecasting | Capture seasonality and trends in item sales      |
| Clustering            | Group products or stores by demand patterns         |
| Reinforcement Learning | Optimize ordering policies under uncertainty       |

---

## Example: Demand Forecasting with Time Series Regression

This example uses historical sales data to forecast future demand for a single product using a supervised learning model (Random Forest).

### Dependencies

```python
!pip install pandas scikit-learn matplotlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Simulated daily sales data
np.random.seed(42)
days = pd.date_range(start="2022-01-01", end="2023-12-31")
data = pd.DataFrame({
    "date": days,
    "sales": (20 + 5 * np.sin(np.linspace(0, 20, len(days))) +
              np.random.normal(0, 2, len(days))).round().astype(int)
})

# Feature engineering
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['day_of_month'] = data['date'].dt.day
data['lag_1'] = data['sales'].shift(1)
data['lag_7'] = data['sales'].shift(7)
data['rolling_mean_7'] = data['sales'].rolling(window=7).mean()
data = data.dropna()

# Train/test split
features = ['day_of_week', 'month', 'day_of_month', 'lag_1', 'lag_7', 'rolling_mean_7']
X = data[features]
y = data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", round(rmse, 2))

# Plot actual vs predicted
plt.figure(figsize=(12,5))
plt.plot(data['date'][-len(y_test):], y_test.values, label="Actual")
plt.plot(data['date'][-len(y_test):], y_pred, label="Predicted")
plt.title("Demand Forecasting (Random Forest)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()

```

## Interpretation
The model uses lag features and calendar-based features to learn from past sales patterns.

RandomForestRegressor handles non-linearity and interactions well without much preprocessing.

The RMSE metric evaluates the prediction error on the test set.

## Real-World Extensions
Use multi-variate features like promotions, weather, pricing, holidays.

Model at the SKU + location level for granular predictions.

Integrate with inventory replenishment systems for automated ordering.

Apply probabilistic forecasting (e.g., quantile regression) for safety stock estimation.

Use deep learning (e.g., LSTM or Temporal Fusion Transformer) for high-volume, multi-SKU forecasting.

## Summary
Machine learning enhances inventory management by accurately forecasting demand, detecting anomalies, and optimizing replenishment. In this example, a Random Forest model predicted future sales using lagged and calendar features. With enough historical data and contextual signals, ML-based inventory systems can significantly reduce stockouts, cut excess inventory, and improve supply chain efficiency.


## Managing Clinical Trials Using Machine Learning

Clinical trials are essential to validating the safety and efficacy of new drugs, devices, or therapies. Managing them involves complex planning, patient recruitment, monitoring, and data analysis. Machine learning (ML) introduces intelligent automation and predictive insights to improve trial design, execution, and outcomes.

---

## Why Use Machine Learning in Clinical Trials?

| Area                    | ML Contribution                                                |
|-------------------------|----------------------------------------------------------------|
| Patient Recruitment      | Predict eligible participants based on EHR data                |
| Risk Monitoring          | Detect protocol deviations or adverse events early             |
| Outcome Prediction       | Forecast trial outcomes from baseline features                 |
| Site Selection           | Identify high-performing sites based on historical performance |
| Dropout Prediction       | Anticipate which patients may leave the study prematurely      |

---

## Key ML Techniques for Clinical Trial Management

| Technique                 | Application                                    |
|--------------------------|-------------------------------------------------|
| Logistic Regression       | Dropout prediction, inclusion/exclusion checks |
| Random Forest             | Risk modeling, outcome classification          |
| Clustering (e.g., KMeans) | Patient subgroup identification                |
| NLP (Transformer models)  | Extract info from clinical notes and reports   |
| Time Series Modeling      | Adverse event forecasting                      |

---

## Example: Predicting Patient Dropout in a Clinical Trial

This example uses synthetic data to train a classification model that predicts whether a participant will drop out of a clinical trial based on baseline characteristics.

### Dependencies

```python
!pip install pandas scikit-learn matplotlib seaborn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate clinical trial participant data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'age': np.random.randint(18, 85, n),
    'bmi': np.random.normal(27, 5, n).round(1),
    'chronic_conditions': np.random.randint(0, 4, n),
    'prior_trials': np.random.randint(0, 2, n),
    'compliance_score': np.random.uniform(0, 1, n),
    'distance_to_site_km': np.random.normal(20, 10, n).round(1),
    'dropout': np.random.choice([0, 1], size=n, p=[0.8, 0.2])  # 0 = stayed, 1 = dropped out
})

# Prepare features and target
X = data.drop('dropout', axis=1)
y = data['dropout']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Dropped"], yticklabels=["Stayed", "Dropped"])
plt.title("Dropout Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', title='Feature Importance')
plt.tight_layout()
plt.show()
```

## Interpreting the Results
Classification Report gives precision, recall, F1-score — useful for assessing how well the model predicts patient dropout.

Confusion Matrix shows true positives, false positives, and so on.

Feature Importance shows which attributes most influence dropout — this insight can guide clinical staff to improve retention (e.g., addressing travel burden or low compliance scores).


## Real-World Applications
Automated Screening: Use ML to filter eligible patients by scanning EHR data.

Adaptive Trial Design: Adjust dosage or enrollment dynamically based on model feedback.

Risk-Based Monitoring: Prioritize monitoring resources to high-risk sites or participants.

Natural Language Processing (NLP): Extract critical endpoints or adverse events from unstructured text.

Synthetic Control Arms: Reduce control group size by augmenting with historical patient data modeled using ML.

## Regulatory and Ethical Considerations
Ensure models are interpretable and clinically validated.

Maintain compliance with GCP, HIPAA, and GDPR.

Bias and fairness must be evaluated, especially in sensitive patient inclusion/exclusion decisions.

Explainability (e.g., SHAP, LIME) is crucial for clinician trust and regulatory approval.

## Summary
Machine learning can streamline clinical trial management by improving recruitment, forecasting outcomes, and reducing risk. In this example, a Random Forest model predicted dropout risk using patient attributes, helping sponsors and clinicians proactively retain participants. With proper validation and ethical safeguards, ML can significantly accelerate and de-risk clinical research.


## Hospital Readmission Management Using Machine Learning

## Introduction

Hospital readmission refers to a patient being admitted again to a hospital within a specific time period after discharge. Reducing unnecessary readmissions is critical for improving patient outcomes and reducing healthcare costs.

Machine learning (ML) can be used to predict which patients are at high risk of being readmitted, allowing hospitals to intervene proactively.



## Use Cases

### 1. Readmission Risk Prediction
- Predict if a patient is likely to be readmitted within 30 days

### 2. Resource Allocation
- Prioritize follow-up and care coordination for high-risk patients

### 3. Treatment Optimization
- Analyze historical data to suggest better care pathways



## Common Features Used

- Age
- Diagnosis codes (ICD-10)
- Length of stay
- Number of prior admissions
- Comorbidities (e.g., diabetes, hypertension)
- Medication count
- Lab test results
- Discharge type (home, rehab, skilled nursing facility)


## Sample Project: Predicting Readmission Risk

**Goal**: Build a binary classification model to predict 30-day readmission.


##  1: Import Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```






##  2: Example Dataset

```python
# Simulated dataset
data = {
    'age': [65, 50, 70, 45, 80, 60, 55],
    'length_of_stay': [5, 2, 10, 3, 7, 4, 6],
    'prior_admissions': [2, 0, 4, 1, 5, 3, 2],
    'num_medications': [10, 5, 15, 6, 13, 8, 9],
    'has_diabetes': [1, 0, 1, 0, 1, 1, 0],
    'was_readmitted': [1, 0, 1, 0, 1, 0, 0]  # 1 = readmitted within 30 days
}

df = pd.DataFrame(data)

X = df.drop('was_readmitted', axis=1)
y = df['was_readmitted']
```

## 3: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 4: Train the Model
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## 5: Evaluate the Model
```python
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 6: Predict New Patient Readmission Risk
```python
# New patient example
new_patient = pd.DataFrame([{
    'age': 72,
    'length_of_stay': 6,
    'prior_admissions': 3,
    'num_medications': 12,
    'has_diabetes': 1
}])

prediction = model.predict(new_patient)[0]
print("Predicted Readmission Risk (1 = Yes, 0 = No):", prediction)
```


## Summary

Machine learning provides a data-driven approach to reduce avoidable readmissions by:

Identifying high-risk patients

Improving discharge planning

Allocating care management resources efficiently

Common models used include logistic regression, decision trees, random forests, gradient boosting machines, and neural networks. Data quality, feature engineering, and domain knowledge play key roles in building accurate models.


## Disease Management Using Machine Learning

## Introduction

Disease management refers to the coordinated efforts to prevent, monitor, and treat chronic and acute medical conditions. Machine learning (ML) enhances disease management by enabling predictive analytics, personalized treatment, early diagnosis, and automated monitoring, resulting in better patient outcomes and reduced healthcare costs.


## Use Cases of Machine Learning in Disease Management

### 1. Early Diagnosis
- Detect diseases like diabetes, cancer, and heart disease based on symptoms, lab results, or imaging.

### 2. Risk Stratification
- Classify patients by the likelihood of disease progression or complications.

### 3. Treatment Personalization
- Recommend personalized medication or care plans based on medical history and response data.

### 4. Disease Progression Forecasting
- Predict how quickly a condition will worsen using time-series or longitudinal data.

### 5. Remote Monitoring and Alerts
- Analyze sensor or wearable data to detect abnormal health patterns in real-time.

---

## Sample Use Case: Predicting Diabetes Based on Clinical Features

**Objective**: Use a classification model to predict whether a patient has diabetes.


##  1: Import Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
```

## 2: Create Example Dataset

```python
# Sample simulated dataset
data = {
    'age': [50, 35, 60, 45, 25, 55, 40],
    'bmi': [30.1, 24.5, 35.0, 28.0, 22.0, 33.5, 26.5],
    'blood_pressure': [140, 120, 150, 130, 115, 145, 125],
    'glucose_level': [180, 100, 200, 160, 90, 170, 130],
    'has_diabetes': [1, 0, 1, 1, 0, 1, 0]  # 1 = diabetic, 0 = non-diabetic
}

df = pd.DataFrame(data)

X = df.drop('has_diabetes', axis=1)
y = df['has_diabetes']
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

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))
```

## 6: Predict for New Patient

```python

new_patient = pd.DataFrame([{
    'age': 48,
    'bmi': 29.0,
    'blood_pressure': 135,
    'glucose_level': 150
}])

prediction = model.predict(new_patient)[0]
print("Predicted Diabetes Status (1 = Yes, 0 = No):", prediction)
```
## Summary
Machine learning improves disease management by:

Enabling early detection and timely interventions

Supporting physicians with data-driven decisions

Automating continuous monitoring and alerting

Enhancing personalization of care plans

Popular models include logistic regression, decision trees, random forests, support vector machines, and neural networks. Effective use of ML requires clean, well-labeled healthcare data, along with clinical validation.

## Forest Management Using Machine Learning

## Introduction

Forest management involves planning and implementing practices for the conservation, restoration, and sustainable use of forest ecosystems. Machine learning (ML) supports data-driven decision-making in forest management by analyzing satellite imagery, ecological metrics, and environmental sensor data to improve monitoring, prediction, and planning.


## Applications of Machine Learning in Forest Management

### 1. Deforestation Detection
- Use satellite imagery to detect illegal logging and forest degradation.

### 2. Forest Fire Prediction
- Predict the likelihood of wildfires based on temperature, humidity, wind, and historical fire data.

### 3. Tree Species Classification
- Classify different tree species using image data from drones or satellites.

### 4. Biomass and Carbon Stock Estimation
- Estimate forest biomass and carbon content using regression and remote sensing.

### 5. Habitat and Biodiversity Mapping
- Analyze spatial and ecological data to track animal populations and species diversity.


## Sample Use Case: Predicting Forest Fire Risk

**Objective**: Use a classification model to predict whether an area is at risk of wildfire based on environmental features.


## Step 1: Import Required Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
```

## 2: Simulated Dataset
```python
# Example forest fire risk dataset
data = {
    'temperature': [30, 22, 35, 28, 20, 40, 33],
    'humidity': [45, 80, 30, 60, 85, 25, 40],
    'wind_speed': [12, 5, 15, 10, 6, 18, 11],
    'rainfall': [0.0, 10.2, 0.0, 2.5, 15.0, 0.0, 0.8],
    'fire_risk': [1, 0, 1, 0, 0, 1, 1]  # 1 = high fire risk, 0 = low fire risk
}

df = pd.DataFrame(data)

X = df.drop('fire_risk', axis=1)
y = df['fire_risk']
```

## 3: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 4: Train the Model

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

## 5: Evaluate the Model

```python
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 6: Predict Fire Risk for New Environmental Conditions

```python
new_area = pd.DataFrame([{
    'temperature': 34,
    'humidity': 35,
    'wind_speed': 14,
    'rainfall': 0.0
}])

prediction = model.predict(new_area)[0]
print("Predicted Fire Risk (1 = High, 0 = Low):", prediction)
```


