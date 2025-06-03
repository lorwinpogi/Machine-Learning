## Supervised Learning Overview

Supervised learning is a type of machine learning focused on predictive modeling, divided into two main categories: **regression** and **classification**. It involves learning from labeled data to make predictions about unseen data.

- **Regression** is used to predict continuous numerical values based on input features, as seen in techniques like **linear regression** and **curve fitting**.
- **Classification** involves predicting discrete class labels and is commonly used in tasks such as spam detection or image recognition.

## Classification vs. Regression

**Classification** in machine learning focuses on predicting **discrete labels** based on feature vectors. Examples include predicting whether a person will have an adverse reaction to a drug or determining expected course grades.

The classification process typically involves:

- Building a **distance matrix**
- Using **binary representation**
- Writing **Python code** to generate data tables

Code examples are often provided to support learning and serve as references for future research.

## Nearest Neighbor Classification

**Nearest neighbor classification** is a method that predicts the label of a new instance by finding the closest example in the training dataset. The core idea is that similar instances tend to have similar labels.

While the approach is simple and intuitive, such as classifying animals like zebras and pythons based on their nearest neighbors, it can result in **misclassifications**. Errors often occur when the training data includes poorly labeled or ambiguous examples, such as mistaking a handwritten **zero** for a **nine**.


## K Nearest Neighbors

**K Nearest Neighbors (KNN)** improves classification accuracy by considering multiple nearby data points rather than relying on just one. This method reduces the impact of **noisy or misclassified data** by using a majority vote among the *k* closest neighbors to determine the label of a new instance.

For example, with **k = 3**, if two neighbors belong to one class and one is an outlier, the majority vote leads to a more reliable classification. This approach offers greater **robustness and reliability** compared to the simpler nearest neighbor method.


## Choosing K in KNN

Selecting the appropriate value of **K** in K-Nearest Neighbors (KNN) is crucial, as an improper choice can lead to poor performance, particularly in **imbalanced datasets** where some classes dominate.

To find the optimal value of K, **cross-validation** is often used. This involves splitting the training data into subsets and testing various K values to identify which yields the best performance.

Despite its simplicity and ease of implementation, KNN can be **inefficient**. It requires storing the entire training dataset and performing comparisons for each prediction, resulting in **high memory usage** and **slower classification times**, especially with large datasets, since it does not build an internal model.

## Model Evaluation Metrics

Four key metrics commonly used in model evaluation are:

- **Sensitivity (Recall)**: Measures the proportion of actual positives correctly identified. It is especially important in **screening tests**—such as mammograms for breast cancer—where failing to detect a positive case can have serious consequences.
- **Specificity**: Measures the proportion of actual negatives correctly identified.
- **Positive Predictive Value (PPV)**: Indicates the likelihood that a positive prediction is correct.
- **Negative Predictive Value (NPV)**: Indicates the likelihood that a negative prediction is correct.

While **accuracy** is a useful metric, relying solely on it can be misleading, especially in **imbalanced datasets**. A deeper understanding of these metrics allows for more effective and context-aware model evaluation.

## Testing Classifiers

Careful **classifier testing** is essential, particularly in high-stakes applications, such as selecting **surgery applicants**—where errors can have serious consequences. Robust evaluation methods ensure reliability and generalisability of predictive models.

Two common testing methods include:

- **Leave-One-Out Cross-Validation (LOOCV)**: Ideal for **small datasets**, this method tests each data point individually while using the rest for training.
- **Repeated Random Subsampling**: Suitable for **larger datasets**, this approach repeatedly splits the data into random training and test sets to evaluate model performance across multiple runs.

Models such as **K-Nearest Neighbors (KNN)** and **logistic regression** are tested using these strategies, offering a **systematic approach** to performance evaluation while managing complexities like **parameter tuning** and **data variability**.

## Logistic Regression Introduction

**Logistic regression** is a commonly used machine learning algorithm for predicting the **probability of an event**, particularly in **binary classification** tasks. Unlike **linear regression**, which predicts continuous values, logistic regression outputs values between 0 and 1, representing the likelihood of a particular outcome.

The model works by computing **weights** for each feature to assess their correlation with the target variable. This approach avoids the nonsensical predictions (e.g., probabilities below 0 or above 1) that can occur with linear regression in classification contexts.

By applying an **optimization process** and the **logistic (sigmoid) function**, logistic regression effectively models the relationship between input features and binary outcomes.

## Building a Logistic Regression Model

To build a **logistic regression model**, the following steps are typically followed:

1. **Prepare the data**:
   - Create a list (or array) of **feature vectors** (input data).
   - Create a corresponding list of **labels** (target outputs).

2. **Initialize the model**:
   - Use `sklearn.linear_model.LogisticRegression` from the **scikit-learn** library.

3. **Fit the model**:
   - Call the `.fit()` method with the feature vectors and labels to train the model.
   - This process computes a set of **weights** for each feature, reflecting their contribution to the prediction.

Example (Python):
```python
from sklearn.linear_model import LogisticRegression

# Example feature vectors and labels
X = [[2.5, 1.3], [3.1, 2.2], [1.1, 0.9]]  # Features
y = [0, 1, 0]                            # Labels

# Create and train the model
model = LogisticRegression()
model.fit(X, y)
```

## List Comprehension in Python

**List comprehension** in Python provides a concise way to create new lists by evaluating an expression for each element in an existing iterable. It is commonly used for tasks such as **data transformation**, **filtering**, and **feature construction** in machine learning workflows.

### Basic Syntax:
```python
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]

```

## Logistic Regression Results

In logistic regression, once the **weights** are determined through training, feature vectors enable **rapid evaluation** of new instances. The prediction speed is **independent of the size of the training dataset**, which provides a significant advantage over methods like K-Nearest Neighbors (KNN).

While KNN requires comparing a new instance to many training examples, logistic regression makes decisions efficiently by applying the learned weights to feature vectors after solving the optimization problem.

## Interpreting Feature Weights

Logistic regression often outperforms other models partly because it assigns **weights** to features, offering valuable insights into their influence on the outcome.

For example, in the **Titanic dataset**:

- Being in a **higher class cabin** significantly **increases** the chance of survival.
- **Age** and being **male** are **negatively correlated** with survival.

However, caution is necessary when interpreting these weights, as **correlations between features** can affect their individual impact and may lead to misleading conclusions.


## Overview of Classification in Machine Learning

**Classification** is a type of supervised learning focused on predicting **discrete labels** (categories) associated with feature vectors, unlike regression, which predicts continuous real numbers.

Examples of classification tasks include:

- Predicting whether a person will have an adverse reaction to a drug
- Assigning grades such as A, B, C, D, etc.

Labels in classification are **finite** and can be either **binary** (two classes) or **multi-class** (more than two classes).



## Nearest Neighbor Classification

The simplest classification method is **Nearest Neighbor (NN)**. It involves **no learning phase**; instead, a new example is classified by assigning the label of its **closest training example** based on a distance metric.

**Example:** Classifying animals as reptiles or not by comparing their nearest neighbors using **binary features** and **distance matrices**.

However, NN is **sensitive to noise and outliers**, which can cause incorrect classifications if the nearest neighbor is noisy or mislabeled.

## K Nearest Neighbors (KNN)

KNN improves on Nearest Neighbor (NN) by considering the labels of the **k nearest neighbors** (usually an odd number) and using **majority voting** to classify. This reduces sensitivity to noise and outliers.

Choosing the right **k** is critical:
- A **too small k** can make the model sensitive to noise.
- A **too large k** may bias the classification toward the most common class, especially in **imbalanced datasets**.

**Cross-validation** is commonly used to select the optimal k by repeatedly splitting the training data and evaluating performance on held-out subsets.

While KNN training is fast (simply storing the data), prediction is slow and **memory-intensive**, as it requires distance comparisons with all training points for each classification.


## Testing Methods for Classifiers

- **Leave-One-Out Cross-Validation (LOOCV)**:  
  Used with **small datasets**. Train on all but one example, test on the excluded example, repeat for all instances, and average the results.

- **Repeated Random Subsampling**:  
  Suitable for **larger datasets**. Randomly split data into training and testing sets multiple times (e.g., 80/20 split), then average the results.


## Logistic Regression for Classification

Logistic regression predicts **probabilities** for discrete outcomes by modeling the **log-odds** of the target class. This approach avoids nonsensical predictions that can occur with linear regression, such as probabilities outside the [0, 1] range.

It computes **weights (coefficients)** for each feature that represent the strength and direction of correlation with the outcome:

- **Positive weight** → feature is positively correlated with the outcome.
- **Negative weight** → feature is negatively correlated with the outcome.

These weights are learned through an **optimization process** using training data, which involves a **logarithmic function** (hence the term "logistic").


## Practical Use of Logistic Regression

Implementation typically uses Python's `sklearn.linear_model.LogisticRegression` with these key methods:

- `.fit()` to train the model on feature vectors and labels.
- `.predict_proba()` to get predicted probabilities for test examples.

The classification threshold is usually set at **0.5** (i.e., probability > 0.5 → positive class) but can be adjusted based on specific application needs.

## Advantages of Logistic Regression

- Once trained, prediction is **fast** and **independent of training set size** because it only requires evaluating a weighted sum of features.
- Often **outperforms K-Nearest Neighbors (KNN)** due to its ability to assign different weights to variables, capturing subtle relationships in the data.
- Provides **interpretable insights** via feature weights, allowing understanding of which features influence predictions and how.



## Interpreting Feature Weights in Logistic Regression

Example from the Titanic dataset:

- Being in **first or second class** has a **positive correlation** with survival, reflecting socio-economic advantage.
- **Older age** negatively correlates with survival, indicating higher risk.
- Being **male** strongly negatively correlates with survival, consistent with historical evacuation priorities.

**Caution:** Correlated features can complicate the interpretation of weights, so they should be analyzed carefully.


## Python Programming Enhancements

- **List comprehension** offers a concise way to create lists from existing lists or ranges with optional filtering, improving code readability and compactness.  
  Example:  
  ```python
  [x*x for x in range(10) if x % 2 == 0]
```





