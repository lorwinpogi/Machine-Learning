# Classification Task in Logistic Regression

## What is a Classification Task?

A classification task is a supervised machine learning problem where the goal is to predict a discrete label (class) for a given input. The output variable is categorical in nature, such as:

- Spam vs Not Spam (Binary classification)
- Disease vs No Disease
- Classifying images as Dog, Cat, or Bird (Multiclass classification)

## Logistic Regression for Classification

Logistic Regression is a statistical model used for binary and multiclass classification problems. Despite the name "regression," it is widely used for classification tasks.

### How it Works:

- Logistic Regression predicts the **probability** that a given input belongs to a particular class.
- It uses the **sigmoid function** (for binary classification) to map any real-valued number into a range between 0 and 1.
  
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
  where \( z = w^T x + b \)

- If the predicted probability is greater than 0.5, the output is class 1; otherwise, it is class 0.

### Key Points:

- Works well for linearly separable data
- Output is interpretable as a probability
- Can be extended to multiclass problems using techniques like One-vs-Rest (OvR)

## Applications

- Email spam detection
- Customer churn prediction
- Disease diagnosis
- Credit scoring



# Logistic Model


The **logistic model** is a statistical model used to predict the probability of a binary outcome (i.e., two possible classes such as 0 or 1, true or false, yes or no). It is widely used in classification tasks in machine learning and statistics.

## Mathematical Representation

The logistic model estimates the probability that a given input \( x \) belongs to class 1 using the **sigmoid function**:

\[
P(y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}
\]

where:

- \( \sigma(z) \) is the **sigmoid function**
- \( z = w^T x + b \) (a linear combination of input features)
- \( w \) is the weight vector
- \( b \) is the bias term

## Characteristics

- Outputs a probability between **0 and 1**
- Uses a **threshold** (commonly 0.5) to make a binary decision
- Trained using **maximum likelihood estimation** or **gradient descent**

## Decision Rule

\[
\text{If } P(y = 1 \mid x) \geq 0.5, \text{ predict } y = 1; \text{ else predict } y = 0
\]

## Cases

- Medical diagnosis (disease vs no disease)
- Credit scoring (default vs non-default)
- Email classification (spam vs not spam)
- Marketing (buy vs not buy)


# Maximum Likelihood


**Maximum Likelihood Estimation (MLE)** is a method used to estimate the parameters of a statistical model. The idea is to find the parameter values that **maximize the likelihood** of the observed data under the model.

In the context of **logistic regression**, MLE is used to find the best weights \( w \) and bias \( b \) that make the observed labels most probable.

## Likelihood Function

Given a dataset of \( n \) samples:
\[
\{(x^{(i)}, y^{(i)})\}_{i=1}^{n}
\]

where \( x^{(i)} \) is the input vector and \( y^{(i)} \in \{0, 1\} \) is the binary label, the **likelihood** of the data under the logistic model is:

\[
L(w, b) = \prod_{i=1}^{n} P(y^{(i)} \mid x^{(i)}; w, b)
\]

For binary classification using the sigmoid function \( \sigma(z) \), this becomes:

\[
L(w, b) = \prod_{i=1}^{n} \sigma(z^{(i)})^{y^{(i)}} (1 - \sigma(z^{(i)}))^{1 - y^{(i)}}
\]

where \( z^{(i)} = w^T x^{(i)} + b \)

## Log-Likelihood

To simplify computation (especially when multiplying many small probabilities), we take the **logarithm** of the likelihood function to get the **log-likelihood**:

\[
\log L(w, b) = \sum_{i=1}^{n} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right]
\]

This is the **objective function** that logistic regression seeks to **maximize**.

## Optimization

- The parameters \( w \) and \( b \) are optimized using algorithms like **gradient ascent** (or **gradient descent** on the negative log-likelihood).
- This results in the best-fitting logistic model for the given training data.

## Summary

- **Goal:** Maximize the probability of observed data
- **Method:** Maximize the log-likelihood function
- **Used in:** Estimating parameters in logistic regression and many other models


# Convexity


In optimization, a function is said to be **convex** if the line segment between any two points on its graph lies **above or on** the graph itself.

Formally, a function \( f(x) \) is **convex** if for any \( x_1, x_2 \) in the domain and \( \lambda \in [0, 1] \):

\[
f(\lambda x_1 + (1 - \lambda)x_2) \leq \lambda f(x_1) + (1 - \lambda)f(x_2)
\]

## Convexity Matters

Convexity is important because:

- **Convex functions have a single global minimum.**
- **Optimization algorithms (like gradient descent) are guaranteed to converge** to the global minimum, not a local one.
- This makes training models easier and more reliable.

## Convexity in Logistic Regression

In logistic regression:

- We optimize the **negative log-likelihood** (also called the **log loss** or **cross-entropy loss**).
- This loss function is **convex** with respect to the model parameters \( w \) and \( b \).

### Therefore:
- There is **one unique solution** (global optimum).
- Optimization methods like **gradient descent** will reliably find the best model parameters.

## Visual Intuition

A convex function typically looks like a **U-shape** curve. Any local minimum is also the **global minimum**.

## Summary

- Convexity ensures stable, reliable training.
- The loss function in logistic regression is convex.
- Convexity guarantees that gradient-based optimization will succeed in finding the best parameters.


# Algorithms


An **algorithm** is a finite sequence of well-defined steps or instructions used to solve a specific problem or perform a computation. In machine learning, algorithms are used to **train models**, **make predictions**, and **optimize performance**.

---

## Algorithms in Logistic Regression

Logistic regression uses optimization algorithms to estimate the best-fit parameters (weights and bias) that minimize the **loss function** (usually the negative log-likelihood or cross-entropy loss).

### Common Algorithms Used:

#### 1. Gradient Descent

- **Goal**: Minimize the cost function by updating weights iteratively.
- **Steps**:
  - Compute the gradient (partial derivatives of the loss).
  - Update weights in the direction opposite to the gradient.
- **Update Rule**:
  \[
  w := w - \alpha \cdot \nabla J(w)
  \]
  where:
  - \( \alpha \) = learning rate
  - \( \nabla J(w) \) = gradient of the cost function

#### 2. Stochastic Gradient Descent (SGD)

- Updates weights using **one training example at a time**.
- Faster per update but more noisy.
- Useful for large datasets.

#### 3. Mini-Batch Gradient Descent

- A compromise between batch and stochastic methods.
- Updates weights using a small subset (mini-batch) of training data.
- Balances speed and stability.

#### 4. Newton's Method (Second-Order Optimization)

- Uses both the gradient and the **Hessian matrix** (second derivatives).
- Faster convergence but computationally more expensive.
- Less common in large-scale logistic regression.

#### 5. Quasi-Newton Methods (e.g., BFGS, L-BFGS)

- Approximate the Hessian instead of computing it exactly.
- More efficient for large problems than standard Newton’s Method.

---

## Summary

| Algorithm                    | Speed        | Accuracy     | Common Use Case                     |
|-----------------------------|--------------|--------------|-------------------------------------|
| Gradient Descent            | Moderate     | High         | Most standard logistic models       |
| Stochastic Gradient Descent | Fast per step| Lower (noisy)| Large datasets, online learning     |
| Mini-Batch GD               | Balanced     | High         | Deep learning, large data           |
| Newton's Method             | Fast conv.   | High         | Small to medium datasets            |
| L-BFGS                      | Efficient    | High         | Logistic regression in libraries    |

---

## Libraries That Implement These

- **Scikit-learn** (`solver='lbfgs'`, `'saga'`, `'liblinear'`)
- **TensorFlow / PyTorch** (for custom gradient-based logistic models)


#  Linear Models in Python Machine Learning

Linear models are a foundational concept in machine learning, used for both regression and classification tasks. They are simple, interpretable, and efficient, making them a great starting point for many problems.

---

##  What Is a Linear Model?

A linear model assumes a linear relationship between input variables (`X`) and the output (`y`). The model is represented as:

\[
y = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b
\]

- \( x_1, x_2, \ldots, x_n \): Input features  
- \( w_1, w_2, \ldots, w_n \): Weights (coefficients)  
- \( b \): Bias (intercept term)

Compact form:

\[
y = \mathbf{w}^\top \mathbf{x} + b
\]

---

##  Types of Linear Models

### 1. Linear Regression (for continuous output)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Example: X and y are your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
```
predictions = model.predict(X_test)
# Ridge Regression (L2 regularization)
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```
# Lasso Regression (L1 regularization)
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

# Logistic Regression (for classification)
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

# Evaluating Linear Models
# For Regression:

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MSE:", mse)
print("R^2 Score:", r2)
```




# For Classification:
```python
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

```

# Normalize or Standardize Data
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

```


# Use Polynomial Features for Non-Linear Relationships

```python
from sklearn.preprocessing import PolynomialFeatures

poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
```

#  Support Vector Machines (SVM) in Python Machine Learning

Support Vector Machines (SVMs) are powerful supervised learning models used for **classification**, **regression**, and **outlier detection**. They are particularly effective in high-dimensional spaces and when the number of dimensions exceeds the number of samples.

---

##  What Is a Support Vector Machine?

SVM aims to find the **optimal hyperplane** that best separates data into classes. The best hyperplane is the one with the **maximum margin** between the two classes.

For non-linearly separable data, SVM uses the **kernel trick** to map input features into a higher-dimensional space where a linear separator may exist.

---

## SVM Works

### 1. **Linear SVM**:
- Finds a hyperplane that separates the data with the largest margin.
- Uses support vectors (data points closest to the hyperplane).

### 2. **Non-Linear SVM**:
- Uses **kernel functions** to project data into higher dimensions:
  - Linear
  - Polynomial
  - Radial Basis Function (RBF)
  - Sigmoid

---

##  SVM with Scikit-learn

###  Importing and Basic Setup

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
```python

# Load Example Dataset

```python
# Load a toy dataset (e.g., Iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Only use two classes for binary classification example
X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

# Load Example Dataset
```python
# Load a toy dataset (e.g., Iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Only use two classes for binary classification example
X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

# SVM Classification Example

```python
# Create and train an SVM classifier
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```


# Common Kernels in SVM

Linear Kernel
```python 
model = SVC(kernel='linear')
```



Polynomial Kernel
```python 
model = SVC(kernel='poly', degree=3)
```
RBF Kernel (default and most common)
```python 
model = SVC(kernel='rbf', gamma='scale')
```
Sigmoid Kernel
```python 
model = SVC(kernel='sigmoid')
```

# Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
```

# SVM for Regression (SVR)

```python
from sklearn.svm import SVR

svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
```

# SVM for Outlier Detection
```python
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
oc_svm.fit(X_train)

# Predict: -1 for outliers, 1 for inliers
preds = oc_svm.predict(X_test)
```


#  Stochastic Gradient Descent (SGD) in Python Machine Learning

Stochastic Gradient Descent (SGD) is an optimization algorithm used to **minimize a loss function** by updating model parameters using a small batch (or a single sample) at a time. It is widely used for training linear models and neural networks due to its speed and efficiency.

---

##  Gradient Descent

Gradient Descent is an optimization algorithm that updates model weights by minimizing the loss function:

\[
\theta = \theta - \eta \cdot \nabla J(\theta)
\]

- \( \theta \): model parameters  
- \( \eta \): learning rate  
- \( \nabla J(\theta) \): gradient of the cost function

---

##   Stochastic Gradient Descent

Instead of computing the gradient over the entire dataset (as in **Batch Gradient Descent**), SGD computes the gradient using **just one sample (or a mini-batch)** at a time, resulting in:

- Faster updates
- Noisy gradients (which helps escape local minima)
- Often better generalization

---

##  Using SGD in Scikit-learn

Scikit-learn provides `SGDClassifier` and `SGDRegressor` for linear models trained with SGD.

```python
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
```

# SGD for Classification
```python
# Generate classification data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the classifier
clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

```


# SGD for Regression
```python
# Generate regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the regressor
reg = SGDRegressor(loss='squared_error', max_iter=1000, tol=1e-3)
reg.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_squared_error
preds = reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, preds))
```


# Learning Rate Strategies


```python
SGDClassifier(learning_rate='constant', eta0=0.01)
SGDClassifier(learning_rate='adaptive', eta0=0.01)
```



# Feature Scaling (Important!)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), SGDClassifier())
pipeline.fit(X_train, y_train)
```


#  Nearest Neighbors in Python Machine Learning

The **Nearest Neighbors** algorithm is a **non-parametric**, **lazy learning** method used for **classification**, **regression**, and **unsupervised learning**. It works by finding the most similar instances (neighbors) to a new data point and using them for prediction.

---

##  K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** is a simple algorithm that:
- Stores all available data
- Predicts the output by measuring distance to the `k` nearest data points

For classification, it uses **majority voting**.
For regression, it returns the **mean or median** of neighbors' values.

---

##  Key Concepts

- **Lazy learner**: No training phase (just memory of dataset)
- **Distance-based**: Common distances:
  - Euclidean (default)
  - Manhattan
  - Minkowski
- **k-value**: Number of neighbors to consider (commonly odd, like 3 or 5)

---

##  KNN with Scikit-learn

###  Import Required Libraries

```python
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
```

# KNN for Classification
```python
# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

# KNN for Regression
```python
# Generate synthetic regression data
X, y = make_regression(n_samples=500, n_features=3, noise=5.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
# Create model
```python
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)
```
# Predict and evaluate
```python
y_pred = knn_reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
```



## Distance Metrics

```python
KNeighborsClassifier(metric='manhattan')  # L1
KNeighborsClassifier(metric='euclidean')  # L2 (default)
KNeighborsClassifier(metric='minkowski', p=3)
```


## Scaling Features

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
pipeline.fit(X_train, y_train)
```


# Gaussian Processes in Python Machine Learning

Gaussian Processes (GPs) are a non-parametric, Bayesian approach to regression and probabilistic modeling. They are particularly useful when:

- You have a small-to-medium-sized dataset
- You want a prediction **with uncertainty**
- The underlying function is complex and potentially non-linear


A **Gaussian Process** is a collection of random variables, any finite number of which have a joint **Gaussian distribution**. You can think of it as a distribution **over functions**.

### Formal Definition

A GP is fully specified by:

- A **mean function**:  
  \[
  m(x) = \mathbb{E}[f(x)]
  \]

- A **covariance function** (kernel):  
  \[
  k(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]
  \]

The GP is written as:

\[
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
\]

---

##  Gaussian Process Regression (GPR) in Python

Use `scikit-learn` for Gaussian Process Regression.

###  Installation

```bash
pip install scikit-learn matplotlib numpy
```


## Sample Code:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Training data
X = np.atleast_2d(np.linspace(0, 10, 10)).T
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Add some noise

# Define kernel (RBF + noise term)
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# Initialize Gaussian Process Regressor
gp = GaussianProcessRegressor(kernel=kernel)

gp.fit(X, y)

# Predict at new points
X_pred = np.atleast_2d(np.linspace(0, 10, 100)).T
y_pred, sigma = gp.predict(X_pred, return_std=True)

plt.figure(figsize=(10, 6))
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(X_pred, y_pred, 'b-', label='Prediction')
plt.fill_between(X_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
                 alpha=0.2, label='95% Confidence Interval')
plt.title("Gaussian Process Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```




#  Decision Trees 

A **Decision Tree** is a supervised learning algorithm used for both **classification** and **regression** tasks. It models decisions and their possible consequences as a tree-like structure of branches.


A Decision Tree is a flowchart-like structure where:

- **Internal nodes** represent tests on features (e.g., "Is feature > 5?")
- **Leaves** represent predicted labels or values
- **Branches** represent decision rules

### Example:

Is Age < 30?
├── Yes: Recommend Product A
└── No: Recommend Product B

---

##  Advantages of Decision Trees

- Easy to understand and interpret
- Requires little data preprocessing
- Can handle both numerical and categorical data
- Performs well with large datasets

---

##  Disadvantages

- Prone to **overfitting**
- Unstable: small changes in data can result in different trees
- Biased towards features with more levels

---

## Decision Trees in Python (with `scikit-learn`)

### 1.  Imports

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```


# Ensemble Methods: Voting Classifier in Python

**Ensemble learning** combines predictions from multiple models to improve accuracy, robustness, and generalisation.

One of the simplest ensemble techniques is the **Voting Classifier**, which aggregates predictions from several base models and selects the most common one (for classification tasks).


A **Voting Classifier** combines multiple classification models (e.g., Logistic Regression, Decision Tree, SVM) and makes a final prediction based on **majority vote**.

There are two types:

### 1. **Hard Voting** (majority voting)

- Predicts the class that gets the **most votes** from the classifiers.

### 2. **Soft Voting** (average probabilities)

- Averages the **predicted probabilities** and chooses the class with the highest average probability.
- Requires classifiers with `predict_proba()` implemented.
- You want a **simple ensemble** method
- You have several **strong but different models**
- You aim to **reduce variance** or **improve generalization**

---

## ⚙️ Implementation in Python (`scikit-learn`)

### 1.  Imports

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_clf = LogisticRegression(max_iter=200)
tree_clf = DecisionTreeClassifier(max_depth=4)
svm_clf = SVC(probability=True)  # Required for soft voting

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('dt', tree_clf), ('svm', svm_clf)],
    voting='hard'
)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('dt', tree_clf), ('svm', svm_clf)],
    voting='soft'
)

voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred))

for clf in (log_clf, tree_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{clf.__class__.__name__} Accuracy: {accuracy_score(y_test, y_pred):.2f}")

```


#  Multiclass and Multioutput Algorithms in Python

In machine learning, **classification problems** aren't always simple binary tasks. Often, we deal with:

- **Multiclass classification**: One label, multiple possible classes
- **Multilabel classification**: Multiple labels per instance
- **Multiclass-multioutput classification**: Multiple classification tasks, each with multiple possible classes

---

## 1. Multiclass Classification

**Multiclass classification** refers to a problem where each sample is assigned exactly **one label from more than two classes**.

### Example:
Classifying digits (0–9) using MNIST.

###  In `scikit-learn`
Most classifiers (e.g., `LogisticRegression`, `SVC`, `RandomForestClassifier`) support multiclass out-of-the-box using **One-vs-Rest (OvR)** or **One-vs-One (OvO)** schemes.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```





