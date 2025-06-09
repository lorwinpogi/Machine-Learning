

## Accuracy
In classification, **accuracy** is the number of correctly classified items divided by the total number of items in the test set. It ranges from 0 (least accurate) to 1 (most accurate). Accuracy is one of the evaluation metrics of model performance. It should be considered alongside **precision**, **recall**, and **F-score** for a fuller understanding of model quality.

## Area Under the Curve (AUC)
In binary classification, **AUC** represents the value under the curve that plots the true positive rate (y-axis) against the false positive rate (x-axis). It ranges from 0.5 (worst) to 1 (best). Also known as the **area under the ROC curve** or **receiver operating characteristic curve**.

## Binary Classification
A classification task where the label is one of two possible classes (e.g., "spam" or "not spam").

## Calibration
**Calibration** is the process of mapping a raw score onto a class membership for binary and multiclass classification. Some trainers have a `NonCalibrated` suffix, indicating they produce raw scores that must be mapped to probabilities.

## Catalog
A **catalog** is a collection of extension methods grouped by task. For example, the `BinaryClassificationCatalog.BinaryClassificationTrainers` lists all available binary classification trainers.

## Classification
A **classification** task predicts a category or class label.  
- **Binary classification**: Two possible categories  
- **Multiclass classification**: More than two categories  

## Coefficient of Determination (R²)
Used in regression tasks to measure how well the data fits a model. Ranges from 0 (no fit) to 1 (perfect fit). Also known as **R-squared** or **R2**.

## Data
Data is central to ML applications. Data is represented by `IDataView` objects which:
- Consist of columns and rows
- Are lazily evaluated (loaded only when needed)
- Include a schema defining each column's type, format, and length

## Estimator
A class that implements the `IEstimator<TTransformer>` interface. An **estimator** defines a transformation (data preparation or model training). Estimators are chained into pipelines and produce **transformers** after training (`Fit`).

## Extension Method
A .NET method defined outside a class but made part of it via the `this` keyword. Extension methods to simplify creation of estimators.

## Feature
A **feature** is a measurable attribute or characteristic used as input for a model. Features are typically numeric and stored in a **feature vector** (e.g., `double[]`).

## Feature Engineering
**Feature engineering** involves designing and extracting useful features from raw data. This can significantly affect model performance.

## F-score
In classification, the **F-score** is the harmonic mean of **precision** and **recall**, offering a single metric to evaluate both.

## Hyperparameter
A **hyperparameter** is a setting of a machine learning algorithm (e.g., number of trees, learning rate). Set before training, it governs how the model learns.

## Label
The **label** is the value a model is trained to predict. For instance, the breed of a dog or the price of a house.

## Log Loss
In classification, **log loss** measures the accuracy of a classifier by penalizing false classifications. Lower log loss indicates better performance.

## Loss Function
A **loss function** measures the difference between actual label values and model predictions. Models are trained by minimizing this function.

## Mean Absolute Error (MAE)
In regression, **MAE** is the average of all absolute errors (differences between predicted and actual values).

## Model
A **model** consists of the parameters and featurization steps used to predict outcomes. A model includes both the transformation pipeline and the trained prediction logic.

## Multiclass Classification
A classification task where the label can be one of three or more categories.

## N-gram
An **N-gram** is a contiguous sequence of `N` words from a text used as a feature (common in text classification tasks).

## Normalization
**Normalization** scales numeric data to a specific range, typically [0, 1]. It improves model training and performance.

## Numerical Feature Vector
A **numerical feature vector** is a vector composed only of numeric values (e.g., `double[]`) used as input to models.

## Pipeline
A **pipeline** includes all steps from data loading and transformation to model training. After training, the pipeline becomes a model that can make predictions.

## Precision
In classification, **precision** is the number of true positives divided by the total number of predicted positives.

## Recall
In classification, **recall** is the number of true positives divided by the total number of actual positives.

## Regularization
**Regularization** reduces model complexity and helps prevent overfitting:
- L1 regularization (Lasso): Encourages sparsity, may zero out weights.
- L2 regularization (Ridge): Minimizes the range of weights, reducing sensitivity to outliers.

## Regression
A **regression** task predicts a real-valued output (e.g., house prices, temperatures).

## Relative Absolute Error
In regression, **relative absolute error** is the sum of absolute errors divided by the sum of distances from the actual values to their mean.

## Relative Squared Error
In regression, **relative squared error** is the sum of squared errors divided by the total squared variation from the actual values to their mean.

## Root Mean Squared Error (RMSE)
**RMSE** is the square root of the average of squared prediction errors. It penalizes larger errors more heavily than MAE.

## Scoring
**Scoring**, also known as **inferencing**, is the process of using a trained model to generate predictions on new data.

## Supervised Machine Learning
A type of machine learning where the model is trained on labeled data. Tasks include classification and regression.

## Training
The process of learning model parameters (e.g., weights, tree splits) from a dataset.

## Transformer
A **transformer** implements the `ITransformer` interface. It transforms one `IDataView` into another, often as the result of training an estimator.

## Unsupervised Machine Learning
A type of machine learning where the model learns patterns or structures in unlabeled data. Common tasks include clustering and dimensionality reduction.

# LinearSVC Classifier 

`LinearSVC` is a linear Support Vector Machine (SVM) classifier implemented in the scikit-learn library. It is designed for classification tasks where the decision boundary between classes is linear. Unlike the general `SVC` classifier with a linear kernel, which uses the libsvm solver, `LinearSVC` uses the liblinear solver optimized for large datasets and high-dimensional feature spaces.

## Key Features

- **Linear classifier:** Finds a hyperplane that best separates classes in the feature space.
- **Efficient for large datasets:** Especially suited for problems with a large number of samples and features.
- **Supports both binary and multiclass classification:** Uses a one-vs-rest scheme for multiclass tasks.
- **Regularization:** Supports L1 and L2 regularization to prevent overfitting.
- **Loss functions:** Supports hinge loss (standard SVM loss) and squared hinge loss.

## How LinearSVC Works

LinearSVC tries to find the optimal hyperplane that maximizes the margin between classes. It does this by minimizing a regularized loss function:

- The loss function measures how many samples are misclassified or within the margin.
- Regularization controls model complexity to avoid overfitting.

The model solves this optimization using a coordinate descent algorithm, which is faster for linear problems compared to the quadratic programming solver used by `SVC`.

## Parameters

Some important parameters include:

- `penalty`: Type of regularization ('l1' or 'l2').
- `loss`: Specifies the loss function ('hinge' or 'squared_hinge').
- `C`: Inverse of regularization strength (smaller values specify stronger regularization).
- `max_iter`: Maximum number of iterations for the solver.
- `dual`: Whether to solve the dual or primal optimization problem (typically `dual=False` when `penalty='l1'`).

## Advantages

- Faster training and prediction on large datasets compared to `SVC` with a linear kernel.
- Memory efficient for high-dimensional data.
- Good default choice for linear classification problems.

## Limitations

- Only supports linear decision boundaries.
- Less flexible than kernelized SVMs for complex datasets.
- Sensitive to feature scaling; input features should typically be normalized or standardized.

## Sample Code:
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearSVC()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## Summary

`LinearSVC` is a powerful and efficient linear classifier suitable for large-scale classification problems where a linear decision boundary suffices. It is widely used in text classification, bioinformatics, and other domains involving high-dimensional data.

# K-Neighbors Classifier

The **K-Neighbors Classifier** (K-Nearest Neighbors, or KNN) is a simple, intuitive, and widely used supervised machine learning algorithm used for classification tasks. It belongs to the family of instance-based learning or lazy learning methods.

- The algorithm classifies a new data point based on the majority class among its **k** nearest neighbors in the feature space.
- The “distance” between data points is typically measured using metrics like Euclidean distance, Manhattan distance, or Minkowski distance.
- The value of **k** (number of neighbors) is a key hyperparameter that influences the model’s performance:
  - A small **k** makes the classifier sensitive to noise (overfitting).
  - A large **k** makes it more robust but might smooth over boundaries (underfitting).

## Key Characteristics

- **Lazy Learning:** KNN does not learn a model explicitly during training. Instead, it stores the training data and performs classification only when a prediction is requested.
- **Non-parametric:** It makes no assumptions about the underlying data distribution.
- **Versatile:** Can be used for both classification and regression (K-Neighbors Regressor).
- **Simple to Implement:** Requires no training phase except storing the data.

## Advantages

- Easy to understand and implement.
- Naturally handles multi-class classification.
- Flexible with different distance metrics and weighting schemes.

## Disadvantages

- Computationally expensive during prediction because it needs to compute distances to all training points.
- Performance can degrade with high-dimensional data (curse of dimensionality).
- Sensitive to irrelevant or noisy features unless feature selection or dimensionality reduction is applied.

## Typical Use Cases

- Pattern recognition
- Recommendation systems
- Image classification
- Anomaly detection

## Sample Code:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

## Summary

K-Neighbors Classifier predicts the label of an input sample by looking at the labels of its closest neighbors, making it a straightforward and effective classification method for many practical problems.

# Support Vector Classifier (SVC)

Support Vector Classifier (SVC) is a supervised machine learning algorithm used for classification tasks. It is based on the Support Vector Machine (SVM) concept, which aims to find the optimal hyperplane that best separates data points of different classes in a feature space.

SVC tries to find a decision boundary (hyperplane) that maximizes the margin between different classes. The margin is defined as the distance between the closest points (called support vectors) of each class to the hyperplane. Maximizing this margin helps improve the model's generalization ability on unseen data.

### Linear vs Non-Linear SVC

- **Linear SVC:** When the data is linearly separable, SVC finds a straight line (in 2D) or a flat hyperplane (in higher dimensions) that separates classes.
- **Non-Linear SVC:** For non-linearly separable data, SVC uses kernel functions (such as the Radial Basis Function (RBF), polynomial, or sigmoid kernels) to implicitly map data into a higher-dimensional space where a linear separator can be found.

## Concepts

- **Support Vectors:** The critical data points closest to the decision boundary that influence the position and orientation of the hyperplane.
- **Kernel Trick:** A method that allows SVC to operate in high-dimensional spaces without explicitly computing the coordinates in that space. It computes inner products between the images of all pairs of data in the feature space.
- **Regularization Parameter (C):** Controls the trade-off between maximizing the margin and minimizing classification errors. A smaller C encourages a wider margin but allows more misclassifications, while a larger C aims for fewer misclassifications but may result in a smaller margin.

## Advantages

- Effective in high-dimensional spaces.
- Works well when the number of features is greater than the number of samples.
- Memory efficient since it uses a subset of training points (support vectors).

## Disadvantages

- Can be less effective on large datasets due to high training time.
- Choosing the right kernel and tuning parameters can be complex.
- Not suitable for very noisy datasets with overlapping classes.

## Usage

SVC is widely used for classification problems such as image recognition, text classification, bioinformatics, and more. It provides flexibility with kernels and regularization, making it a powerful tool for various applications.

## Sample Code:

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

# Ensemble Classifiers

Ensemble classifiers are machine learning models that combine multiple individual classifiers to improve overall prediction performance. Instead of relying on a single model, ensemble methods aggregate the predictions of several models to produce a more robust and accurate result.


The primary concept behind ensemble classifiers is that a group of weak learners can combine their predictions to form a stronger learner. By combining multiple models, ensemble methods reduce the risk of overfitting and improve generalization on unseen data.

Ensemble classifiers use various strategies to combine the base models, such as:

- **Bagging (Bootstrap Aggregating):** Builds multiple models from different subsets of the training data created by random sampling with replacement. Each model is trained independently, and the final prediction is made by averaging (for regression) or majority voting (for classification).  
  Example: Random Forest.

- **Boosting:** Trains models sequentially, each new model focusing more on the errors made by previous models. This iterative correction reduces bias and improves accuracy.  
  Examples: AdaBoost, Gradient Boosting Machines (GBM), XGBoost.

- **Stacking (Stacked Generalization):** Combines multiple base classifiers by training a meta-classifier on their outputs. The meta-classifier learns how to best combine the predictions of base models.

## Advantages of Ensemble Classifiers

- **Improved Accuracy:** By aggregating multiple models, ensembles typically achieve better predictive performance than individual models.
- **Reduced Overfitting:** Combining diverse models reduces the chance of overfitting to noise in the training data.
- **Robustness:** Ensemble methods are often more stable and less sensitive to the choice of a specific model or training set.

## Common Ensemble Classifiers

- **Random Forest:** An ensemble of decision trees trained with bagging and feature randomness to create uncorrelated trees.
- **AdaBoost:** Boosting method that adjusts weights of incorrectly classified samples to focus subsequent models on harder cases.
- **Gradient Boosting:** Builds models sequentially by optimizing a loss function, improving predictions stage by stage.
- **XGBoost:** An optimized and scalable implementation of gradient boosting with regularization for better performance.

## Use Ensemble Classifiers

Ensemble classifiers are especially useful when:

- You need high accuracy and robustness.
- The dataset is complex or noisy.
- Individual models perform moderately but differently.
- You want to reduce the risk of selecting a poorly performing model.

---

Ensemble classifiers have become a cornerstone of modern machine learning due to their ability to deliver state-of-the-art results across various tasks and datasets.



## Sample Code:
```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("AdaBoost Accuracy:", accuracy_score(y_test, ada_pred))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
```
