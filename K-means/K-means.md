# Silhouette Scoring

Silhouette Score is a **metric used to evaluate the quality of clustering** in unsupervised learning. It quantifies how well each data point lies within its cluster compared to other clusters.

### Formula

For a data point \( i \):
- \( a(i) \): Mean intra-cluster distance (average distance to other points in the same cluster)
- \( b(i) \): Mean nearest-cluster distance (lowest average distance to points in another cluster)

Silhouette score for a point is:

\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

- \( s(i) \) ranges from **-1 to 1**
  - Near **+1**: Point is well-matched to its own cluster, and poorly matched to others
  - Near **0**: Point is on the boundary of two clusters
  - Near **-1**: Point may have been assigned to the wrong cluster

## Use Silhouette Score

- **Internal validation**: Doesn't require ground truth labels
- Helps determine the **optimal number of clusters**
- Useful for comparing different clustering algorithms or hyperparameters

## Python Implementation

### 1. Import Libraries

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=4, random_state=0)
labels = kmeans.fit_predict(X)

score = silhouette_score(X, labels)
print(f'Silhouette Score: {score:.3f}')

sample_silhouette_values = silhouette_samples(X, labels)

fig, ax = plt.subplots()
y_lower = 10
for i in range(4):
    ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()
    
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.nipy_spectral(float(i) / 4)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_title("Silhouette plot for the clusters")
ax.set_xlabel("Silhouette coefficient values")
ax.set_ylabel("Cluster label")
ax.axvline(x=score, color="red", linestyle="--")
plt.show()

```


## Choosing Optimal Number of Clusters

```python
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f'Clusters: {n_clusters}, Silhouette Score: {score:.3f}')

```


# Elbow Method in Machine Learning (Python)


The **Elbow Method** is a technique used to determine the **optimal number of clusters (k)** in a **clustering algorithm** like **KMeans**.

It helps to find the value of `k` where adding more clusters **doesn't significantly improve the model**, based on the **inertia** (within-cluster sum of squares, or WCSS).

## Key Idea

- **Inertia / WCSS** measures how internally coherent the clusters are.
- As `k` increases, **inertia decreases** (clusters are smaller).
- The **"elbow" point** in the inertia vs. k plot represents the optimal number of clusters—after that, gains in reducing inertia diminish.

## Python Implementation

### 1. Import Libraries

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, 'bo-', linewidth=2, markersize=8)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia / WCSS')
plt.xticks(k_values)
plt.grid(True)
plt.show()

```

# Inertia 


Inertia is a **metric used to evaluate the quality of clustering** in **KMeans**.

Inertia is the **sum of squared distances** between each data point and the **centroid of the cluster** to which it belongs.

\[
\text{Inertia} = \sum_{i=1}^{k} \sum_{x_j \in C_i} \| x_j - \mu_i \|^2
\]

- \( C_i \): Cluster \( i \)
- \( \mu_i \): Centroid of cluster \( i \)
- \( x_j \): Data point in cluster \( i \)

### Interpretation

- **Lower inertia** means **tighter clusters** (better fit).
- **Higher inertia** means **looser clusters** (poor fit).
- Inertia **always decreases** as the number of clusters \( k \) increases.

However, a **very low inertia** with a high \( k \) might lead to **overfitting**, which is why we use the **Elbow Method** to find the optimal balance.

## Importance of Inertia?

- It helps **evaluate KMeans clustering**.
- It is used in the **Elbow Method** to find the optimal number of clusters.

## Python Example

### 1. Import Required Libraries

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
print(f'Inertia: {kmeans.inertia_:.2f}')

inertias = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method showing Inertia vs k')
plt.grid(True)
plt.show()
```

# Variance 


In statistics and machine learning, **variance** measures the **spread of data points** around their **mean**.

### Mathematical Definition

For a dataset with values \( x_1, x_2, ..., x_n \) and mean \( \mu \), the **variance** is:

\[
\text{Variance} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2
\]

- High variance = Data is **spread out**.
- Low variance = Data is **tightly clustered** around the mean.

---

## Variance in Machine Learning

In ML, variance has **two main roles**:

### 1. **Descriptive Statistic**

- Helps summarize the variability of feature values.
- Useful in **feature scaling**, **normalization**, and **PCA** (Principal Component Analysis).

### 2. **Model Evaluation**

- **Bias-Variance Tradeoff**:
  - **High variance model** overfits training data (poor generalization).
  - **Low variance model** may underfit.

\[
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
\]

---

## Python Examples

### 1. Variance of a 1D Dataset

```python
import numpy as np

data = [2, 4, 4, 4, 5, 5, 7, 9]
variance = np.var(data)
print(f'Variance: {variance:.2f}')

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Variance of each feature
print(df.var())

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=20)

model = DecisionTreeRegressor()
scores = cross_val_score(model, X, y, scoring='r2', cv=5)

print(f'CV Scores: {scores}')
print(f'Mean: {np.mean(scores):.3f}, Variance: {np.var(scores):.4f}')

```





