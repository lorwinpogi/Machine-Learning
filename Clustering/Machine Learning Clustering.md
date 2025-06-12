# Clustering in Machine Learning with Python

Clustering is a type of **unsupervised machine learning** technique that involves grouping data points such that those within the same group (called a **cluster**) are more similar to each other than to those in other groups.

Unlike supervised learning, clustering does **not use labels**. Instead, it identifies patterns, structures, or groupings in the data based solely on feature similarity.

## Use of Clustering

Clustering is used in many real-world applications, including:

- Customer segmentation in marketing
- Document or news article classification
- Social network analysis
- Image segmentation
- Anomaly or fraud detection
- Recommender systems

---

## Common Clustering Algorithms

### 1. **K-Means Clustering**

- Divides data into *k* non-overlapping subgroups.
- Minimizes the **intra-cluster variance** (distance between points and the centroid).
- Fast and widely used.

### 2. **Hierarchical Clustering**

- Builds a tree (dendrogram) of clusters.
- Two types: **Agglomerative** (bottom-up) and **Divisive** (top-down).
- No need to pre-specify the number of clusters.

### 3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

- Groups together points that are closely packed.
- Can find clusters of arbitrary shape.
- Identifies outliers as noise.

### 4. **Mean Shift**, **OPTICS**, **Spectral Clustering**, etc.

---

## K-Means Clustering Example in Python

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Create and fit KMeans model
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot the clusters and centroids
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X')  # Centroids
plt.title("K-Means Clustering Example")
plt.show()
```


# Supervised vs. Unsupervised Learning

In machine learning, supervised learning involves training a model on labelled data—each input is paired with an associated label. The model learns to infer patterns and generalise these relationships to unseen data. In contrast, unsupervised learning does not use labelled data. Instead, it seeks to uncover hidden structures or patterns in the data. A common technique in unsupervised learning is clustering, where the algorithm groups similar data points based on feature similarity.



# Clustering Concepts and Techniques

## Variability in Clustering

Clustering aims to identify natural groupings in data by analyzing feature vectors. One key concept is variability, which refers to the sum of distances between points within a cluster and the cluster's mean (centroid). Although this resembles variance, variability in clustering uniquely penalizes larger, more diverse clusters more severely than smaller ones. This is intentional, as it helps avoid the need for normalization and ensures that cluster size appropriately influences the overall grouping quality.

## Objective Function in Clustering

In clustering, optimizing the objective function requires more than simply minimizing dissimilarity. Without constraints, this could result in having as many clusters as there are data points, which would lead to zero intra-cluster dissimilarity—an undesirable outcome. To create meaningful clusters, constraints are applied, such as maintaining a minimum distance between clusters or limiting the number of clusters, to prevent overfitting and ensure practical grouping.



# Hierarchical Clustering

Hierarchical clustering starts by assigning each data point to its own cluster, resulting in as many clusters as data points. The algorithm then repeatedly merges the two most similar clusters, gradually reducing the number of clusters until all points belong to a single cluster. Usually, the merging process is halted before this final step, and a dendrogram is used to visualize the merging history. This tree-like structure helps determine an appropriate number of clusters by identifying natural divisions in the data.

## Linkage Metrics

In hierarchical clustering, the way cluster similarity is measured is defined by the linkage metric. Single linkage uses the shortest distance between any two points in two clusters. Complete linkage considers the farthest distance between any two points, and average linkage uses the mean distance between all pairs of points across clusters. The choice of linkage metric significantly affects the clustering result. Although hierarchical clustering provides a detailed history of cluster formation, it relies on a greedy algorithm. This means that local merging decisions may not lead to the best overall clustering structure, and the method can be computationally expensive for large datasets.



# K-Means Clustering

The K-Means algorithm is a widely used and efficient clustering method. It begins by randomly selecting K centroids. Each data point is then assigned to the nearest centroid, forming initial clusters. The centroids are updated by computing the mean of the points in each cluster, and the assignment step is repeated. This process continues iteratively until the centroids stabilize and the clusters no longer change. K-Means is computationally efficient and typically converges quickly. It works well when the number of clusters is known and the data is relatively well-separated. However, it may struggle when the cluster count is uncertain or when clusters vary in size and shape.

## Choosing K in K-Means

Selecting the appropriate number of clusters, K, is crucial in K-Means clustering. A poor choice can lead to misleading or meaningless groupings. The initial centroids also influence the final outcome so that different runs may produce different results. To improve robustness, it helps to incorporate domain knowledge, experiment with various K values, or use hierarchical clustering as a guide. Techniques such as the Elbow Method can also aid in identifying the optimal K by plotting the rate of decrease in within-cluster variability.



# Feature Scaling in Clustering

Before applying clustering algorithms, it is important to ensure that all features contribute equally. If features have vastly different scales, those with larger ranges can dominate the clustering process. To prevent this, standardization methods such as Z-scaling are applied to transform the data so that each feature has a mean of zero and a standard deviation of one. For example, features like weight can overpower binary variables like gender or glasses unless the data is scaled. Proper scaling often reveals meaningful clusters that were previously obscured, although challenges such as low sensitivity may still persist even if specificity improves.



# Evaluating Clustering Outcomes

Evaluating clustering results requires more than inspecting cluster shapes or sizes. For instance, a study of high-risk patients showed that while 83 patients were identified, only 26 belonged to a dense, meaningful cluster. When varying K across values like 2, 4, and 6, the clustering with K equal to 4 resulted in the highest concentration of high-risk individuals. This suggests that multiple underlying factors contribute to the classification of high-risk patients, and a more nuanced clustering strategy is necessary to effectively capture this complexity.

---

# Summary

| Concept                     | Key Idea                                                                 |
|----------------------------|--------------------------------------------------------------------------|
| Supervised Learning         | Uses labeled data to predict outcomes.                                  |
| Unsupervised Learning       | Finds structure in unlabeled data (e.g., clustering).                   |
| Variability in Clustering   | Penalizes large, diverse clusters more; avoids normalization.           |
| Objective Function          | Requires constraints to prevent trivial solutions (1 point = 1 cluster).|
| Hierarchical Clustering     | Builds nested clusters; visualized with dendrograms.                    |
| K-Means Clustering          | Iterative centroid update; needs predefined `K`.                        |
| Choosing `K`                | Can affect quality; try Elbow Method or use hierarchical for guidance.  |
| Feature Scaling             | Ensures fair influence of all features; improves clustering accuracy.   |
| Evaluation                  | Use real-world validation and metrics like Silhouette Score.            |



# Clustering Overview and Constraints

Clustering is an unsupervised learning task where feature vectors without labels are grouped into natural clusters to infer structure from data. The clustering problem is often formulated as minimizing dissimilarity, typically measured by the sum of squared distances from each point to its cluster centroid. To avoid trivial solutions, such as creating one cluster per point, constraints are applied—commonly by limiting the number of clusters (e.g., at most K clusters).

Choosing the number of clusters, **K**, is critical. This can be guided by domain knowledge or by evaluating clustering quality metrics over different K values. Hierarchical clustering on smaller data subsets is often used to help determine a suitable K for algorithms like K-Means.


# Hierarchical Clustering

Hierarchical clustering begins with each data point in its own cluster, resulting in n clusters. The algorithm then iteratively merges the two most similar clusters until all points are grouped into a single cluster. The process can be visualized using a dendrogram, which records the order and distances at which clusters are merged.

The outcome of hierarchical clustering depends heavily on the **linkage metric**, which defines the distance between clusters:

- **Single linkage**: minimum distance between any pair of points in different clusters  
- **Complete linkage**: maximum distance between any pair of points in different clusters  
- **Average linkage**: average distance between all pairs of points in different clusters

While hierarchical clustering is deterministic for a given linkage metric, it uses a greedy algorithm that may not yield globally optimal results. It is also computationally expensive—typically **O(n³)** time complexity, although optimizations reduce it to **O(n²)** for some linkage types. Despite this, hierarchical clustering allows exploration of data structure at multiple resolutions by cutting the dendrogram at different heights to obtain different numbers of clusters.


# K-Means Clustering

K-Means clustering is significantly faster than hierarchical methods and is well-suited for large datasets. However, it requires pre-specifying the number of clusters, K. The basic steps of the K-Means algorithm are:

1. Randomly select K initial centroids from the data.
2. Assign each point to the nearest centroid based on distance.
3. Recalculate the centroid of each cluster by averaging all assigned points.
4. Repeat the assignment and update steps until the centroids stabilize.

Centroids in K-Means may not correspond to actual data points since they are calculated as averages. The time complexity of one iteration is **O(K × n × d)**, where *n* is the number of points and *d* is the number of features. The algorithm usually converges quickly with only a few iterations needed.

K-Means is sensitive to the initial placement of centroids. Poor initialization can lead to suboptimal clustering or longer convergence times. Common strategies to address this include:

- Spreading initial centroids widely across the feature space
- Running the algorithm multiple times with different initializations and selecting the best result based on minimum dissimilarity
- Using smarter initialization techniques such as **K-Means++**

It is also possible for K-Means to produce empty clusters if no points are assigned to a centroid. This can be mitigated by reinitializing the centroid or redistributing points.


# Variability, Dissimilarity, and Objective Function

The **variability** of a cluster is defined as the sum of squared distances between each point and the cluster’s centroid. The **dissimilarity** of an entire clustering is the sum of variabilities across all clusters. This metric is not normalized by cluster size, which intentionally penalizes large, diverse clusters more heavily than small, dense ones. Normalizing by cluster size would treat clusters equally, regardless of how spread out the points are—something undesirable in many practical clustering scenarios.

The primary optimization goal in clustering is to minimize total dissimilarity subject to constraints, such as a fixed number of clusters (K). These constraints are necessary to avoid trivial solutions like assigning each point to its own cluster, which results in zero variability but offers no insight.

---

# Feature Scaling and Its Importance

Clustering algorithms rely on distance calculations (e.g., Euclidean), so features with larger numeric ranges can dominate the clustering process. Therefore, **feature scaling** is crucial to ensure that all features contribute equally.

Common scaling techniques include:

- **Z-scaling (standardization)**: Transforms features to have zero mean and unit standard deviation
- **Min-max scaling**: Scales features linearly to a fixed range, typically [0, 1]

Scaling significantly improves clustering outcomes by preventing dominant features from skewing the results. For instance, in patient clustering tasks, unscaled age values might outweigh binary indicators like prior heart attacks or ST elevation.

# Practical Example: Medical Patient Clustering

A dataset of patients with four features: heart rate, number of prior heart attacks, age, and ST elevation (binary), along with a binary outcome indicating death from a heart attack, was analyzed to find clusters enriched with high-risk patients.

Initially, clustering without feature scaling produced poor results. Clusters had similar proportions of positive outcomes, indicating that the algorithm failed to group high-risk patients effectively. After applying Z-scaling, the clustering quality improved significantly. Clusters began to show distinct differences in the fraction of positive outcomes, allowing identification of high-risk patient groups.

Even with better specificity, sensitivity remained low—many high-risk patients still weren’t captured within the highest-risk clusters. By increasing the number of clusters to 4 or 6, more high-risk subgroups were revealed. These clusters captured patients with distinct risk patterns, such as older age or prior heart attacks, demonstrating that varying K and scaling choices can expose finer details in the data.

This example highlights the **iterative nature of clustering**: tuning parameters like K, choosing appropriate scaling methods, and analyzing outcomes are essential to uncovering meaningful patterns in real-world datasets.






