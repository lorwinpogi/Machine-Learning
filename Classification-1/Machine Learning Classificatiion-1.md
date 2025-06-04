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



