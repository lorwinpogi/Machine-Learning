**Logistic Regression Training**

A logistic regression model will be trained using the cleaned and transformed pumpkin data from the previous tutorial. Participants are advised to continue in the same notebook found in the ML for beginners repository under the regression and logistics folders, ensuring that all previous code cells are executed before adding new code. The resulting data frame should feature rows corresponding to pumpkin packages and columns with one-hot and ordinal features.

**Data Preparation**
Data preparation involves dividing the dataset into input features (X) and labels (Y), where Y represents the color column and X contains all other features. The dataset is then split into training and test sets for model training and evaluation. Following this, the focus shifts to creating the logistic regression model.

**Model Predictions**
After training the logistic regression model using input features and labels, it can predict labels for the test data, resulting in a list of predicted labels, where each label is represented as either zero or one. The classification report provides valuable insights into the quality of these predictions.

**Confusion Matrix**
The confusion matrix reveals that there were 162 true negatives (accurate orange predictions) and 22 true positives (accurate white predictions). It also shows 4 false positives (pumpkins incorrectly predicted as white) and 11 false negatives (pumpkins incorrectly predicted as orange). These values provide valuable insights into the predictive performance of the model.

**Prediction Accuracy**
Summing the first row of a prediction matrix reveals the total orange predictions, while summing the first column shows the actual number of orange packages. The overall sum gives the total predictions made, which can be used to calculate prediction accuracy. However, accuracy alone can be misleading in cases of unbalanced or skewed data.

**Model Evaluation Metrics**
In scenarios with unbalanced data, relying solely on accuracy can lead models to predict the majority class, diminishing their utility. Instead, metrics like Precision, Recall, and F1 Score are crucial, as Precision reflects the true positive predictions relative to all positive predictions made.

**F1 Score Importance**
Recall value measures the accuracy of identifying actual white packages among all predictions, and it is crucial alongside precision for assessing model quality. These metrics are often combined into the F1 score, which is vital for evaluating models trained with unbalanced datasets. The classification report also provides macro and weighted averages for these metrics.

**Precision Averages**
Precision calculations involve different averages, such as the macro average, which considers both positive and negative classes equally, and the weighted average, which accounts for class distribution in the dataset. In the example provided, 83% of pumpkins are orange and 17% are white, leading to a weighted average that reflects this imbalance. This video illustrates how to train a logistic regression model with unbalanced data and interpret the results.

**Preparing Data for Logistic Regression**
Begin with a cleaned and transformed pumpkin dataset where rows represent pumpkin packages and columns represent one-hot and ordinal features created previously.
Separate the dataset into input features X (all columns except the label) and label Y (the "color" column indicating pumpkin color).
Split the data into training and test sets: training data is used to train the model, and test data is reserved to evaluate the model's performance on unseen data.

**Training and Using a Logistic Regression Model**
Instantiate a logistic regression class and train (fit) it using the training input features X_train and training labels y_train.
After training, use the model to predict labels on the test input data to generate predicted labels (zeros and ones corresponding to pumpkin color classes).

**Understanding the Confusion Matrix**
- The confusion matrix summarizes prediction results:
- True Negatives (TN): Correctly predicted orange pumpkins (162 cases).
- True Positives (TP): Correctly predicted white pumpkins (22 cases).
- False Positives (FP): Incorrectly predicted white pumpkins (4 cases).
- False Negatives (FN): Incorrectly predicted orange pumpkins (11 cases).

True positives and true negatives appear on the diagonal of the matrix.
Summing rows or columns gives total predictions for a class or total actual instances, respectively, enabling detailed analysis of prediction distribution.


Metrics for Model Evaluation
| Metric            | Definition | Use Case |
| :---------------- | :------: | ----: |
| Accuracy      |   Fraction of all correct predictions over total predictions   | Good general metric, but misleading on skewed data |
| Precision          |   Fraction of correctly predicted positive cases out of all positive predictions   | Measures correctness of positive predictions |
| Recall    | Fraction of correctly predicted positive cases out of all actual positive cases  | Measures completeness in detecting positives|
| F1 Score |  Harmonic mean of precision and recall, balancing both metrics   | Preferred metric for unbalanced datasets |


Accuracy is intuitive but can be deceptive with unbalanced data, as a model predicting the majority class can always have high accuracy but poor usefulness.
Precision and recall complement each other and are combined into the F1 score to provide a balanced quality measure, especially important for unbalanced data scenarios.


Averages in Classification Reports
| Average Type              | Calculation Method |     Interpretation  |
| :---------------- | :------: | ----: |
| Macro Average| Simple average of metric values for each class   | Treats all classes equally |
| Weighted Average  |   Average weighted by the number of instances per class (e.g., 83% orange, 17% white)   | Reflects class imbalance in the metric |

The macro average treats each class equally, regardless of class size, while the weighted average accounts for class imbalance by weighting metrics by class prevalence in the dataset.


**Data Analysis**
To predict class labels such as orange or white, logistic regression is a suitable method. Before model development, it's crucial to analyze the dataset for necessary clean-up or transformations. A preliminary inspection of the data will be conducted by reviewing a few rows and selectively focusing on key features.

**Data Visualization**
Unnecessary features such as city name, package variety, origin, item size, and color will be removed, and rows with missing values will be dropped, resulting in a cleaner dataset. Seaborn will be used to visualize the number of orange and white pumpkins by variety, allowing for a better understanding of the data, particularly that most pumpkins are orange, while some miniature and Howden white types are exceptions.

**Data Transformation**
The dataset for pumpkin prediction is skewed, necessitating transformations for effective predictions. Much of the data is in string form, but logistic regression requires numerical input. Understanding the process of converting categorical features to numerical values with one-hot vectors is critical for this transformation.

**Encoding Features**
For features without implied order, a one-hot encoder is used, while ordinal encoding is applied to features with a defined order, such as pumpkin sizes, converting them into integers. The output is managed through a column transformer, ensuring data remains in a pandas DataFrame. After transformation, the item size column displays integers from 0 to 6, and other features like the city name are encoded into several one-hot variables.

**Final Data Checks**
Vector columns are created for each city, and pumpkin colors are numerically encoded as 0 for orange and 1 for white using a label encoder. To verify the transformation, Seaborn is utilized to visualize the size distribution of orange and white pumpkins with a categorical plot and a swarm plot.

**Logistic Regression Model**
A transformation is performed on the pumpkin color label, encoding 'orange' and 'white' as 0 and 1 using a label encoder. To visualize the distribution of pumpkin sizes for each color, a cat plot and a swarm plot will be utilized with Seaborn.

**Data Set Overview and Objective**
The data set used is the pumpkin data set previously employed in linear regression videos, aiming to predict pumpkin color (orange or white) based on other features. Logistic regression is suitable since the target is a categorical class, not a numeric value.
Initial data exploration involves printing rows and selecting relevant features: city, name, package, variety, origin, item size, and color. Rows with missing values are dropped to clean the data.

**Data Visualization and Class Imbalance**
Seaborn, a popular data visualization library, is used to plot the number of orange and white pumpkins across different varieties, with color represented as hue. This visualization reveals that most pumpkins are orange, few are white, and some varieties like miniature pumpkins appear in both colors, while Howden white type pumpkins are all white.
The data is skewed due to the imbalance between the two classes (orange and white pumpkins), indicating the need for careful data transformation before model training

**Handling Categorical Features**
Logistic regression requires numerical input, so categorical string features must be transformed into numerical representations.
One-hot encoding is used for categorical features without an implied order (city, package, variety, origin). This creates binary columns for each category.
Ordinal encoding is applied to the item size feature, which has an inherent order. Sizes are mapped to integers from 0 to 6, preserving the category order.

**Data Transformation Pipeline**
Encoders (one-hot and ordinal) are combined using a column transformer to apply appropriate transformations to each feature and output a clean pandas DataFrame.
After transformation, the item size column is replaced by numeric values, and city names are split into multiple one-hot encoded columns.

**Label Encoding and Final Visualization**
The target variable, pumpkin color, is transformed from strings ("orange", "white") to numeric labels (0 and 1) using a label encoder, making it suitable for logistic regression.
Seaborn visualizations, such as cat plots and swarm plots, are used again to verify the correctness of transformations and to better understand the distribution of pumpkin sizes and colors across varieties.


**ROC Curve Introduction**
Evaluate your logistic regression model's performance using the ROC curve, a visualization tool named after its military application in the 1940s. This video continues from previous lessons, encouraging you to open your existing notebook, run the prior code, and incorporate the new coding instructions presented.

**Binary Classification**
ROC curves are essential for evaluating a binary classifier's performance as its decision threshold changes. A binary classifier, such as the logistic regression model discussed earlier, categorizes data into two distinct classes. The choice of threshold is crucial in determining the classification outcome.

**Classification Thresholds**
Output values from the model range between 0 and 1, with a common classification threshold set at 0.5, categorizing values below as orange and above as white. Adjusting this threshold will influence the results derived from the confusion matrix, which will be examined in detail later. Understanding how to calculate various metrics using the confusion matrix is essential for evaluating model performance.

**True/False Positive Rates**
To construct the ROC curve, we must evaluate the true and false positive rates. The true positive rate, or recall, measures the proportion of correctly identified white packages from all actual white packages, while the false positive rate assesses the fraction of wrongly classified orange packages as white. By selecting different classification threshold values, we can compute and plot the respective true and false positive rates.

**ROC Curve Calculation**
The ROC curve is generated by plotting the positive rate against the false positive rate across various classification thresholds. Using code, we can easily implement this by calling the ROC curve function with actual and predicted labels, resulting in rates and thresholds needed for the plot. Understanding the shape of the ROC curve is important as it relates to the corresponding formulas.

**ROC Curve Shape**
When the threshold is set to zero, all inputs are classified as positive, resulting in true positive and false positive rates of one, while a threshold of one classifies all inputs as negative, yielding rates of zero. Consequently, the ROC plot will always feature points at (0, 0) and (1, 1), with the intermediate points reflecting the quality of predictions.

**ROC AUC Score**
The goal is to achieve a true positive rate close to one and a false positive rate near zero, culminating in an ideal ROC curve that goes from (0,0) to (0,1) to (1,1). The ROC curve for the logistic regression model closely matches this ideal, indicating effective data preparation and training. This success instills confidence in the model's predictive capabilities.

**Ideal ROC Curve**
The goal is to achieve a true positive rate close to one and a false positive rate near zero, resulting in an ideal ROC curve that rises directly to the top right corner. The logistic regression model's ROC curve closely approximates this ideal, indicating effective data preparation and model training. This outcome instills confidence in the model's predictive capabilities.

**ROC Curve and Logistic Regression Performance**
The ROC curve (Receiver Operating Characteristic curve) is a visualization tool to analyze the performance of binary classifiers as their decision threshold varies. It originated from military radar receivers in the 1940s.
A binary classifier assigns inputs into one of two categories. Logistic regression is a binary classifier because it classifies pumpkins as either orange or white.
Logistic regression outputs values between 0 and 1; a classification threshold (commonly 0.5) determines the predicted class. Changing this threshold affects the confusion matrix and classification results.



**True Positive Rate, False Positive Rate, and Thresholds**
The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR) for different classification thresholds:
True Positive Rate (Recall): Fraction of correctly predicted positive cases out of all actual positives.
False Positive Rate: Fraction of incorrectly predicted positives out of all actual negatives.
To generate the ROC curve, calculate TPR and FPR at various thresholds, then plot TPR (y-axis) vs. FPR (x-axis).

**Behavior of ROC Curve at Extreme Thresholds**
At threshold = 0, all inputs are classified as positive, so TPR = 1 and FPR = 1.
At threshold = 1, all inputs are classified as negative, so TPR = 0 and FPR = 0.
Hence, the ROC curve always includes points (0,0) and (1,1), with the curve's shape between these points reflecting model quality.

**Ideal ROC Curve and Model Evaluation**
The ideal ROC curve rises vertically from (0,0) to (0,1), then horizontally to (1,1), representing perfect classification (high TPR, low FPR).
A ROC curve close to this ideal shape indicates a well-trained logistic regression model with good predictive power.
Observing a near-ideal ROC curve confirms effective data preparation and model training, increasing confidence in prediction quality.


