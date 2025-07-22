# Azure Responsible AI Dashboard for Debugging Machine Learning Models

## Introduction to Responsible AI Dashboard

The Responsible AI Dashboard is a unified platform for operationalizing responsible AI practices in real-world scenarios. It brings together various tools developed by Microsoft Research and the open-source community into a single interface, streamlining workflows and improving discoverability.

By addressing common challenges like tool fragmentation and steep learning curves, the dashboard offers a cohesive environment for both model debugging and responsible decision-making.

The Azure Responsible AI (RAI) Dashboard is an interactive tool that helps data scientists and ML practitioners analyze, interpret, and debug machine learning models. It consolidates various Responsible AI capabilities into one unified UI and provides insights around model fairness, explainability, error analysis, and data quality.

It is part of the `responsibleai` package from the Azure Machine Learning ecosystem.


## Key Features

| Capability         | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **Error Analysis** | Visualize and drill down into subgroups where the model makes incorrect predictions. |
| **Fairness**       | Assess and mitigate performance disparities across sensitive features.      |
| **Explainer**      | View feature importance and SHAP explanations for global and local behavior. |
| **Data Explorer**  | Detect and review data imbalances and quality issues.                       |
| **Counterfactuals**| Explore minimal changes to inputs that flip the model prediction.           |
| **Causal Analysis**| Understand causal relationships and influence of features on outcomes.      |


## Installation

```bash
pip install raiwidgets[notebooks] raiutils responsibleai
```

## Setup in Code

## 1. Train a Model and Prepare Data
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, stratify=data.target, random_state=0)

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
```



 ## 2. Create RAI Insights Object

```python
from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIInsights

rai_insights = RAIInsights(
    model=clf,
    train=X_train,
    test=X_test,
    target_column='target',
    task_type='classification',
    categorical_features=[]
)
```


## 3. Add Debugging Modules

```python
rai_insights.explainer.add()
rai_insights.error_analysis.add()
rai_insights.fairness.add(sensitive_features=['sepal length (cm)'])  # Replace with actual sensitive feature if applicable
rai_insights.counterfactual.add()
rai_insights.causal.add()
```


## 4. Compute the Analyses

```python
rai_insights.compute()
```


## 5. Launch the Dashboard

```python
ResponsibleAIDashboard(rai_insights)
```

### Typical Use Cases in Debugging

| Use Case                       | RAI Feature                    | How It Helps                                                  |
| ------------------------------ | ------------------------------ | ------------------------------------------------------------- |
| Identify model blind spots     | Error Analysis                 | Detects subgroups with high error or misclassification rates  |
| Explain unexpected predictions | Explainer (SHAP)               | Shows which features influenced each decision                 |
| Address unfair outcomes        | Fairness                       | Highlights disparities in performance across sensitive groups |
| Diagnose overfitting           | Data Explorer + Error Analysis | Compare errors across train/test and subgroups                |
| Test robustness                | Counterfactuals                | See how small changes to inputs affect predictions            |
| Investigate feature causality  | Causal Analysis                | Understand which features *drive* model behavior              |


### Export and Integration
You can also export the RAIInsights results for reuse or inspection:

```python
rai_insights.save(path='rai_insights_output')
# Later
from responsibleai import RAIInsights
loaded_rai = RAIInsights.load(path='rai_insights_output')
ResponsibleAIDashboard(loaded_rai)
```

#### Notes
Azure RAI Dashboard can run in local Jupyter, VSCode notebooks, or Azure Machine Learning Studio.

It works with tabular datasets and supports classification and regression tasks.

Designed for integration into ML workflows to meet ethical, legal, and performance debugging needs.

### Responsible AI Dashboard

The Responsible AI Dashboard is a comprehensive tool for implementing responsible AI practices. Hosts Besmira and Manush present insights into this innovative resource. Explore its functionalities and applications.

### Integration of Tools

The integration effort responds to challenges faced by machine learning engineers and data scientists due to fragmented tools that complicated sharing results and learning. By consolidating mature tools into a single dashboard, the initiative streamlines model debugging and supports responsible AI decision-making. A demo showcasing these integrations follows.

### Model Debugging

Model debugging focuses on error identification, diagnostic operations tied to data exploration, and model interpretability. Fairness remains a key consideration throughout these processes.


### Decision Making

The dashboard integrates model debugging with business decision-making, enabling users to understand metrics, examine error distribution, and make responsible, real-world choices. With tools like Econ ML, users can detect specific issues—such as elevated error rates in certain housing cohorts—and apply targeted mitigation strategies. This dual approach balances technical accuracy with practical impact.

### Model Statistics

Model analysis shows older houses are more likely to sell for less than the median, largely due to underrepresentation of high-value older homes in the dataset. To address this, more data on older houses is needed for retraining. Interpretability tools reveal key prediction drivers and demonstrate how counterfactual analysis supports informed decisions. Diagnosis follows error detection to uncover root causes.  
Box plots visualize prediction distributions, e.g., older houses show higher probability of being predicted to sell below median price. The data explorer reveals imbalances like underrepresented expensive older homes. Targeted mitigation includes collecting more data for such cohorts.


### Fairness Issues

Fairness in AI economic analysis is essential. The dashboard identifies sensitive attributes that contribute to counterfactuals and highlights their influence on interpretability metrics. It offers fairness indicators and will include fair learning capabilities, helping users assess and mitigate unfairness while effectively using sensitive data.

Fairness insights come from:

- **Error Analysis**: Sensitive attributes defining high-error cohorts  
- **Interpretability**: Sensitive features appearing as top influencers  
- **Counterfactual Analysis**: Sensitive attributes flipping predictions  

Plans are in place to integrate **Fairlearn**, an open-source fairness toolkit, into a dedicated dashboard section for deeper analysis and mitigation.


### Open Source Collaboration

The Responsible AI Toolbox is open source, promoting collaboration through customizable, modular components. It supports extended functionality, such as model comparison and mitigation strategies across dashboards. Users can build custom components and collaborate directly with the development team.  Its modular design supports adding or removing components for expanded functionality.  Part of the larger **Responsible AI Toolbox**, which includes both front-end visual tools and back-end machine learning capabilities for comparison and mitigation.


### Error Analysis

Automatically identifies model errors using an error tree that highlights cohorts with high error rates.  
Example: An apartment pricing model shows a 24% error rate for bigger, older houses compared to an 11% base rate, accounting for 75% of total errors.  
Problematic cohorts can be saved for further analysis.


### Model Interpretability

Explains how the model makes predictions, globally and at the individual level.  
Displays feature importance across the dataset and compares across cohorts, e.g., "year built" matters more for new houses.  
Investigates features like "overall finish quality," which may negatively affect price predictions.  
Flags reliance on spurious or sensitive correlations.  
Enables debugging of individual predictions through local feature importance.

### Counterfactual Analysis

Determines the minimal change needed to reverse a model prediction.  
Example: A home predicted to sell below median flips to above median if "overall finish quality" increases from 6 to 9.  
Useful for fairness debugging and providing actionable recommendations, e.g., identifying if sensitive attributes like ethnicity influence outcomes.


## Core Areas of the Dashboard: Responsible Business Decision Making

Focuses on extracting actionable insights from historical data for business stakeholders, separate from model predictions.  
Built on Econ ML, an open-source package using double machine learning to estimate causal effects and avoid spurious correlations.

### Aggregate Causal Effects

Displays the average effect of a change (e.g., adding a garage) on actual housing price across the dataset, with confidence intervals.

### Individual Causal Effects

Shows how specific changes impact the price of an individual house.

### Causal What-If

Explores how altering a feature value changes the real-world outcome, not just the model’s prediction.

### Ideal Policy

Recommends targeted treatment strategies per cohort for maximum overall gain. Supports decision-making for use cases like construction planning or marketing.

| Feature             | Prediction What-If                                      | Causal What-If                                              |
|---------------------|----------------------------------------------------------|-------------------------------------------------------------|
| **Focus**           | Impact on model prediction                               | Impact on actual market price                               |
| **Data Source**     | Trained machine learning model                           | Historical data via causal inference (e.g., Econ ML)        |
| **Purpose**         | Debug model behavior, understand sensitivities           | Inform business decisions, estimate real-world effects      |



