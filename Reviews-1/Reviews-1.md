#  Natural Language Toolkit (NLTK)

NLTK (Natural Language Toolkit) is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.


##  Installation

```bash
pip install nltk
```

To download datasets and corpora:
```python
import nltk
nltk.download()
```

For specific packages:
```python
nltk.download('punkt')       # Tokenizers
nltk.download('stopwords')   # Stopwords
nltk.download('wordnet')     # WordNet Lemmatizer
nltk.download('averaged_perceptron_tagger')  # POS tagger
```

## Basic Components of NLTK

### 1. Tokenization
Tokenization is splitting text into words or sentences.
```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLTK is a powerful Python library. It helps with NLP tasks."
print(sent_tokenize(text))
print(word_tokenize(text))
```


### 2. Stopwords Removal
Stopwords are common words that may be removed from text (like “the”, “and”, “is”).
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
words = word_tokenize("This is an example showing off stop word filtration.")
filtered = [w for w in words if w.lower() not in stop_words]
print(filtered)
```

### 3. Stemming
Stemming reduces words to their root form.

```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = ["python", "pythoner", "pythoning", "pythonly"]
for w in words:
    print(ps.stem(w))
```

### 4. Lemmatization
Lemmatization is similar to stemming but returns real words.

```python 
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos="v"))
```

### 5. Part of Speech (POS) Tagging
POS tagging labels each word with its grammatical role.
```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize

sentence = "Python is a great programming language."
print(pos_tag(word_tokenize(sentence)))
```

### 6. Named Entity Recognition (NER)
NER identifies names of people, places, organizations, etc.

```python
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

sentence = "Apple was founded by Steve Jobs in California."
tokens = word_tokenize(sentence)
tags = pos_tag(tokens)
tree = ne_chunk(tags)
print(tree)
```


### 7. WordNet – Lexical Database
WordNet is a large lexical database of English.

```python
from nltk.corpus import wordnet

syns = wordnet.synsets("program")
print("Synonyms:", [s.name() for s in syns])
print("Definition:", syns[0].definition())
print("Examples:", syns[0].examples())
```

### Corpus Access
NLTK provides many datasets like movie reviews, Gutenberg texts, etc.

```python
from nltk.corpus import gutenberg

print(gutenberg.fileids())
print(gutenberg.words('austen-emma.txt')[:20])
```

### Text Classification (Naive Bayes Example)
```python
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

def word_feats(words):
    return {word: True for word in words}

train = [({'Python': True}, 'pos'), ({'bad': True}, 'neg')]
classifier = NaiveBayesClassifier.train(train)

print(classifier.classify({'Python': True}))  # pos
print(classifier.classify({'bad': True}))     # neg
```


#  Exploratory Data Analysis (EDA) 

**Exploratory Data Analysis (EDA)** is the process of analyzing data sets to summarize their main characteristics, often using visual methods. EDA helps understand the data structure, spot anomalies, detect patterns, and test hypotheses.



##  Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 1. Load the Dataset
```python
df = pd.read_csv('your_dataset.csv')
df.head()
```

### 2. Basic Information
```python
df.shape
df.info()
df.dtypes
```

### Summary Statistics
```python
df.describe()
```

### 3. Checking Missing Values
```python
df.isnull().sum()
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
```


### 4. Understanding Categorical Features
```python
df['category_column'].value_counts()
sns.countplot(data=df, x='category_column')
```

### 5. Distribution of Numerical Features
```python
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()
```
Or use seaborn for single features:
```python
sns.histplot(df['numeric_column'], kde=True)
```

### 6. Correlation Analysis
```python
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm')
```

### 7. Grouping & Aggregation
```python
df.groupby('category_column')['numeric_column'].mean()
```

### 8. Outlier Detection
Boxplot
```python
sns.boxplot(x=df['numeric_column'])


```
Z-score or IQR

```python
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
df_outliers = df[(z_scores > 3).any(axis=1)]
```

### 9. Pairwise Relationships
```python
sns.pairplot(df, hue='target_column')
```

### 10. Feature Engineering Ideas
Binning numerical variables

Creating new features from existing ones

Encoding categorical features (Label Encoding / One-Hot Encoding)

Scaling/normalizing numerical columns

### 11. Handling Missing Data
```python
# Drop missing values
df.dropna(inplace=True)

# Or fill them
df['column'].fillna(df['column'].mean(), inplace=True)
```

### 12. Data Cleaning
Fix inconsistent formatting (e.g., "yes"/"Yes"/"Y")
Remove duplicates:

```python
df.duplicated().sum()
df.drop_duplicates(inplace=True)
```

### 13. Save Cleaned Data
```python
df.to_csv('cleaned_dataset.csv', index=False)
```

#  Pandas DataFrame Operations 

A **DataFrame** is a 2-dimensional labeled data structure in **Pandas**, similar to a table in SQL or an Excel spreadsheet. This guide covers key DataFrame operations for data manipulation and analysis.


##  1. Creating a DataFrame

```python
import pandas as pd

# From a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['London', 'Paris', 'Berlin']
}
df = pd.DataFrame(data)
```

## 2. Inspecting the Data
```python
df.head()         # First 5 rows
df.tail()         # Last 5 rows
df.info()         # Data types and nulls
df.describe()     # Summary stats (numerical)
df.columns        # List of column names
df.index          # Row indices
df.shape          # Rows and columns
```

## 3. Selecting Data
```python
df['Name']               # Single column (Series)
df[['Name', 'City']]     # Multiple columns (DataFrame)
```

Rows
```python
df.loc[0]                # By label/index
df.iloc[0]               # By position
df[1:3]                  # Row slicing
```

Filtering
```python
df[df['Age'] > 30]                     # Conditional filter
df[(df['Age'] > 25) & (df['City'] == 'Paris')]  # Multiple conditions
```

## 4. Adding and Modifying Data
```python
df['Country'] = ['UK', 'France', 'Germany']    # Add column
df['Age'] = df['Age'] + 1                      # Update column
```

Rename Columns
```python
df.rename(columns={'Name': 'Full Name'}, inplace=True)
```

## 5. Removing Data

```python
df.drop('Country', axis=1, inplace=True)       # Drop column
df.drop(1, axis=0, inplace=True)               # Drop row by index
```

## 6. Iterating Over Rows
```python
for index, row in df.iterrows():
    print(row['Name'], row['City'])
```

## 7. Aggregation

```python
df['Age'].mean()
df.groupby('City')['Age'].mean()
df.agg({'Age': ['min', 'max', 'mean']})
```

## 8. Sorting Data

```python
df.sort_values(by='Age')                 # Ascending
df.sort_values(by='Age', ascending=False)  # Descending
```

## 9. Handling Missing Data

```python
df.isnull().sum()              # Count missing values
df.dropna()                    # Drop rows with NaN
df.fillna(value=0)             # Replace NaN with value
```

## 10. Merging, Joining, Concatenating
Merge (like SQL join)
```python
pd.merge(df1, df2, on='key')
```

Concatenation
```python
pd.concat([df1, df2], axis=0)   # Stack rows
pd.concat([df1, df2], axis=1)   # Stack columns
```


## 11. String Operations
```python
df['Name'].str.upper()
df['City'].str.contains('on')
```
## 12. Apply Functions
```python
df['AgeGroup'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Old')
```

## 13. Exporting and Importing Data
```python
df.to_csv('output.csv', index=False)
df = pd.read_csv('output.csv')
```
