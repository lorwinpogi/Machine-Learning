# Tokenization in NLP 

**Tokenization** is a core step in **Natural Language Processing (NLP)**. It involves splitting raw text into smaller units called **tokens**. These tokens can be:

- Words
- Characters
- Subwords
- Sentences

The purpose of tokenization is to simplify the text so that machine learning models can process and analyze it more effectively.



## Why Tokenization is Important

-  Makes text **manageable** for algorithms.
- Converts **unstructured data** (text) into **structured form**.
- Enables downstream NLP tasks like **text classification**, **sentiment analysis**, **machine translation**, etc.

For example, the sentence:

> `"Tokenization is essential in NLP."`

can be tokenized into:

- **Word tokens**: `["Tokenization", "is", "essential", "in", "NLP", "."]`
- **Character tokens**: `["T", "o", "k", "e", "n", "i", "z", "a", "t", "i", "o", "n", " ", "i", "s", " ", "e", ...]`



## Types of Tokenization

### 1. **Word Tokenization**
Splits text by whitespace or punctuation.

```python
import nltk
nltk.word_tokenize("Hello world! This is NLP.")
# Output: ['Hello', 'world', '!', 'This', 'is', 'NLP', '.']
```

### 2. **Sentence Tokenization**
Splits text into sentences.

```python
import nltk
nltk.sent_tokenize("Hello world! This is NLP.")
# Output: ['Hello world!', 'This is NLP.']
```

### 3. Character Tokenization
Splits text into individual characters.
```python
list("NLP")
# Output: ['N', 'L', 'P']
```



### 4. Subword Tokenization
Split words into smaller units, or subwords.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "unhappiness"
tokens = tokenizer.tokenize(text)
print("Subword Tokens:", tokens)
#Output: Subword Tokens: ['un', '##happiness']
```


# Embeddings in NLP 

**Embeddings** are dense, continuous vector representations of discrete data — in NLP, this typically means turning **words**, **phrases**, or **sentences** into numerical vectors that capture their **meaning** and **relationships**.

Unlike one-hot encoding (which is sparse and high-dimensional), embeddings are:

- **Low-dimensional** (e.g., 50 to 1024 dimensions)
- **Real-valued**
- **Semantically meaningful**



## Why Embeddings Are Important in NLP

Natural language is inherently **ambiguous** and **contextual**. Embeddings allow machine learning models to:

- Understand **semantic similarity** (e.g., "king" and "queen" are related)
- Capture **syntactic relationships**
- Handle **vocabulary size reduction**
- Enable **transfer learning** with pre-trained embeddings



## One-Hot Encoding vs Word Embeddings

| Property              | One-Hot Encoding        | Word Embeddings            |
|-----------------------|-------------------------|----------------------------|
| Dimensionality        | High                    | Low                        |
| Sparsity              | Sparse (mostly zeros)   | Dense                      |
| Semantic Information  | None                    | Rich, learned from data    |
| Example Similarity    | All words equally distant| Similar words are close    |


## Common Types of Embeddings

### 1. **Word Embeddings**
- Each word gets a fixed vector.
- Examples:
  - **Word2Vec**
  - **GloVe**
  - **FastText**

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
vector = model.wv['king']
```

### 2. **Subword Embeddings**
Breaks words into smaller units (e.g., FastText handles rare or misspelled words better).

Helps with out-of-vocabulary (OOV) words.

### 3. **Contextual Embeddings**
Vectors change depending on context.

Examples:

ELMo

BERT

GPT

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("Embeddings are powerful", return_tensors="pt")
outputs = model(**inputs)
embedding = outputs.last_hidden_state
```



# Parsing & Part-of-Speech Tagging 

**Part-of-Speech (POS) Tagging** is the process of assigning each word in a sentence its grammatical category (or "part of speech"), such as noun, verb, adjective, etc.

### Example:
For the sentence:

> `"The quick brown fox jumps over the lazy dog."`

A POS tagger would output:


Where:
- `DT` = Determiner
- `JJ` = Adjective
- `NN` = Noun
- `VBZ` = Verb (3rd person singular)
- `IN` = Preposition



## Why is POS Tagging Important?

- Helps understand the **structure** and **syntax** of sentences
- Aids in downstream tasks like:
  - **Named Entity Recognition**
  - **Dependency Parsing**
  - **Information Extraction**
  - **Machine Translation**
- Clarifies **ambiguity** (e.g., "book" as noun vs verb)




**Parsing** refers to analyzing the grammatical structure of a sentence. It determines how words relate to each other and helps build a **syntax tree** or **dependency graph**.

There are two major types:

### 1. **Constituency Parsing**
- Breaks a sentence into nested **phrases** (constituents).
- Builds a **parse tree** showing phrase structure.

**Example Tree**:

(S
(NP The quick brown fox)
(VP jumps
(PP over
(NP the lazy dog))))


### 2. **Dependency Parsing**
- Represents grammatical relationships between words using **directed arcs**.
- Builds a **dependency tree** showing which words modify others.

**Example:**

jumps → fox (subject)
jumps → over → dog (object of preposition)



## POS Tagging Techniques

### 1. **Rule-Based Tagging**
- Uses hand-crafted rules and dictionaries.
- Accurate for simple structures but inflexible.

### 2. **Statistical Tagging**
- Based on probabilistic models like:
  - Hidden Markov Models (HMM)
  - Maximum Entropy Models

### 3. **Machine Learning-Based Tagging**
- Trained on annotated corpora using:
  - Decision Trees
  - Support Vector Machines (SVM)
  - Conditional Random Fields (CRFs)

### 4. **Neural Network-Based Tagging**
- Uses deep learning models such as:
  - LSTM/GRU with CRF
  - BERT-style transformers for contextual tagging



## Parsing Techniques

### 1. **Top-Down / Bottom-Up Parsing (Constituency)**
- Classical methods from formal grammar (e.g., CFG).
- Used in tools like the Stanford Parser.

### 2. **Transition-Based Parsing (Dependency)**
- Builds the tree incrementally using stack and buffer.
- Fast and widely used.

### 3. **Graph-Based Parsing**
- Treats parsing as a graph optimization problem.
- Considers all word pairs and selects the best tree.


## Libraries for POS Tagging & Parsing

| Library       | POS Tagging | Parsing             | Notes                             |
|---------------|-------------|---------------------|-----------------------------------|
| **spaCy**     | ✅          | ✅ (Dependency)     | Fast, production-ready            |
| **NLTK**      | ✅          | ✅ (Constituency)   | Educational, flexible             |
| **Stanza**    | ✅          | ✅ (Both)           | Stanford NLP, deep learning-based |
| **Transformers** | ✅       | ❌ (needs add-ons)  | Contextual tagging via BERT       |
| **CoreNLP**   | ✅          | ✅ (Both)           | Java-based, powerful              |


## Example with spaCy (Python)

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

# POS Tagging
for token in doc:
    print(token.text, token.pos_, token.tag_)

# Dependency Parsing
for token in doc:
    print(token.text, "→", token.head.text, "(", token.dep_, ")")
```


# Word and Phrase Frequencies in NLP 



**Word and phrase frequencies** refer to how often individual words or sequences of words (phrases) appear in a given corpus or document. Frequency analysis is one of the most fundamental steps in **Natural Language Processing (NLP)** and **text mining**.

- **Word Frequency**: Count of how many times each word occurs.
- **Phrase (n-gram) Frequency**: Count of how often sequences of words (n-grams) occur.


## Why Frequency Matters in NLP

Frequency analysis helps:

- Identify important or common terms
- Extract **keywords** or **topics**
- Build **vocabularies** for models
- Generate **features** for machine learning
- Perform **text classification**, **sentiment analysis**, and **information retrieval**



## Types of Frequency Analysis

### 1. **Unigram Frequency**
- Frequency of individual words.
- Example:
  - `"NLP is fun. NLP is useful."`
  - Unigrams: `{"NLP": 2, "is": 2, "fun": 1, "useful": 1}`

### 2. **N-gram Frequency**
- Frequency of contiguous word sequences:
  - **Bigrams** (2-word phrases): `"NLP is"`, `"is fun"`
  - **Trigrams** (3-word phrases): `"NLP is fun"`

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["NLP is fun", "NLP is useful"]
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)
print(vectorizer.vocabulary_)
```


### 3. **Term Frequency (TF)**
Raw count of how many times a term appears in a document.

### 4. **Term Frequency-Inverse Document Frequency (TF-IDF)**
Weighs a term’s importance by how common it is in a document and how rare it is across documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["NLP is fun", "NLP is useful"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
print(vectorizer.vocabulary_)
```


# N-grams in NLP 


**N-grams** are contiguous sequences of `n` items (typically words or characters) from a given sample of text or speech.

- **Unigram** = 1 word  
- **Bigram** = 2-word sequence  
- **Trigram** = 3-word sequence  
- **4-gram, 5-gram, etc.** = longer sequences

### Example:
For the sentence:

> `"Natural Language Processing is awesome"`

- **Unigrams**: `["Natural", "Language", "Processing", "is", "awesome"]`  
- **Bigrams**: `["Natural Language", "Language Processing", "Processing is", "is awesome"]`  
- **Trigrams**: `["Natural Language Processing", "Language Processing is", "Processing is awesome"]`



## Why Are N-grams Important?

N-grams help capture **local context** and **co-occurrence patterns** in text.

### Key Uses:
- Text classification
- Sentiment analysis
- Machine translation
- Autocomplete and next-word prediction
- Spelling correction
- Language modeling (e.g., GPT)



## Types of N-grams

### 1. **Word-level N-grams**
- Operate on word tokens
- Useful for semantic or topic-based tasks

### 2. **Character-level N-grams**
- Operate on characters
- Useful for language modeling, spelling correction, or identifying text patterns

---

## How to Generate N-grams (Python Example)

### Using `nltk`:
```python
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

text = "Natural language processing is powerful"
tokens = word_tokenize(text)
bigrams = list(ngrams(tokens, 2))
print(bigrams)
```

Using scikit-learn:

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["Natural language processing is powerful"]
vectorizer = CountVectorizer(ngram_range=(1, 2))  # unigrams + bigrams
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
```




