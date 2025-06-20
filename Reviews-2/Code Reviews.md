##  Process2_House_Reviews

This script processes the `Hotel_Reviews.csv` dataset with the following steps:

1. **Import the required library**  
   - `pandas` is used for data manipulation.

2. **Load the dataset**  
   - Reads the CSV file named `Hotel_Reviews.csv` into a DataFrame called `df`.

3. **Drop latitude and longitude columns**  
   - Columns `lat` and `lng` are removed as they are not needed for analysis.

4. **Standardise hotel addresses**  
   - A function `replace_address()` is defined to map full hotel addresses to just `'City, Country'` based on known patterns.
   - It replaces variations of the same city-country pairs (e.g., full street addresses) with consistent location names.

5. **Apply the address simplification**  
   - The address replacement function is applied row-wise to the `Hotel_Address` column.

6. **Drop unnecessary column**  
   - `Additional_Number_of_Scoring` is dropped because it's redundant or irrelevant.

7. **Recalculate 'Total_Number_of_Reviews'**  
   - This column is replaced with the actual number of reviews per hotel, using `groupby` and `transform("count")`.

8. **Recalculate 'Average_Score'**  
   - The average `Reviewer_Score` per hotel is calculated using `groupby` and `transform("mean")`, rounded to 1 decimal place.

9. **Print hotel count per city-country location**  
   - Uses `groupby()` to count unique hotel names per simplified hotel address.
   - The result is printed as a DataFrame showing how many unique hotels are in each city-country pair.

##  sentiment_analysis_housingNLP

This script performs Natural Language Processing (NLP) on hotel reviews to assess sentiment using VADER and clean text by removing stop words.

###  1. Import Libraries and NLTK Resources
- `time`, `pandas`, and `nltk` are used for data processing and timing.
- Downloads the **VADER lexicon** and **English stopwords** from NLTK.

### 2. Initialize VADER Sentiment Analyzer
- `SentimentIntensityAnalyzer` from NLTK is used to score reviews based on sentiment (positive/negative/neutral).

###  3. Define Sentiment Calculation Function
```python
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]
```
### 4. Load the Cleaned Reviews Dataset
- Loads `Hotel_Reviews_Filtered.csv` from the same directory.
- This file is the result of previous cleaning (e.g., tag processing, column drops, address standardisation).

```python
df = pd.read_csv("Hotel_Reviews_Filtered.csv")
```
### 5. Remove Stop Words
Uses NLTK’s English stopword list to filter out common words like the, is, and.

Improves the effectiveness of sentiment analysis by reducing linguistic noise.

Applies the cleaning to both Negative_Review and Positive_Review.

```python
cache = set(stopwords.words("english"))

def remove_stopwords(review):
    return " ".join([word for word in str(review).split() if word.lower() not in cache])

df["Negative_Review"] = df["Negative_Review"].apply(remove_stopwords)   
df["Positive_Review"] = df["Positive_Review"].apply(remove_stopwords)
```
### 6. Calculate Sentiment Scores
Uses SentimentIntensityAnalyzer from NLTK’s VADER to get the compound sentiment score for each review.

Adds two new columns:

Negative_Sentiment

Positive_Sentiment
```python
df["Negative_Sentiment"] = df["Negative_Review"].apply(calc_sentiment)
df["Positive_Sentiment"] = df["Positive_Review"].apply(calc_sentiment)
```

### 7. Preview Sentiment Scores
Sorts the dataset by sentiment scores to quickly review the most negative and most positive reviews.

```python
df = df.sort_values(by="Negative_Sentiment", ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]].head())

df = df.sort_values(by="Positive_Sentiment", ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]].head())
```

### 8. Reorder Columns for Readability
Moves key features like Hotel_Name, sentiment scores, and tags to the front for better inspection and future analysis.

```python
columns_order = [
    "Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score",
    "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment",
    "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler",
    "Business_trip", "Group", "Family_with_young_children",
    "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"
]
df = df.reindex(columns=columns_order)
```

### 9. Save Results
Writes the fully processed dataset to a new CSV called Hotel_Reviews_NLP.csv in the current working directory.

```python
df.to_csv("Hotel_Reviews_NLP.csv", index=False)
```


##  Data Cleaning & Feature Engineering for Hotel Reviews

### 1. Load Dataset
- Loads the raw dataset `Hotel_Reviews.csv` using pandas.

```python
df = pd.read_csv("Hotel_Reviews.csv")
```

### 2. Drop Unused Location Columns
Removes geospatial data lat and lng since they're not needed.

```python
df.drop(["lat", "lng"], axis=1, inplace=True)
```
### 3. Standardise Hotel Addresses
Replaces detailed hotel addresses with a simplified City, Country format.

```python
def replace_address(row):
    if "Netherlands" in row["Hotel_Address"]:
        return "Amsterdam, Netherlands"
    elif "Barcelona" in row["Hotel_Address"]:
        return "Barcelona, Spain"
    elif "United Kingdom" in row["Hotel_Address"]:
        return "London, United Kingdom"
    elif "Milan" in row["Hotel_Address"]:        
        return "Milan, Italy"
    elif "France" in row["Hotel_Address"]:
        return "Paris, France"
    elif "Vienna" in row["Hotel_Address"]:
        return "Vienna, Austria" 

df["Hotel_Address"] = df.apply(replace_address, axis=1)
```

### 4. Drop Redundant Score Column
Removes Additional_Number_of_Scoring, which overlaps with calculated stats.
```python
df.drop(["Additional_Number_of_Scoring"], axis=1, inplace=True)
```

### 5. Recalculate Review Count & Average Score per Hotel
Ensures Total_Number_of_Reviews reflects actual number of reviews per hotel.

Computes Average_Score from actual reviewer scores.

```python

df["Total_Number_of_Reviews"] = df.groupby('Hotel_Name')["Hotel_Name"].transform('count')
df["Average_Score"] = round(df.groupby('Hotel_Name')["Reviewer_Score"].transform('mean'), 1)

```

### 6. Clean Tags Column
Strips square brackets and converts list-like strings to comma-separated strings.
```python
df.Tags = df.Tags.str.strip("[']")
df.Tags = df.Tags.str.replace(" ', '", ",", regex=False)
```

### 7. Create Tag-Based Binary Features
Creates binary columns for various traveller types and purposes.

```python
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)
```

### 8. Drop Irrelevant Columns
Removes columns unrelated to current analysis or already represented elsewhere.

```python
df.drop([
    "Review_Total_Negative_Word_Counts",
    "Review_Total_Positive_Word_Counts",
    "days_since_review",
    "Total_Number_of_Reviews_Reviewer_Has_Given"
], axis=1, inplace=True)
```

### 9. Save Cleaned Dataset
Saves the processed DataFrame to Hotel_Reviews_Filtered.csv in the current working directory.

```python
df.to_csv("Hotel_Reviews_Filtered.csv", index=False)
```

