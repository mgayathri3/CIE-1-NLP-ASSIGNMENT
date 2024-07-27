import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import string


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Read the CSV file
reviews = pd.read_csv('Reviews.csv', nrows=5000)

# Preprocess the text to remove punctuation marks
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    if not isinstance(text, str):
        return ''
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in string.punctuation]
    return ' '.join(filtered_tokens)

# Apply the preprocessing function to the reviews
reviews['filtered_text'] = reviews['Text'].apply(preprocess_text)

# Print the processed text without quotes
for index, row in reviews.loc[1:5, ['Text', 'filtered_text']].iterrows():
    original_text = row['Text'].strip('"')
    filtered_text = row['filtered_text'].strip('"')
    print(f"Original: {original_text}")
    print(f"Filtered: {filtered_text}\n")