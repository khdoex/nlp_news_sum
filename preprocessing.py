import nltk
import spacy
from nltk.corpus import stopwords   
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from datasets import load_dataset
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_dataset("cnn_dailymail", '3.0.0')
article = dataset['train']['article']
summary = dataset['train']['highlights']
ids = dataset['train']['id']

# Define data chunking function
def data_chunks(dataset, chunk_size=40000):
    article = dataset['train']['article']
    summary = dataset['train']['highlights']
    ids = dataset['train']['id']
    chunked_data = []
    for i in range(0, len(article), chunk_size):
        chunk = {
            'id': ids[i:i+chunk_size],
            'article': article[i:i+chunk_size],
            'summary': summary[i:i+chunk_size]
        }
        chunked_data.append(chunk)
        # Save each chunk as a CSV file
        chunk_df = pd.DataFrame(chunk)
        chunk_df.to_csv(f'cnn_chunk_{i//chunk_size}.csv', index=False)
    return chunked_data

chunked_data = data_chunks(dataset)

# Define preprocessing functions
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

def remove_html(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

def remove_punctuation(text):
    exclude = set(string.punctuation)
    return ''.join(char for char in text if char not in exclude)

def clean_fields(df):
    df['article'] = df['article'].apply(lambda x: remove_url(x))
    df['summary'] = df['summary'].apply(lambda x: remove_url(x))
    
    df['article'] = df['article'].apply(lambda x: remove_html(x))
    df['summary'] = df['summary'].apply(lambda x: remove_html(x))
    
    df['article'] = df['article'].apply(lambda x: preprocess_text(x))
    df['summary'] = df['summary'].apply(lambda x: preprocess_text(x))
    
    df['article'] = df['article'].apply(lambda x: remove_punctuation(x))
    df['summary'] = df['summary'].apply(lambda x: remove_punctuation(x))
    return df

# Apply preprocessing to each chunk and save
for i, chunk in enumerate(chunked_data):
    chunk_df = pd.DataFrame(chunk)
    clean_chunk_df = clean_fields(chunk_df)
    clean_chunk_df.to_csv(f'clean_cnn_chunk_{i}.csv', index=False)

# Load and print the first few rows of the cleaned chunk
#clean_chunk_df = pd.read_csv('clean_cnn_chunk_0.csv')
#print(clean_chunk_df.head())



#print the min-avg-max-median of the length of the articles
article_length = clean_chunk_df['article'].apply(lambda x: len(str(x).split()))
print(f"Min: {article_length.min()}")
print(f"Max: {article_length.max()}")
print(f"Avg: {article_length.mean()}")
print(f"Median: {article_length.median()}")
print(f"Standard Deviation: {article_length.std()}")
print(f"Variance: {article_length.var()}")
"""
Min: 1
Max: 1864
Avg: 607.2325963959024
Median: 550.0
Standard Deviation: 324.19258779822667
Variance: 105100.8339833109
"""

#print the min-avg-max-median of the length of the summaries
summary_length = clean_chunk_df['summary'].apply(lambda x: len(str(x).split()))
print(f"Min: {summary_length.min()}")
print(f"Max: {summary_length.max()}")
print(f"Avg: {summary_length.mean()}")
print(f"Median: {summary_length.median()}")
print(f"Standard Deviation: {summary_length.std()}")
print(f"Variance: {summary_length.var()}")


"""
Min: 1
Max: 84
Avg: 41.727499314573414
Median: 42.0
Standard Deviation: 9.387466121887579
Variance: 88.12452018958703
"""


