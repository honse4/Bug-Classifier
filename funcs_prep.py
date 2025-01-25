import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
nltk.download('punkt')

def clean_text(text, lemmatizer, stop_words): 
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = '\n'.join([line for line in text.split('\n') if 'org.eclipse' not in line]) 
    text = re.sub(r'(\bat\b\s+\S+\.\S+\(.*?\))|(\bException\b.*?\n\s*at\s+.*?\(.*?\))|(\bError\b.*?\n\s*at\s+.*?\(.*?\))', '', text, flags=re.DOTALL) 
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower()
    tokens = word_tokenize(text) 
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]) 
    return text
    
def prepare_df(data, lemmatizer, stop_words):
    data['description'] = data['short_description']+' '+ data['long_description']
    data['description'] = data['description'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    data['word_count_desc'] = data['description'].apply(lambda row: len(word_tokenize(row)))
    data = data.drop(columns=['short_description', 'long_description']) 
    return data

def vectorizer(data, column, n_components, features):
    tfidf = TfidfVectorizer(max_features=features, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(data[column]).toarray()
    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(tfidf_matrix) 
    df = pd.DataFrame(pca_matrix, columns=[f'PC{i+1}' for i in range(n_components)]) 
    return df

def processed_df(df1, df2):
    df1.reset_index(drop=True, inplace=True) 
    df2.reset_index(drop=True, inplace=True) 

    df_combined = pd.concat([df1, df2], axis=1)
    return df_combined

def w2v(data,vector_size=100, window=5, min_count=1, workers=4):
    sentences = data['description'].apply(lambda x: word_tokenize(x))
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model
