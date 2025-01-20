import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk 
from nltk.tokenize import word_tokenize 
from gensim.models import Word2Vec
nltk.download('punkt')

def clean_text(text, lemmatizer, stop_words): 
    text = re.sub(r'at\s+\S+\.\S+\(.*?\)', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '',text, flags=re.MULTILINE)
    text = '\n'.join([line for line in text.split('\n') if 'org.eclipse' not in line])
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower()
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]) 
    return text
    
def combine_columns(data, lemmatizer, stop_words):
    data['description'] = data['short_description']+' '+ data['long_description']
    data['description'] = data['description'].apply(lambda x: clean_text(x, lemmatizer, stop_words))

    data['product'] = data['bug_id'] + ' '+ data['component_name'] + ' ' + data['product_name']
    data['product'] = data['product'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    return data

def vectorizer(data, column, n_components):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(data[column]).toarray()
    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(tfidf_matrix) 
    df = pd.DataFrame(pca_matrix, columns=[f'PC{i+1}' for i in range(n_components)]) 
    return df

def train_word2vec(data, column): 
    tokenized_data = data[column].tolist() 
    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4) 
    return model 
def get_word2vec_vectors(tokens, model): 
    vectors = [model.wv[word] for word in tokens if word in model.wv] 
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def processed_df(df1, df2, df3):
    df1.reset_index(drop=True, inplace=True) 
    df2.reset_index(drop=True, inplace=True) 
    df3.reset_index(drop=True, inplace=True)

    df_combined = pd.concat([df1, df2, df3], axis=1)
    df_combined = df_combined.drop(columns=['bug_id', 'component_name','product_name', 'short_description', 'long_description', 'description', 'product'])
    return df_combined