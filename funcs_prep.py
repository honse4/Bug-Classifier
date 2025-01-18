import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

def clean_text(text, lemmatizer, stop_words): 
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]) 
    return text
    
def combine_columns(data, lemmatizer, stop_words):
    data['description'] = data['short_description']+' '+ data['long_description']
    data['description'] = data['description'].apply(lambda x: clean_text(x, lemmatizer, stop_words))

    data['product'] = data['bug_id'] + ' '+ data['component_name'] + ' ' + data['product_name']
    data['product'] = data['product'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    return data

def vectorizer(data, column, n_components):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data[column]).toarray()
    pca = PCA(n_components=n_components)
    pca_matrix = pca.fit_transform(tfidf_matrix) 
    df = pd.DataFrame(pca_matrix, columns=[f'PC{i+1}' for i in range(n_components)]) 
    return df

def processed_df(df1, df2, df3):
    df1.reset_index(drop=True, inplace=True) 
    df2.reset_index(drop=True, inplace=True) 
    df3.reset_index(drop=True, inplace=True)

    df_combined = pd.concat([df1, df2, df3], axis=1)
    df_combined = df_combined.drop(columns=['bug_id', 'component_name','product_name', 'short_description', 'long_description', 'description', 'product'])
    return df_combined