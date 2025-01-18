from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def year(date, is_year_last = True) :
    if is_year_last:
        date_obj = datetime.strptime(date,'%d-%m-%Y')
    else:
        date_obj = datetime.strptime(date,'%Y-%m-%d')
        
    return date_obj.year

def year_plot(data, source, is_year_last = True):
    years = pd.DataFrame([year(date, is_year_last) for date in data['creation_date']], columns=['year'])
    year_plot = years['year'].value_counts().sort_index().plot(kind='bar')
    year_plot.set_ylabel("Frequency")
    year_plot.set_title(f"Bug Reports Count For Each Year - {source}")

def severity_plot(data, source):
    severity_plot = data['severity_code'].value_counts().sort_index().plot(kind='bar')
    severity_plot.set_ylabel("Frequency")
    severity_plot.set_title(f"Severity Code Counts - {source}")

def label_plot(data, source):
    label_plot = data['label'].value_counts().sort_index().plot(kind='bar')
    label_plot.set_ylabel("Frequency")
    label_plot.set_title(f"Label Counts - {source}")

def label_sev_plot(data, source):
    comb = data.groupby(['label', 'severity_code']).agg({'severity_code': 'count'}).rename(columns={'severity_code':'count'}).reset_index()
    pivot = comb.pivot(index='severity_code', columns='label',values='count')
    pivot.plot(kind='bar')
    plt.title(f"Combined Label and Severity Plot - {source}")
    plt.xlabel("Severity code for each label")
    plt.ylabel("Counts")
    plt.show()

def clean_text(text, lemmatizer, stop_words): 
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]) 
    return text