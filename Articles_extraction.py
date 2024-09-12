# -*- coding: utf-8 -*-
"""
Created on Apr 12 10:30:00 2024

@author: Nanou Ouben
"""

# Importing important libraries

import re
import string
import nltk
from Bio import Entrez # for the Pubmed API
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import gensim
from gensim import corpora


# Import necessary pacakges from NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# The research query on PubMed

query = """
    ((pregnant women) AND (smoking OR alcohol OR obesity) 
         AND (infant health OR baby health OR child health)) 
            AND (future consequences OR long-term effects)
            """
            
# Function that fetches articles' ids
def fetch_article_ids(query, max_results=100): # n° of articles = 100
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

# Function to fetch the content of the articles
def fetch_article_details(ids):
    ids_str = ",".join(ids)
    handle = Entrez.efetch(db="pubmed", id=ids_str, retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    return records

# Fetch the ids of the articles corresponding to the query
article_ids = fetch_article_ids(query)

# Fetch the detail of these articles
article_details = fetch_article_details(article_ids)

########################################## Text preprocessing #########################################

# Handcrafted List of non-pertinent words to add to the list of stopwords

additional_stopwords = {
    'pregnancy', 'women', 'pregnant', 'use', 'study', 
    'research', 'maternal', 'placenta',  'occasionally',
    '3rd', 'trimester', 'week', 'risk', 'pattern',
    'breast', 'feed','third' ,'expression', 'gestational'
    , 'healthy', 'author', 'medical', 'doctor',  
    'food', 'thing', 'placental', 'foetal', 
    'expression','exposure','year','association', 
    'associated','mother','article', 'woman', 
    'infection', 'knowledge','someone', 'permanent',
    'management','week', 'possibility', 
    'instead', 'itervention', 'factor','aim',
    'fetal','cohort', 'time', 'adverse','dietary',
    'among', 'including', 'taboo'}


# Text pretreatment function
def preprocess_text(text):
    # lowercasing
    text = text.lower()
    # deleting punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Word Tokenization
    words = word_tokenize(text)
    # uploading the stopwords
    stop_words = set(stopwords.words('english'))
    # adding the handcrafted list of stopwords to the nltk english stopwords
    all_stopwords = stop_words.union( additional_stopwords)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in all_stopwords]
    preprocessed_text = ' '.join(words)
    return preprocessed_text



# Preprocessing the text of the articles
preprocessed_texts = []

for article in article_details['PubmedArticle']:
    # Extracting the articles' titles
    title = article['MedlineCitation']['Article']['ArticleTitle']
    # Extracting the abstracts
    abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0] if 'Abstract' in article['MedlineCitation']['Article'] else 'No abstract available'
    # Combining the titles and the abstracts
    full_text = f"{title} {abstract}"
    # Preprocessing the combined texts
    preprocessed_text = preprocess_text(full_text)
    preprocessed_texts.append(preprocessed_text)
    print(preprocessed_texts)



########################################### Embeddings ################################################

# BERT embeddings

from transformers import BertTokenizer, BertModel
import torch

# Import the BERT model and Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate the BERT embeddings
def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the special token [CLS] as a text embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        embeddings.append(cls_embedding)
    return embeddings

# Generate the embeddings for the preprocessed texts
embeddings = get_bert_embeddings(preprocessed_texts)

# Show the embeddings for the 5 first articles
for i, embedding in enumerate(embeddings[:5]):
    print(f"Embedding pour l'article {i+1}:\n{embedding}\n")




# TF-IDF Representations

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the preprocessed snippets  into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
tfidf_matrix

# Transform the TF-IDF matrix into a DataFrame for visualization
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Show the heading the TF-IDF matrix
print(tfidf_df.head())

# Save the TF-IDF matrix in a CSV File 
output_path = './output/tfidf_abstracts.csv'
tfidf_df.to_csv(output_path, index=False)


# Function to plot the co-occurrence matrix
def plot_cooccurrence_matrix(text_data, Articles, top_n_words=20):
    # Vectorize the text
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(text_data)
    words = vectorizer.get_feature_names_out()

    # Compute the co-occurrence matrix
    cooccurrence_matrix = (dtm.T * dtm)
    cooccurrence_matrix.setdiag(0)
    cooccurrence_matrix = cooccurrence_matrix.toarray()
    
    # Normalize the co-occurrence matrix
    cooccurrence_matrix_normalized = cooccurrence_matrix / cooccurrence_matrix.max()

    # Apply the log transformation
    cooccurrence_matrix_log = np.log1p(cooccurrence_matrix_normalized)

    # Get the top n words most present in the corpus
    word_counts = np.array(dtm.sum(axis=0)).flatten()
    top_n_indices = word_counts.argsort()[-top_n_words:]
    top_words = [words[i] for i in top_n_indices]

    # Plot the co-occurrence matrix pour for the top N words
    top_cooccurrence_matrix = cooccurrence_matrix_log[top_n_indices][:, top_n_indices]

    plt.figure(figsize=(10, 10))
    plt.imshow(top_cooccurrence_matrix, cmap='viridis')
    plt.xticks(range(len(top_words)), top_words, rotation=90)
    plt.yticks(range(len(top_words)), top_words)
    plt.colorbar()
    plt.title(f'Matrice de Co-occurrence pour {Articles}')
    plt.show()
    

plot_cooccurrence_matrix(preprocessed_texts, 'Articles', top_n_words=20)


