"""
Created on Apr 21 14:55:08 2024

@author: N Ouben
"""

# Importing necessary libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download('punkt')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()


# Define specific combos of tokens that indicate negative or positive sentiments
negative_combinations = {
    ('not', 'fine'): 'negative',
    ('not', 'healthy'): 'negative',
    ('fetal', 'alcohol'): 'negative',
    ('died',): 'negative',
    ('no', 'amount'): 'negative',
    ('not', 'okay'): 'negative',
    ('FAS',): 'negative',
    ('unsafe',): 'negative',
    ('risk',): 'negative',
    ('dangerous',): 'negative',
    ('harm',): 'negative',
    ('damage',): 'negative',
    ('fail', 'placenta'): 'negative',
    ('preterm',): 'negative',
    ('harmed',): 'negative',
    ('wrong',): 'negative',
    ('complication',): 'negative',
    ('miscarriage',): 'negative',
    ('weight',): 'negative',
    ('underweight',): 'negative',
    ('small',): 'negative',
    ('diabetes',): 'negative',
    ('smoking',): 'negative',
    ('obese',): 'negative',
    ('adhd',): 'negative',
    ('hyperactivity',): 'negative',
    ('deficient',): 'negative',
    ('disease',): 'negative',
    ('sick',): 'negative',
    ('permanent','consequences'): 'negative',
    ('learning','problems'): 'negative',
    ('said','no'): 'negative',
    ('uneasy'): 'negative',
    ('no','safe'): 'negative',
    ('health','issues'): 'negative',
    ('low','iq'): 'negative',
    ('emotional','problems'): 'negative',
    ('dysplasia',): 'negative',
    ('respirator',): 'negative',
    ('overweight',): 'negative',
    ('unhealthy',): 'negative',
    ('violent',): 'negative',
    ('obesity',): 'negative'
}

positive_combinations = {
    ('healthy',): 'positive',
    ('fine',): 'positive',
    ('safe',): 'positive',
    ('good',): 'positive',
    ('great',): 'positive',
    ('perfect',): 'positive',
    ('amazing',): 'positive',
    ('unharmed',): 'positive',
    ('is','okay'): 'positive',
    ('was','okay'): 'positive',
    ('were','okay'): 'positive',
    ('is','ok'): 'positive',
    ('ok'): 'positive',
    ('well', 'balanced'): 'positive',
    ('no', 'problems'): 'positive',
    ('no', 'issues'): 'positive',
    ('smart',): 'positive',
    ('happy',): 'positive',
    ('no','problem'): 'positive',
    ('less','worried'): 'positive',
    ('glad'): 'positive',
    ('successful'): 'positive',
    ('positive'): 'positive',
    ('well'): 'positive'
}

# text preprocessing function
def text_preprocess(comment):
    # Suppression des balises HTML
    comment = re.sub(r'<[^>]+>', '', comment)
    # Tokenisation du texte
    tokens = word_tokenize(comment)
    # Conversion en minuscules
    tokens = [token.lower() for token in tokens]
    # Suppression de la ponctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [token.translate(table) for token in tokens]
    # Lemmatisation
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stripped if word]
    return lemmatized_words

