# -*- coding: utf-8 -*-
"""
Created on Apr 12 12:38:31 2024

@author: N Ouben
"""

# Importing libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Pretreatment of extracted comments with NLTK
def nltk_preprocess(comment):
    # Deleting HTML tags
    comment = re.sub(r'<[^>]+>', '', comment)
    # Word Tokenization
    tokens = word_tokenize(comment)
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    # Deleting punctuation marks
    table = str.maketrans('', '', string.punctuation)
    stripped = [token.translate(table) for token in tokens]
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in stripped if word not in stop_words]
    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    # Return the clean corpus
    return ' '.join([f"'{word}'" for word in lemmatized_words])

def main():
    # Specify the path where to save the CSV file
    baby_center_path = './scrapped_data/baby_center_extraction_2024-04-12.csv'
    reddit_path = './scrapped_data//extraction_2024-04-12.csv'

    # Loading the CSV files
    baby_center_df = pd.read_csv(baby_center_path)
    reddit_df = pd.read_csv(reddit_path)

    # Deleting the duplicates
    baby_center_df.drop_duplicates(inplace=True)
    reddit_df.drop_duplicates(inplace=True)

    # Processing the missing values
    baby_center_df.fillna('', inplace=True)
    reddit_df.fillna('', inplace=True)

    # Applying the preprocessing code on the comment columns of both dataframes
    baby_center_df['Processed Comment'] = baby_center_df['Comment'].apply(nltk_preprocess)
    reddit_df['Processed Comment'] = reddit_df['Comment'].apply(nltk_preprocess)

    # Save the preprocessed data
    baby_center_df.to_csv('./preprocessed_baby_center.csv', index=False)
    reddit_df.to_csv('./preprocessed_reddit.csv', index=False)

    # Show the 1st preprocessed comments of both dataframes
    print(baby_center_df['Processed Comment'].head())
    print(reddit_df['Processed Comment'].head())

    print("Preprocessing complete. Data saved to 'preprocessed_baby_center.csv' and 'preprocessed_reddit.csv'.")

if __name__ == "__main__":
    main()
