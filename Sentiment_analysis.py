# -*- coding: utf-8 -*-
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
    #Deleting HTML tags
    comment = re.sub(r'<[^>]+>', '', comment)
    # Tokenizating text
    tokens = word_tokenize(comment)
    # lowercasing
    tokens = [token.lower() for token in tokens]
    # deleting punctuation marks
    table = str.maketrans('', '', string.punctuation)
    stripped = [token.translate(table) for token in tokens]
    # Lemmatization: optional
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stripped if word]
    return lemmatized_words

# Verify the specific combos of tokens
def check_token_combinations(tokens, combinations):
    
    '''
    Parameters
    ----------
    tokens : string
        preprocessed lemmatized words.
    combinations : dictonnary : string
        combinations of tokens for a better sentiment analysis.

    Returns
    -------
    bool
       verify the presence of token in the combination.

    '''
    
    for n in range(1, 3):  # Verify unigrams and bigrams
        for gram in ngrams(tokens, n):
            if gram in combinations:
                return True
    return False

# Apply TextBlob's sentiment analysis model
def textblob_sentiment_analysis(tokens):
    blob = TextBlob(' '.join(tokens))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Apply Vader's sentiment analysis model
def vader_sentiment_analysis(tokens):
    vs = analyzer.polarity_scores(' '.join(tokens))
    return vs['compound'], vs['pos'], vs['neu'], vs['neg']

# classify the sentiments depending on the calculated scores by each model
def classify_sentiment(vader_score, textblob_score, tokens):
    if check_token_combinations(tokens, negative_combinations):
        return 'negative'
    if check_token_combinations(tokens, positive_combinations):
        return 'positive'
    if vader_score >= 0.4 or textblob_score >= 0.4: # review scores
        return 'positive'
    elif vader_score <= -0.2 or textblob_score <= -0.2: # review scores
        return 'negative'
    else:
        return 'neutral'
    
# 
def main():
    # specify the filepaths
    baby_center_path = './data/baby_center_extraction_2024-04-12.csv'
    reddit_path = './data/extraction_2024-04-12.csv'

    # Load CSV files
    baby_center_df = pd.read_csv(baby_center_path)
    reddit_df = pd.read_csv(reddit_path)

    # Delete duplicates
    baby_center_df.drop_duplicates(inplace=True)
    reddit_df.drop_duplicates(inplace=True)

    # replacing missing values
    baby_center_df.fillna('', inplace=True)
    reddit_df.fillna('', inplace=True)

    # Applying the preprocessing function on the comments columns
    baby_center_df['Tokens'] = baby_center_df['Comment'].apply(text_preprocess)
    reddit_df['Tokens'] = reddit_df['Comment'].apply(text_preprocess)

    # Apply sentiment analysis with TextBlob & VADER
    baby_center_df['TextBlob_Polarity'], baby_center_df['TextBlob_Subjectivity'] = zip(*baby_center_df['Tokens'].apply(textblob_sentiment_analysis))
    reddit_df['TextBlob_Polarity'], reddit_df['TextBlob_Subjectivity'] = zip(*reddit_df['Tokens'].apply(textblob_sentiment_analysis))

    baby_center_df['VADER_Compound'], baby_center_df['VADER_Positive'], baby_center_df['VADER_Neutral'], baby_center_df['VADER_Negative'] = zip(*baby_center_df['Tokens'].apply(vader_sentiment_analysis))
    reddit_df['VADER_Compound'], reddit_df['VADER_Positive'], reddit_df['VADER_Neutral'], reddit_df['VADER_Negative'] = zip(*reddit_df['Tokens'].apply(vader_sentiment_analysis))

    # Sentiments classification
    baby_center_df['Sentiment'] = baby_center_df.apply(lambda row: classify_sentiment(row['VADER_Compound'], row['TextBlob_Polarity'], row['Tokens']), axis=1)
    reddit_df['Sentiment'] = reddit_df.apply(lambda row: classify_sentiment(row['VADER_Compound'], row['TextBlob_Polarity'], row['Tokens']), axis=1)

    # Save the data
    baby_center_df.to_csv('./sentiment_analysis_output/sentiment_processed_baby_center_adjusted.csv', index=False)
    reddit_df.to_csv('./sentiment_analysis_output/sentiment_processed_reddit_adjusted.csv', index=False)

    # Show the first 5 comments for verification
    print(baby_center_df[['Comment', 'Tokens', 'TextBlob_Polarity', 'TextBlob_Subjectivity', 'VADER_Compound', 'VADER_Positive', 'VADER_Neutral', 'VADER_Negative', 'Sentiment']].head())
    print(reddit_df[['Comment', 'Tokens', 'TextBlob_Polarity', 'TextBlob_Subjectivity', 'VADER_Compound', 'VADER_Positive', 'VADER_Neutral', 'VADER_Negative', 'Sentiment']].head())

    print("Sentiment analysis complete. Data saved to 'sentiment_processed_baby_center_adjusted.csv' and 'sentiment_processed_reddit_adjusted.csv'.")


    # Results' details
    def sentiment_details(df, dataset_name):
        positive = df[df['Sentiment'] == 'positive'].shape[0]
        neutral = df[df['Sentiment'] == 'neutral'].shape[0]
        negative = df[df['Sentiment'] == 'negative'].shape[0]
        total = df.shape[0]
        print(f"\nSentiment distribution for {dataset_name}:")
        print(f"Positive: {positive} ({positive/total:.2%})")
        print(f"Neutral: {neutral} ({neutral/total:.2%})")
        print(f"Negative: {negative} ({negative/total:.2%})")
        print(f"Average TextBlob Polarity: {df['TextBlob_Polarity'].mean():.2f}")
        print(f"Average VADER Compound: {df['VADER_Compound'].mean():.2f}")

    sentiment_details(baby_center_df, 'Baby Center')
    sentiment_details(reddit_df, 'Reddit')

    # Show the 1st 5 pos, neg and neutral comments
    print("\nExamples of positive comments (Baby Center) :")
    print(baby_center_df[baby_center_df['Sentiment'] == 'positive']['Comment'].head(5).tolist())

    print("\nExamples of neutral comments (Baby Center) :")
    print(baby_center_df[baby_center_df['Sentiment'] == 'neutral']['Comment'].head(5).tolist())

    print("\nExamples of negative comments(Baby Center) :")
    print(baby_center_df[baby_center_df['Sentiment'] == 'negative']['Comment'].head(5).tolist())

    print("\nExamples of positive comments (Reddit) :")
    print(reddit_df[reddit_df['Sentiment'] == 'positive']['Comment'].head(5).tolist())

    print("\nExamples of neutral comments (Reddit) :")
    print(reddit_df[reddit_df['Sentiment'] == 'neutral']['Comment'].head(5).tolist())

    print("\nExamples of negative comments (Reddit) :")
    print(reddit_df[reddit_df['Sentiment'] == 'negative']['Comment'].head(5).tolist())


    # Pie chart of sentiments' distribution
    def plot_sentiment_distribution(df, dataset_name, colors):
        sentiment_counts = df['Sentiment'].value_counts(normalize=True)
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                startangle=140, colors=colors, textprops={'fontweight': 'bold'})
        plt.title(f'Distribution of sentiments for {dataset_name}')
        plt.axis('equal')
        plt.savefig(f'./sentiment_analysis_output/sentiment_distribution_{dataset_name}.png')
        plt.show()

    # Histogram of polarity scores with Textblob
    def plot_polarity_histogram(df, dataset_name):
        plt.figure(figsize=(10, 6))
        plt.hist(df['TextBlob_Polarity'], bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of polarity scores TextBlob for {dataset_name}')
        plt.xlabel('Polarity Score')
        plt.ylabel('Frequence')
        plt.savefig(f'./sentiment_analysis_output/polarity_histogram_{dataset_name}.png')
        plt.show()

    # Bar plot showing the proportion of each sentiment in both datasets
    def plot_proportion_bar_chart(df1, df2, label1, label2):
        sentiments = ['positive', 'neutral', 'negative']
        df1_proportions = df1['Sentiment'].value_counts(normalize=True)
        df2_proportions = df2['Sentiment'].value_counts(normalize=True)

        df1_proportions = df1_proportions.reindex(sentiments, fill_value=0)
        df2_proportions = df2_proportions.reindex(sentiments, fill_value=0)

        bar_width = 0.35
        index = range(len(sentiments))

        plt.figure(figsize=(10, 6))
        plt.bar(index, df1_proportions, bar_width, label=label1, color='#0072B2')
        plt.bar([i + bar_width for i in index], df2_proportions, bar_width, label=label2, color='#56B4E9')
        
        plt.xlabel('Sentiment')
        plt.ylabel('Proportion of comments')
        plt.title('Proportion of comments between baby center and reddit')
        plt.xticks([i + bar_width / 2 for i in index], sentiments)
        for i, v in enumerate(df1_proportions):
            plt.text(i - 0.1, v + 0.02, f"{v:.1%}", color='black', fontweight='bold')
        for i, v in enumerate(df2_proportions):
            plt.text(i + bar_width - 0.1, v + 0.02, f"{v:.1%}", color='black', fontweight='bold')
        plt.legend()
        plt.savefig('./sentiment_analysis_output/proportion_bar_chart.png')
        plt.show()

    # radar chart of polarity scores: textblob and vader
    def plot_radar_chart(df1, df2, label1, label2):
        labels = ['TextBlob Polarity', 'TextBlob Subjectivity', 'VADER Compound',
                  'VADER Positive', 'VADER Neutral', 'VADER Negative']
        stats1 = [
            df1['TextBlob_Polarity'].mean(),
            df1['TextBlob_Subjectivity'].mean(),
            df1['VADER_Compound'].mean(),
            df1['VADER_Positive'].mean(),
            df1['VADER_Neutral'].mean(),
            df1['VADER_Negative'].mean()
        ]
        stats2 = [
            df2['TextBlob_Polarity'].mean(),
            df2['TextBlob_Subjectivity'].mean(),
            df2['VADER_Compound'].mean(),
            df2['VADER_Positive'].mean(),
            df2['VADER_Neutral'].mean(),
            df2['VADER_Negative'].mean()
        ]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        stats1 += stats1[:1]
        stats2 += stats2[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles, stats1, color='#0072B2', alpha=0.25)
        ax.fill(angles, stats2, color='#56B4E9', alpha=0.25)
        ax.plot(angles, stats1, color='#0072B2', linewidth=2, label=label1)
        ax.plot(angles, stats2, color='#56B4E9', linewidth=2, label=label2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.title('Comparing the scores between Baby Center & Reddit')
        plt.savefig('./sentiment_analysis_output/radar_chart.png')
        plt.show()

    colors = ['#0072B2', '#56B4E9', '#92C5DE']

    # Create and save the graphics: Baby Center
    plot_sentiment_distribution(baby_center_df, 'Baby Center', colors=colors)
    plot_polarity_histogram(baby_center_df, 'Baby Center')
    plot_proportion_bar_chart(baby_center_df, reddit_df, 'Baby Center', 'Reddit')
    
    # Create and save the graphics: Baby Center
    plot_sentiment_distribution(reddit_df, 'Reddit', colors=colors)
    plot_polarity_histogram(reddit_df, 'Reddit')
    plot_radar_chart(baby_center_df, reddit_df, 'Baby Center', 'Reddit')


if __name__ == "__main__":
    main()
