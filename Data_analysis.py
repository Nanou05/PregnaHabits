# -*- coding: utf-8 -*-
"""
Created on Apr 12 18:25:23 2024


@author: N Ouben
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import coo_matrix
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Handcrafted list of additionnal stopwords

additional_stopwords = {
    'like', 'know', 'well', 'said', 'also', 'one', 'get', 'would', 'u', 'told', 'think', 'could', 
    'keep', 'going', 'bad', 'time', 'asked', 'ob', 'half', 'glass', 'birthday', 'said', 'get', 
    'absorbed', 'placenta', 'sits', 'nah', 'took', 'sip', 'christmas', 'red', 'tri', 'talking', 
    'literal', 'le', 'oz', 'feel', 'strange', 'uneasy', 'full', 'free', 'version', 'miss', 'appeal', 
    'mil', 'doula', 'trying', 'suggest', 'end', 'basically', 'past', 'due', 'like', '23', 'oz', 'bath', 
    'help', 'relax', 'going', 'occasionally', '3rd', 'trimester', 'week', 'fine', 'first', 'huge', 
    'craving', 'mine', 'towards', 'end', 'check', 'book', 'expecting', 'better', 'pros', 'cons', 
    'many', 'taboo', 'helped', 'much', 'less', 'worried', 'exactly', '20', 'pages', 'actual', 'believe', 
    'worth', 'risk', 'love', 'celebrate', 'birth', 'breast', 'feed', 'drunk', 'mom', 'drank', 'third', 
    'tri', 'think', 'likely', 'recs', 'changed', 'since', '80s', 'fine', 'haha', 'midwife', 'thing', 
    'every', 'usually', 'month', 'nice', 'treat', 'never', 'done', 'recommend', 'however', 'gf', 'would', 
    'date', 'night', 'dinner', 'healthy', 'kiddos', '6', '3', 'issue', 'author', 'medical', 'doctor', 
    'food', 'scientist', 'either', 'economist', 'iirc', 'interview', 'expert', 'write', 'book', 'talk', 
    'statistics', 'thing', 'understanding', 'actual', 'science', 'shallow', 'read', 'excellent', 
    'article', 'woman', 'decided', 'follow', 'advice', 'listeria', 'ended', 'losing',  
    'someone', 'permanent', 'consequences', 'small', 'daughter', 'diagnosed', 'iugr', 'week', '3lb', '13oz', 'minor', 'contributed', 'failing', 'possibility', 'waiting', 'juice', 
    'instead', 'glad', 'thanksgiving', 'yesterday', 'interesting', 'conversation', 
    'obviously', 'recommend', 'discus', 'lack', 'solid', 'evidence', 'light', 'consumption', 'fa', 
    'associated', 'binge', 'drinking', 'known', 'rest', 'seems', 'body', 'gene', 'process', 'hard', 
    'pinpoint', 'expert', 'chosen', 'use', 'blanket', 'statement', 'precaution', 'rule', 'allow', 
    'eating', 'done', 'simply', 'feel', 'need', 'something', 'make', 'feel', 
    'far', 'thought', 'made', 'feel', 'good', 'easily', 'stayed', 'away', 'really', 'hot', 'subject', 
    'people', 'extremely', 'strong', 'opinion', 'relationship', 'known', 'ruin', 'everything', 'whether', 
    'willing', 'roll', 'dice', 'yes', 'hate', 'weird', 'smell', 'sometimes', 'taste', 'found', 'jelly', 
    'belly', 'flavored', 'bean', 'trick', 'enough', 'amount', 'completely', 'turned', 
    'okay', 'mean', 'learning', 'problem', 'later', 'road', 'even', 'please', 'advise', 'others', 
    'try', 'eating', 'thing', 'cooked', 'cheese', 'pretzel', 'brat', 'funny', 'craving', 'even', 
    'definitely', 'give', 'though', 'hate', 'fan', 'weekend', 'two', 'hard', 'day', 'especially', 
    'wow', 'really', 'use', 'enjoy', 'consumption', 'night', 'problem', 'mocktails', 'idea', 'lol', 
    'suggest', 'talking', 'midwife', 'answer', 'may', 'surprise', 'social', 'drinker', 'still', 'crave', 
    'tomorrow', 'symptom', 'subsided', 'normal', 'forget', 'pregnant', 'lol', 'nonalcoholic', 'option', 
    'stay', 'away', 'well', 'small', '05', 'baby', 'go', 'na', 'heineken', '0', 'found', 'content', 
    'still', 'every', 'couple', 'case', 'usually', 'grab', 'bubbly', 'water', 'carbonation', 'also', 
    'know', 'lessened', 'still', 'continued', 'several', 'cognitive', 'least', 'sound', 'fine', 
    'issue', 'worth', 'stick', 'dessert', 'missing', 'le', 'nauseous', 'went', 'week', 'husband', 
    'single', 'daynight', 'try', 'bunch', 'different', 'sad', 'bit', 'worth', 'stock', 'new', 'stuff', 
    'maybe', 'day', 'breastfeeding', 'probably', 'every', 'day', 'big', 'drinker', 'enjoy', 'daily', 
    'commenting', 'aching', 'months', 'bubbly', 'champagne', 'mixing', 'sparkling', 'water', 'oj', 
    'ice', 'donâ€™t', 'alcoholic', 'beverages', 'aside', 'beginning', 'still', 'felt', 'guilty', 'make', 
    'micheladas', 'mineral', 'today', 'decided', 'squirt', 'sounded', 'good', 'together', 'tasted', 
    'mimosa', 'completely', 'false', 'many', 'lady', 'last', 'board', 'december', '2021', 'looked', 
    'false', 'narrative', 'false', 'amount', '100', 'risking', 'rest', 'life', 'find', 'new', 'friends', 
    'sound', 'like', 'idiots', 'serious', 'dangerous', 'cant', 'stop', 'reach', 'review', 'think', 
    'whole', 'month', 'harsh', 'people', 'need', 'hard', 'truth', 'maybe', 'anyone', 'tell', 'truth', 
    'kick', 'butt', 'completely', 'false', 'sources', 'versus','gained','got', 'removed', 'gain', 'blood',
    'nt','mother', 'way','control','thank','girl','boy','study', 'quit','quitting','lot','year','long',
    'neither', 'one', 'parent', 'mom', 'knew','drink', 'drinking','pregnant','student', 'heavy', 'http', '/',
    'nurse','born', 'higher','diet', 'pregnancy','kid','pg','see','want','1','2','9','7','6','0',
    'child','lb','test','correlation', 'factor','gestational', 'comment','eat','female','chromosome','want','difference',
    'male','question','cause','point','started','effect','welcome','subreddit','development','research',
    'cooler','say','little','nbsp', 'thyroid','cold','ca','eat','shot','seriously','160','husband','wife','MIL',
    'old','young','change', 'already','second','friend','without','show','currently','linking',
    'x','gen','bc','anyway','look','dad','hormone','litterally','appointment','turkey','quite','although',
    'wanted','mention','mentionned','clearly','look','person','understand','dad','start','seen',
    'dr','rather','sex','came','easy','number','yet','guess','post','sugar','hour','able','doc','etc',
    'thougrouht','yeah','getting','always','whatever', 'technically','used','everyone','number','gave',
    'heard','ask','almost','actually','sample','op','gd','anyway','using','place','family','develop','dh',
    'put','act','example','ago','cig','totally','wait','making','around','must','let','often','tried','dont',
    'real','tried','prove','home','son','taking','totally','wanting','want','wo','entire','go','range','anything',
    'based','im','might','deal','thinking','throughout','seem','human','sister','reading','school','men','looking',
    'report','group','personally','mentioned','honestly','literally','back','oh','debate', 'page','ie','face','take',
    'thread','course','next','age','ago','absolutely','remember','google','call','ate','posted','telling','lifestyle','infant',
    'call','cut','certain','infant','telling','choose','similar','another','account','possible','work','action','pretty', 'deleted',
    'contact','automatically','sub','saying','guy', 'compare','comparable', 'gt','white','gender','drama','level', 'moderator','bot','data','result'
}


# Text cleaning function 
def further_clean_text(text):
    
    '''
    Parameters
    ----------
    text : string
        Comments' textual content.

    Returns
    -------
    string
        cleaned, lowercased, tokenized text.
    '''
    
    # Lowercasing
    words = re.findall(r'\b\w+\b', str(text).lower())
    # Removing the stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Removing the additionnal stopwords
    filtered_words = [word for word in words if word not in additional_stopwords]
    # Removing punctuations
    filtered_words = [word for word in filtered_words if word not in string.punctuation]

    return ' '.join(filtered_words)

def generate_word_cloud(text, title, file_path):
    
    '''
    Parameters
    ----------
    text : string
        cleaned comments' textual content.
    title : string
        comments' titles.
    file_path : string
        the path to the dataset.

    Returns
    -------
    a plot of a wordcloud.

    '''
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(file_path)
    plt.show()

def extract_topics(text_data, num_topics=5, num_words=10):
    
    '''
    Parameters
    ----------
    text_data : string
        cleaned comments' text.
    num_topics : int, optional
        Sets the number of topics to show. The default is 5.
    num_words : int, optional
        Sets the number of words present in a topic. The default is 10.

    Returns
    -------
    topics : string
        The prevalent topics in the text.
    '''
    
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    topics = []
    for index, topic in enumerate(lda.components_):
        topics.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]])

    return topics


def plot_cooccurrence_matrix(text_data, forum_name, top_n_words=20):
    
    '''
    Parameters
    ----------
    text_data : string
        comments' cleaned textual content.
    forum_name : string
        name of the forum where the data was scraped from.
    top_n_words : int, optional
        Number of words composing the co-occurrence matrix. The default is 20.

    Returns
    -------
    a plot of the co-occurrence matrix.

    '''
    
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(text_data)
    words = vectorizer.get_feature_names_out()

    # Calculate co-occurrence matrix
    cooccurrence_matrix = (dtm.T * dtm)
    cooccurrence_matrix.setdiag(0)
    cooccurrence_matrix = cooccurrence_matrix.toarray()

    # Normalize the co-occurrence matrix
    cooccurrence_matrix_normalized = cooccurrence_matrix / cooccurrence_matrix.max()

    # Apply log transformation
    cooccurrence_matrix_log = np.log1p(cooccurrence_matrix_normalized)

    # Get the top N words
    word_counts = np.array(dtm.sum(axis=0)).flatten()
    top_n_indices = word_counts.argsort()[-top_n_words:]
    top_words = [words[i] for i in top_n_indices]

    # Plot the co-occurrence matrix for the top N words
    top_cooccurrence_matrix = cooccurrence_matrix_log[top_n_indices][:, top_n_indices]

    plt.figure(figsize=(10, 10))
    plt.imshow(top_cooccurrence_matrix, cmap='viridis')
    plt.xticks(range(len(top_words)), top_words, rotation=90)
    plt.yticks(range(len(top_words)), top_words)
    plt.colorbar()
    plt.title(f'Co-occurrence Matrix for {forum_name}')
    plt.savefig(f'./occurence_analysis_output/adjusted_cooccurrence_matrix_{forum_name}.png')
    plt.show()

def plot_topics_wordclouds(topics, forum_name, num_topics=5):
    for i, topic in enumerate(topics):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(topic))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {i+1} for {forum_name}')
        plt.axis('off')
        plt.savefig(f'./occurence_analysis_output/topic_{i+1}_wordcloud_{forum_name}.png')
        plt.show()

def main():
    # Load the CSV files
    baby_center_df = pd.read_csv('./preprocessed_baby_center.csv')
    reddit_df = pd.read_csv('./preprocessed_reddit.csv')


    baby_center_df['Further Cleaned Comment'] = baby_center_df['Processed Comment'].apply(further_clean_text)
    reddit_df['Further Cleaned Comment'] = reddit_df['Processed Comment'].apply(further_clean_text)

    # Save the pretreated data
    baby_center_df.to_csv('./occurence_analysis_output/further_preprocessed_baby_center.csv', index=False)
    reddit_df.to_csv('./occurence_analysis_output/further_preprocessed_reddit.csv', index=False)

    # Generate the combined text for the word clouds
    combined_text_baby_center = ' '.join(baby_center_df['Further Cleaned Comment'].tolist())
    combined_text_reddit = ' '.join(reddit_df['Further Cleaned Comment'].tolist())

    # Generate and save the word clouds
    generate_word_cloud(combined_text_baby_center, 'Word Cloud for Baby Center Comments', './occurence_analysis_output/word_cloud_baby_center.png')
    generate_word_cloud(combined_text_reddit, 'Word Cloud for Reddit Comments', './occurence_analysis_output/word_cloud_reddit.png')

    # Extract the main topics
    baby_center_topics = extract_topics(baby_center_df['Further Cleaned Comment'].tolist())
    reddit_topics = extract_topics(reddit_df['Further Cleaned Comment'].tolist())

    # Show the topics
    print("Baby Center Topics:")
    for i, topic in enumerate(baby_center_topics):
        print(f"Topic {i+1}: {', '.join(topic)}")

    print("\nReddit Topics:")
    for i, topic in enumerate(reddit_topics):
        print(f"Topic {i+1}: {', '.join(topic)}")

    # Analyze the co-occurrences
    print("Co-occurrence analysis for Baby Center Comments:")
    plot_cooccurrence_matrix(baby_center_df['Further Cleaned Comment'].tolist(), 'Baby Center')

    print("Co-occurrence analysis for Reddit Comments:")
    plot_cooccurrence_matrix(reddit_df['Further Cleaned Comment'].tolist(), 'Reddit')

    # Visualize the topics with the word clouds
    plot_topics_wordclouds(baby_center_topics, 'Baby Center')
    plot_topics_wordclouds(reddit_topics, 'Reddit')


    print(baby_center_df[['Processed Comment', 'Further Cleaned Comment']].head())
    print(reddit_df[['Processed Comment', 'Further Cleaned Comment']].head())

    print("Additional preprocessing complete. Data saved to 'further_preprocessed_baby_center.csv' and 'further_preprocessed_reddit.csv'.")

if __name__ == "__main__":
    main()