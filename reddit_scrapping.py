# -*- coding: utf-8 -*-
"""
Created on Apr 12 12:15:53 2024

@author: N Ouben
"""


import praw
import pandas as pd
import datetime
import logging

def scrape_reddit(logging, exact_titles, identifiant, secret, agent, limit):
    r = praw.Reddit(client_id=identifiant, client_secret=secret, user_agent=agent)
    comments_df = pd.DataFrame(columns=['id', 'Subreddit', 'Title', 'Post Author', 'Flair', 'Post Date', 'Comment Author', 'Comment Date', 'Comment'])

    for title in exact_titles:
        logging.info(f'Searching for exact title: {title}')
        query = f'title:"{title}"'
        submissions = r.subreddit('all').search(query, sort='new', time_filter='all')

        for submission in submissions:
            logging.info(f'Checking submission: {submission.title} (created: {datetime.datetime.fromtimestamp(submission.created_utc)})')
            if title.lower() == submission.title.lower():
                logging.info(f'Matched submission: {submission.title}')
                post_created_at = datetime.datetime.fromtimestamp(submission.created_utc)
                post_author = submission.author.name if submission.author else "Unknown"
                flair = submission.link_flair_text if submission.link_flair_text else "No Flair"
                
                # Retrieve all comments
                submission.comments.replace_more(limit=None)
                for comment in submission.comments.list():
                    if hasattr(comment, 'body'):
                        comment_created_at = datetime.datetime.fromtimestamp(comment.created_utc)
                        comment_author = comment.author.name if comment.author else "Unknown"
                        new_row = pd.DataFrame([{
                            'id': submission.id, 
                            'Subreddit': submission.subreddit.display_name, 
                            'Title': submission.title,
                            'Post Author': post_author, 
                            'Flair': flair, 
                            'Post Date': post_created_at,
                            'Comment Author': comment_author,
                            'Comment Date': comment_created_at,
                            'Comment': comment.body
                        }])
                        comments_df = pd.concat([comments_df, new_row], ignore_index=True)

    return comments_df

