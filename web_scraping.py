# -*- coding: utf-8 -*-
"""
Created on Apr 12 11:56:11 2024

@author: N Ouben
"""

# Importing necessary libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import os
import json
from urllib.parse import urljoin, urlparse



def get_json_ld(soup):
    json_ld_script = soup.find('script', type='application/ld+json')
    if json_ld_script:
        try:
            json_ld = json.loads(json_ld_script.string)
            return json_ld
        except json.JSONDecodeError:
            print("Failed to parse JSON-LD script.")
    return None

def get_post_details_from_json_ld(json_ld):
    if not json_ld or '@graph' not in json_ld:
        print("JSON-LD not found or invalid format.")
        return None

    for entry in json_ld['@graph']:
        if entry['@type'] == 'DiscussionForumPosting':
            title = entry.get('headline', 'No title found').strip()
            author = entry['author'].get('name', 'No author found').strip()
            date = entry.get('datePublished', 'No date found')
            date = datetime.datetime.fromisoformat(date[:-1]).strftime('%Y-%m-%d %H:%M:%S')
            post_content = entry.get('text', 'No content found').strip()
            comments = entry.get('comment', [])
            formatted_comments = []
            for comment in comments:
                comment_author = comment['author'].get('name', 'No comment author').strip()
                comment_date = comment.get('datePublished', 'No comment date')
                comment_date = datetime.datetime.fromisoformat(comment_date[:-1]).strftime('%Y-%m-%d %H:%M:%S')
                comment_content = comment.get('text', 'No comment content').strip()
                formatted_comments.append((comment_author, comment_date, comment_content))
            return {
                'title': title,
                'author': author,
                'date': date,
                'post_content': post_content,
                'comments': formatted_comments
            }
    return None

def get_next_page_url(soup, base_url):
    pagination_links = soup.find_all('a', href=True)
    for link in pagination_links:
        if 'next' in link.text.lower():
            next_page_url = link['href']
            if not next_page_url.startswith('http'):
                next_page_url = urljoin(base_url, next_page_url)
            print(f"Found next page link: {next_page_url}")
            return next_page_url
    print("No next page link found.")
    return None

def get_comments_from_post(post_url):
    all_comments = []
    visited_urls = set()
    base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(post_url))
    next_url = post_url

    with requests.Session() as session:
        while next_url and next_url not in visited_urls:
            print(f"Fetching post URL: {next_url}")
            response = session.get(next_url)
            if response.status_code != 200:
                print(f"Failed to fetch post URL: {next_url} with status code {response.status_code}")
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            json_ld = get_json_ld(soup)
            post_details = get_post_details_from_json_ld(json_ld)

            if post_details:
                for comment in post_details['comments']:
                    all_comments.append({
                        'Title': post_details['title'],
                        'Post Author': post_details['author'],
                        'Post Date': post_details['date'],
                        'Post Content': post_details['post_content'],
                        'Comment Author': comment[0],
                        'Comment Date': comment[1],
                        'Comment': comment[2]
                    })

            visited_urls.add(next_url)
            next_url = get_next_page_url(soup, base_url)
            if not next_url or next_url in visited_urls:
                print("No next page found, ending pagination.")
                break

    return all_comments

def scrape_multiple_posts(post_urls):
    all_data = []
    for post_url in post_urls:
        post_data = get_comments_from_post(post_url)
        all_data.extend(post_data)

    if all_data:
        return pd.DataFrame(all_data)
    else:
        return pd.DataFrame()

# List of specific post URLs to scrape
post_urls = [
    'https://community.babycenter.com/post/a77520091/red-wine', 
    'https://community.babycenter.com/post/a77858239/beer-in-pregnancy',
    'https://community.babycenter.com/post/a77603427/glass-of-wine-while-pregnant',
    'https://community.babycenter.com/post/a77359251/drinking-alcoholics-beverages?cid=2608280222',
    'https://community.babycenter.com/post/a38279209/i_cheated...._with_a_wine_cooler',
    'https://community.babycenter.com/post/a33523408/i_drank_sangria',
    'https://community.babycenter.com/post/a4720855/are_overweight_woman_who_are_pgttc_selfish',
    'https://community.babycenter.com/post/a77241718/35-and-pregnant?cid=2606945041',
    'https://community.babycenter.com/post/a77956666/dieting-during-pregnancy-high-bmi',
    'https://community.babycenter.com/post/a77533335/smoking-during-pregnancy',
    'https://community.babycenter.com/post/a56649325/smoking_weed_during_pregnancy',
    'https://community.babycenter.com/post/a77432424/smoking-and-pregnancy',
    'https://community.babycenter.com/post/a77368426/smoking',
    'https://community.babycenter.com/post/a77876189/smoking'
]

# Scrape the discussions
df = scrape_multiple_posts(post_urls)

# Display the retrieved data
print(df.head())

current_date = datetime.date.today().strftime("%Y-%m-%d")

# Specify the directory where to save the CSV file
directory = './scrapped_data'  # Replace with the desired directory path
if not os.path.exists(directory):
    os.makedirs(directory)

csv_file = f"baby_center_extraction_{current_date}.csv"
csv_file_path = os.path.join(directory, csv_file)

# Save the data to a CSV file
df.to_csv(csv_file_path, index=False)

print(f"Saved results to: {csv_file_path} as {csv_file}")
print('Script execution completed.')
