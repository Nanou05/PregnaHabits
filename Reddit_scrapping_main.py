# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:59:28 2024

@author: N Ouben
"""

import datetime
import os
import sys
import logging
import warnings
import argparse
from reddit_scraper import scrape_reddit


warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_file(filepath):
    if not os.path.isfile(filepath):
        with open(filepath, 'w') as f:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reddit Scraper')
    parser.add_argument('--initial', action='store_true', help='Perform initial extraction (all posts)')
    args = parser.parse_args()

    logger = setup_logging()
    exact_titles = ['Did you drink alcohol while pregnant? How did your kids turn out?', 
                    'Smoking while pregnant', 
                    'Just found out my mom smoked cigarettes while pregnant with me', 
                    'Did you smoke during pregnancy and why?', 
                    'Anyone smoke weed while pregnant ?',
                    'Cannabis and pregnancy', 
                    'My friend said itâ€™s okay to smoke weed when pregnant', 
                    'Vaped while pregnant', 
                    'Drinking (wine )ðŸ· during pregnancy Did you ? How often...?', 
                    'Pregnant and alcoholic',
                    'How bad is drinking and smoking while pregnant?',
                    'Can alcohol intake during early pregnancy affect the embryo/fetus before the umbilical cord is developed? If so, how?', 
                    'Momsâ€™ Obesity in Pregnancy Is Linked to Lag in Sonsâ€™ Development and IQ',
                    'Pregnant women with obesity and diabetes may be more likely to have a child with ADHD',
                    'Obesity and Pregnancy', 
                    'Nervous about pregnancy while overweight',
                    'For those with a high BMI, how was your pregnancy experience?', 
                    'Pregnant and obese?',
                    'Obese and Pregnant'
                    ]
    
    limit = 100000  # Define a limit for the number of results

    id = 'GcxyVyMIziIDNbMMHxmZEg'
    secret = 'HJoN4azBDnlyEbVNn9soyxqIx-3cFA'
    agent = 'NLP_project'
    
    try:
        logger.info("Performing extraction (all posts)")
        comments = scrape_reddit(logger, exact_titles, id, secret, agent, limit)

        logger.info(f"Retrieved {len(comments)} comments")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    current_date = datetime.date.today().strftime("%Y-%m-%d")

    log_file = f"logs_{current_date}.log"
    error_file = f"error_{current_date}.log"

    log_dir = "./web_scrapping/reddit/scrapping/logs/"
    error_dir = "./web_scrapping/reddit/scrapping/errors/"
    data_dir = "./web_scrapping/reddit/scrapping/data/"

    create_directory(log_dir)
    create_directory(error_dir)
    create_directory(data_dir)

    log_path = os.path.join(log_dir, log_file)
    error_path = os.path.join(error_dir, error_file)

    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

    create_file(log_path)
    create_file(error_path)

    try:
        csv_file = f"extraction_{current_date}.csv"
        csv_file_path = os.path.join(data_dir, csv_file)

        comments.to_csv(csv_file_path, ",", index=False, mode='a', header=True, encoding="utf-8")

        os.chmod(csv_file_path, 0o444)

        logging.info(f'Saved results to {csv_file}')
        logging.info('Script execution completed.')

    except Exception as e:
        logging.exception(f'An error occurred: {str(e)}')

        with open(error_path, 'a') as error_log:
            error_log.write(f'{datetime.datetime.now()} - An error occurred: {str(e)}\n')

