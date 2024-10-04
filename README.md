# PregnaHabits

## Project Overview

PregnaHabits is a research project aimed at studying the effects of certain unhealthy lifestyle choices during pregnancy on infant health. This project utilizes various data collection and analysis techniques, including web scraping, Reddit data extraction, sentiment analysis, and machine learning models.

## Project Goal

The primary goal of PregnaHabits is to investigate and analyze the potential impacts of unhealthy lifestyle habits during pregnancy on the health outcomes of infants. By leveraging diverse data sources and advanced analytical techniques, this project seeks to provide insights that could inform prenatal care practices and public health initiatives.

## Project Structure

The project consists of the following files:

1. `Articles_extraction.py`: Script for extracting relevant articles from web sources.
2. `Data_analysis.py`: Main script for analyzing the collected data.
3. `Reddit_scraping_main.py`: Main script for scraping data from Reddit.
4. `Sentiment_analysis.py`: Script for performing sentiment analysis on the collected text data.
5. `model.py`: Contains the machine learning models used in the project.
6. `reddit_scraping.py`: Contains functions for scraping data from Reddit.
7. `requirements.txt`: List of Python dependencies required for the project.
8. `web_data_preprocessing.py`: Script for preprocessing data collected from web sources.
9. `web_scraping.py`: Contains functions for general web scraping tasks.

## Setup and Installation

To set up the project environment:

1. Clone this repository to your local machine.
2. Ensure you have Python 3.7+ installed.
3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

To run the project:

1. Start with data collection by running the web scraping and Reddit scraping scripts:

   ```
   python web_scraping.py
   python Reddit_scraping_main.py
   ```

2. Preprocess the collected data:

   ```
   python web_data_preprocessing.py
   ```

3. Perform sentiment analysis:

   ```
   python Sentiment_analysis.py
   ```

4. Run the main data analysis script:

   ```
   python Data_analysis.py
   ```

5. If using machine learning models, run:

   ```
   python model.py
   ```

## Description of the Python scripts

#### `articles_extraction.py`

- **Objective**: This script performs a comprehensive analysis of scientific articles related to children's health and the long-term effects of pregnant women's habits (smoking, alcohol, obesity) on the baby's health.

- **Main functions**:

  1. **fetch_article_ids(query, max_results=100)**:
     - Formulates the query and retrieves the IDs of relevant articles from the PubMed database.

     - **Output**: List of article IDs.

  2. **fetch_article_details(ids)**:
     - Retrieves the details of the articles using the IDs.

     - **Output**: Article details in XML format.

  3. **preprocess_text(text)**:
     - Performs text preprocessing, including converting to lowercase, removing punctuation, tokenization, stopword removal, and lemmatization.

     - **Output**: Preprocessed text.

  4. **get_bert_embeddings(texts)**:
     - Generates embeddings for the preprocessed texts using the BERT model.

     - **Output**: List of embeddings.

  5. **plot_cooccurrence_matrix(text_data, Articles, top_n_words=20)**:
     - Displays the co-occurrence matrix of the most frequent words in the preprocessed texts.

     - **Output**: Visualization of the co-occurrence matrix.

  6. **generate_word_cloud(text, title)**:
     - Generates a word cloud for the combined text.

     - **Output**: Visualized word cloud.

  7. **extract_topics(texts, num_topics=5, num_words=10)**:
     - Extracts topics from the preprocessed texts using the LDA (Latent Dirichlet Allocation) model.

     - **Output**: List of topics and associated words.

  8. **plot_topics_wordclouds(topics, title_prefix)**:
     - Visualizes the topics with word clouds.

     - **Output**: Word clouds for each topic.

- **Outputs**:
  - `output/tfidf_abstracts.csv`: Contains the TF-IDF matrix of the articles.
  - Visualizations generated in the script (word clouds, co-occurrence matrices, etc.).

## Contributing

Contributions to PregnaHabits are welcome. Please fork the repository and submit a pull request with your proposed changes.

## Ethics and Privacy

This project involves collecting and analyzing sensitive data related to pregnancy and infant health. All data collection and usage complies with relevant privacy laws and ethical guidelines (GDPR). Personal data is not collected and any collected information won't be shared on this repo.

## License
(yet to be defined)
