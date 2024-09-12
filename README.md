# PregnaHabits

## Study of effects of some unhealthy Life styles during pregnancy on the heath of the infant


### Description of the Python script

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