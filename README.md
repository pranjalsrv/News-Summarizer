### CYBINT News Summarizer

This project consists of multiple components:
- Topic Modelling: An algorithm that models the topics consisting in the news article corpora which makes it easier so that people can set preferences and get preferred news articles first. Works as a subtype of search engine as well, semantically searching through topics.
- Abstractive Summarizer: A summarization model that summarizes the news article.
- Lateral News API: Search a query and get recommended news with summaries

Instructions to run summarizer web app:
1. Clone the repo
2. Open bash, cd to CYBINT-news-summarizer/API folder
3. Run command: <br />
     - `uvicorn T5_api:app`  (for T5 summarizer) 
     - `uvicorn bart_api:app` (for BART summarizer) 
4. Go to localhost:8000 in your favorite browser. Type in headline and content of the article to be summarized.
5. Press submit


Instructions to run LDA web app:
1. Clone the repo
2. Open up 3 bash terminals. 
3. First Terminal-> cd to CYBINT-news-summarizer/Semantic Similarity
4. Run command: <br /> `uvicorn semantic_similarity_API:app --port 8081`
5. Second Terminal -> cd to CYBINT-news-summarizer/Topic Modelling
6. Run command:  <br /> `uvicorn lda2vec_API:app --port 8082`
7. Third Terminal -> cd to CYBINT-news-summarizer/Combiner
8. Run command:  <br /> `uvicorn semantic_topic_model_api:app --port 5000`
9. Fire up your favorite browser and go to localhost:5000
10. Type in the topic you want to search and press submit

Instructions to run Lateral API:
1. Clone the repo
2. Open bash, cd to CYBINT-news-summarizer/API/Lateral API
3. Run command: <br /> `uvicorn lateral_testing:app --port 5000`
4. Make requests using query 

Instructions on running the ElasticSearch Pipeline (Docker Implementation)
1. Clone the repo
2. Open 3 terminals
3. First Terminal-> cd to ElasticSearch Pipeline/
4. Run command: <br /> `docker pull docker.elastic.co/elasticsearch/elasticsearch:7.9.1`
5. Run command (For dev, single node deployment): <br /> `docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.9.1`
6. Second Terminal-> cd to ElasticSearch Pipeline/
7. Run command: <br /> `python updater.py`
8. Third Terminal-> cd to ElasticSearch Pipeline/
9. Run command: <br /> `uvicorn app:app --port 5000`
10. Make requests to the API now!
