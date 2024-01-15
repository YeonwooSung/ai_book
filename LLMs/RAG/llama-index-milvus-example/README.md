# llama-index-milvus-example
Open AI APIs with Llama Index and Milvus Vector DB for RAG testing

Welcome to my ridiculously simple web scraping -> Retrieval Augmented Generation (RAG) example. This example walks through:

* Scraping a website with Apify webcrawler (requires a free account)
* Generating embeddings with OpenAI APIs (requires an OpenAI account)
* Loading embeddings into a Milvus vector store (Dockerfile included)
* Spinning up a Gradio chat to ask questions of your LLM with context plugged in

Getting it going is fairly easy. I used `pipenv` but use whatever environment you'd like.

1. cd milvus
2. sudo docker compose up -d
3. cd ..
4. pipenv install
5. cp .env.sample .env
6. Fill in your OpenAI token and Apify token
7. Open spider.py and fill in the URL you want to crawl
8. python spider.py
9. python query.py

Once Milvus is installed you can open localhost:8000 to log into Attu to view your setup, collections, and browse the vector store.

Enjoy!
