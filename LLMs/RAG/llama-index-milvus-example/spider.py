from llama_index import VectorStoreIndex, StorageContext, download_loader
from llama_index.readers.schema.base import Document
from llama_index.vector_stores import MilvusVectorStore
import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

crawlUrl = "https://docs.apify.com/academy/web-scraping-for-beginners"

with open("spidersettings.json", "r") as f:
   params = json.load(f)

vector_store = MilvusVectorStore(
   host = "localhost",
   port = "19530",
   collection_name = "webscrape"
)

def transform_dataset_item(item):
    return Document(
        text=item.get("text"),
        extra_info={
            "url": item.get("url"),
        },
    )

ApifyActor = download_loader("ApifyActor")
reader = ApifyActor(os.environ["APIFY_API_TOKEN"])
docs = reader.load_data(
    actor_id="apify/website-content-crawler",
    dataset_mapping_function=transform_dataset_item,
    run_input={"startUrls": [{"url": crawlUrl}], **params}
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("What is this documentation about?")
print(response)
