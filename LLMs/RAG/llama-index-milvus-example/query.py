from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import MilvusVectorStore
from langchain import OpenAI
import openai
import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

def chatbot(input_text):

    vector_store = MilvusVectorStore(
       host = "localhost",
       port = "19530",
       collection_name = "webscrape"
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine()
    
    response = query_engine.query(input_text)
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Sample Content Querying")

iface.launch(share=True)
