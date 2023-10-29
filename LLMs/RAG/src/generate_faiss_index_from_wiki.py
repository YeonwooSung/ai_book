"""
This code is a copied version of @simjeg's code from:
<https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag/comments>

Credits to @simjeg for this code.
"""

import faiss
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk


wikipedia_path = Path("/path/to/270k_wikipiedia_stem_articles") # https://www.kaggle.com/datasets/mbanaei/all-paraphs-parsed-expanded
embedding_size = 384
batch_size = 256
max_length = 512
checkpoint = 'BAAI/bge-small-en-v1.5'
embedding_size = 384

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint).cuda().half()


def transform(batch):
    if 'BAAI' in checkpoint:
        batch["text"] = ["Represent this sentence for searching relevant passages: " + x for x in batch["text"]]
    elif checkpoint == "intfloat/e5-small-v2":
        batch["text"] = ["passage: "+ x for x in batch["text"]]

    tokens = tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=max_length)
    return tokens.to("cuda")


# Create faiss index, it will use the same index as wikipedia_index (not the "id", but the row index)
faiss_index = faiss.IndexFlatL2(embedding_size)

# Create dataset and dataloader
dataset = load_from_disk(wikipedia_path)    
dataset.set_transform(transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Compute embeddings
outputs = np.zeros((len(dataset), embedding_size), dtype=np.float16)
with torch.inference_mode():
    for i, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader)):
        embeddings = model(**batch).pooler_output
        embeddings = F.normalize(embeddings, p=2, dim=1)
        outputs[batch_size*i:batch_size*(i+1)] = embeddings.detach().cpu().numpy()


# Add embeddings to faiss index (it will use the same index as wiki_2023_index.parquet)
faiss_index.add(outputs)
faiss.write_index(faiss_index, str(wikipedia_path / f"faiss_index_{checkpoint.split('/')[-1]}.index"))
