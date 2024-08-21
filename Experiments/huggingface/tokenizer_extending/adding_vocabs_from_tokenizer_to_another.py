from datasets import load_dataset

from transformers import AutoTokenizer, AutoModel

# pick the model type
tokenizer1 = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer2 = AutoTokenizer.from_pretrained("roberta-base")

print("Before adding roberta:", len(tokenizer1))

tokens_in_roberta_not_in_bert = set(tokenizer2.vocab).difference(tokenizer1.vocab)
tokenizer1.add_tokens(list(tokens_in_roberta_not_in_bert))

print("After adding roberta:", len(tokenizer1))


model = AutoModel.from_pretrained("bert-base-multilingual-cased")
model.resize_token_embeddings(len(tokenizer1))
