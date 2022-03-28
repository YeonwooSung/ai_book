from transformers import AutoTokenizer

print('Encode text with tokenizer - 1) tokenize, 2) convert tokens to ids')

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)