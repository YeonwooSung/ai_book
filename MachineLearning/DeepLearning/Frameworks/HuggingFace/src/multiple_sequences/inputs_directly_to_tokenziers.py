from transformers import AutoTokenizer


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Huggingface tokenizers could handle the pure string inputs

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)

print(model_inputs)

# It also handles multiple sequences at a time, with no change in the API

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences)

print(model_inputs)
