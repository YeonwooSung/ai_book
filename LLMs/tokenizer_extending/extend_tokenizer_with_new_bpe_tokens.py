from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


# pick the model type
model_type = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModel.from_pretrained(model_type)

# Original vocab size.
print(len(tokenizer))
# Note the outputs are 100s indices which points to unknown tokens.
print(tokenizer("የቀን 7 ሰዓት አማርኛ ዜና ሚያዝያ"))


# Take the first 1000 sentences from cc100. 
am_dataset = load_dataset("cc100", "am")
am_train = iter(am_dataset['train'][i]['text'] for i in range(1000))

# Train a new tokenizer using the am_train and the old tokenizer object.
new_tokenizer = tokenizer.train_new_from_iterator(am_train, vocab_size=100_000)

tokenizer.add_tokens(list(new_tokenizer.vocab))

print(new_tokenizer("የቀን 7 ሰዓት አማርኛ ዜና ሚያዝያ"))
print(tokenizer("የቀን 7 ሰዓት አማርኛ ዜና ሚያዝያ"))
