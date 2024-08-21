import requests
from transformers import AutoTokenizer, AutoModel


response = requests.get('https://unicode.org/Public/emoji/13.0/emoji-sequences.txt')

with open('emoji.txt', 'w') as fout:
    for line in response.content.decode('utf8').split('\n'):
        if line.strip() and not line.startswith('#'):
            hexa = line.split(';')[0]
            hexa = hexa.split('..')            
            if len(hexa) == 1:
                ch = ''.join([chr(int(h, 16)) for h in hexa[0].strip().split(' ')])
                print(ch, end='\n', file=fout)
            else:
                start, end = hexa
                for ch in range(int(start, 16), int(end, 16)+1):
                    #ch = ''.join([chr(int(h, 16)) for h in ch.split(' ')])
                    print(chr(ch), end='\n', file=fout)


# pick the model type
model_type = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModel.from_pretrained(model_type)

# add emojis
new_tokens = [e.strip() for e in open('emoji.txt')] 

# check if the tokens are already in the vocabulary
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens))

# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))
