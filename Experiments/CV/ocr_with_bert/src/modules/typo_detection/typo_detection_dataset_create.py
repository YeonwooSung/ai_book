import pickle
import random
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from typo_detection_dataset import TypoDataset

words = []
labels = []
sources = ['amazon_medium', 'imdb_medium']
for source in sources:
    words.extend(pickle.load(open(f"../../data/files_pickle/words_{source}.pickle", "rb")))
    labels.extend(pickle.load(open(f"../../data/files_pickle/labels_{source}.pickle", "rb")))

data = list(zip(words, labels))
random.shuffle(data)
words, labels = zip(*data)
print(f'Number of sentences in dataset: {len(words)}')

random_idx = random.randint(0, len(words))
random_idx = 22
random_words = words[random_idx]
random_labels = labels[random_idx]
for word, label in zip(random_words, random_labels):
    print("%-30s %-1s" % (word, label))

plt.style.available

# +
labels_lens = [len(l) for l in labels]

plt.style.use('fast')
plt.figure(figsize=(10,6))
plt.hist(labels_lens, bins=25, color='orange')
plt.xlabel('Number of words in text')
plt.ylabel('Number of texts')
plt.savefig('Number_of_sentences.jpg')
plt.show()


# +
labels_one = sum([sum(l) for l in labels])
labels_all = sum([len(l) for l in labels]) 

print(f'Percentage of misspelled words in dataset: {round(labels_one / labels_all, 4)}')

# +
ds = TypoDataset()
tokenized_words, tokenized_labels = ds._tokenize_and_preserve_labels(random_words, random_labels)
tokenized_ids = ds.tokenizer.convert_tokens_to_ids(tokenized_words)

print("%-30s %-30s %-1s" % ("WORD", "TOKENIZER ID", "LABEL"))
print('-' * 80)
for word, idx, label in zip(tokenized_words, tokenized_ids, tokenized_labels):
    print("%-30s %-30s %-1s" % (word, idx, label))
# -

zip_list = list(zip(words, labels))
random.shuffle(zip_list)
words, labels = zip(*zip_list)

train_part = 20000
val_part = 4000
modes = ['train', 'val']
words_m = [words[:train_part], words[train_part:(train_part + val_part)]]
labels_m = [labels[:train_part], labels[train_part:(train_part + val_part)]]

for mode, word, label in zip(modes, words_m, labels_m):
    ds = TypoDataset(mode=mode)
    inp, tg, msk = ds.prepare_dataset(word, label, out_path='../../data/typo_ds/amazon_imdb_big_20k_4k')

print("%-10s %-10s %-10s" % ("WORD_ID", "WORD_LABEL", "WORD_MASK"))
for inp_t, tg_t, msk_t in zip(inp[22], tg[22], msk[22]):
    print("%-10s %-10s %-10s" % (inp_t.item(), tg_t.item(), msk_t.item()))


