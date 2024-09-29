from sentence_transformers import SentenceTransformer,models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.losses import DenoisingAutoEncoderLoss
import torch
from torch.utils.data import DataLoader
import re


data = [
    'TSDAE (Tranformer and Sequential Denoising AutoEncoder) is one of most popular unsupervised training method. The main idea is reconstruct the original sentence from a corrupted sentence. The TSDAE model consists of two parts: an encoder and a decoder.',
    ' TSDAE encodes corrupted sentences into fixed-sized vectors and requires the decoder to reconstruct the original sentences from this sentence embedding.'
]
sentences = []


def generate_sentence(jd):
    spliter = re.compile(r'\.\s?\n?')
    list_of_sentences = spliter.split(jd)
    if len(sentences)<100_000:
        sentences.extend([i for i in list_of_sentences if len(i)>30])


for sent in data:
    generate_sentence(sent)

print('number of sentence',len(sentences))
def clean_sentence(text):
    text = text.lower()
    text = re.sub("[^ A-Za-z0-9.&,\-]"," ",text)
    text = re.sub(' +',' ',text)
    return text
sentences = [clean_sentence(i) for i in sentences]

# The DenoisingAutoEncoderDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence] 
# add noise in traning data
train_data = DenoisingAutoEncoderDataset(sentences)
loader = DataLoader(train_data,batch_size=4,shuffle=True)

gte_model = models.Transformer('thenlper/gte-base')
polling = models.Pooling(gte_model.get_word_embedding_dimension(),'cls')
model = SentenceTransformer(modules = [gte_model,polling])
loss = DenoisingAutoEncoderLoss(model,tie_encoder_decoder = True)

model.fit([(loader,loss)],
    epochs=1,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True
)
model.save('output/gte-base-fine-tune')
