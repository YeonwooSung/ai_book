from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

print(config)


# Different model instantiation method
bert_model = BertModel.from_pretrained("bert-base-cased")


# save model
path = './save_path'
model.save_pretrained(path)  # this will save config.json and pytorch_model.bin in the given directory.