import os

from transformers import BertTokenizer
from typo_detection_train import prepare_dataloader, train_detector
from typo_detection_model import TypoDetectorBERT

source_name = 'amazon_imdb_big_20k_4k'

output_dir = f'../../data/typo_models/{source_name}'
data_dir = f'../../data/typo_ds/{source_name}'
epochs = 5
lr = 3e-5

dl_train = prepare_dataloader(inputs_path=os.path.join(data_dir, 'inputs_train.pt'),
                              labels_path=os.path.join(data_dir, 'labels_train.pt'),
                              masks_path=os.path.join(data_dir, 'masks_train.pt'),
                              batch_size=8,
                              mode='train')

dl_val= prepare_dataloader(inputs_path=os.path.join(data_dir, 'inputs_val.pt'),
                           labels_path=os.path.join(data_dir, 'labels_val.pt'),
                           masks_path=os.path.join(data_dir, 'masks_val.pt'),
                           batch_size=16,
                           mode='val')

dataloaders = {'train': dl_train,
               'val': dl_val}

tokenizer = BertTokenizer.from_pretrained(data_dir)
model = TypoDetectorBERT()

train_detector(model,
               tokenizer,
               dataloaders,
               output_dir,
               epochs,
               lr,
               wandb_log=False)
